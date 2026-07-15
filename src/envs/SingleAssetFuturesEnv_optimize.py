"""
SingleAssetFuturesEnv_optimize.py
Gymnasium environment for single-asset perpetual futures trading with RL.

Key differences from MultiAssetPortfolioEnv.py:
  - Single asset (e.g. BTC) instead of 3-asset portfolio
  - Long AND short positions via tanh action space [-1, +1]
  - Leverage support (configurable, default 2x)
  - Perpetual futures funding rate cost
  - Margin accounting with forced liquidation
  - CVaR (Conditional Value-at-Risk) penalty in reward
  - Slippage model

MDP Formulation:
  State  s_t = [X_{t-W:t} in R^{8 x W}]  (8 features, W=30 lookback)
               concat [position, unrealized_pnl, nav_ratio, margin_ratio,
                       drawdown, loss_signal, time_in_position]
  Action a_t in [-1, +1]  (target position: -1=max short, 0=flat, +1=max long)
  Reward r_t = pnl - lambda_risk * risk - lambda_turn * |delta_pos|
               - lambda_dd * dd^2 - lambda_cvar * cvar - funding_cost

Episode design: random start offset for diverse market regime exposure.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


FEATURE_COLS = [
    "Close", "MA7", "MA21", "RSI", "MACD", "Volatility",
    "Volume_zscore", "Funding_rate",
]
N_FEATURES = len(FEATURE_COLS)  # 8


class SingleAssetFuturesEnv(gym.Env):
    """
    Parameters
    ----------
    price_df            : pd.DataFrame with columns FEATURE_COLS + ["CloseRaw", "Funding_rate"]
    reward_type         : "sortino_style" | "sharpe_style" | "raw"
    window_size         : int, lookback window W (default 30)
    episode_len         : int, steps per episode (default 365)
    leverage            : float, position leverage multiplier (default 2.0)
    taker_fee           : float, futures taker fee per unit notional (default 0.0004 = 0.04%)
    slippage_bps        : float, slippage in basis points (default 1.0)
    funding_base_rate   : float, daily funding rate base (default 0.0003 = 0.03%/day)
    maintenance_margin  : float, min margin ratio before liquidation (default 0.05 = 5%)
    liquidation_penalty : float, penalty on NAV when liquidated (default 0.01 = 1%)
    lambda_risk         : float, risk penalty coefficient (default 0.3)
    lambda_turnover     : float, position change penalty (default 0.001)
    lambda_drawdown     : float, quadratic drawdown penalty (default 2.0)
    lambda_cvar         : float, CVaR tail-risk penalty (default 0.5)
    cvar_alpha          : float, CVaR quantile (default 0.05 = worst 5%)
    position_threshold  : float, no-trade zone threshold (default 0.05)
    risk_window         : int, rolling window for risk estimation (default 14)
    reward_clip         : float, clip reward to [-clip, clip] (default 5.0)
    initial_nav         : float, starting account value (default 10000.0)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_df,
        reward_type:         str   = "sortino_style",
        window_size:         int   = 30,
        episode_len:         int   = 365,
        leverage:            float = 2.0,
        taker_fee:           float = 0.0004,
        slippage_bps:        float = 1.0,
        funding_base_rate:   float = 0.0003,
        maintenance_margin:  float = 0.05,
        liquidation_penalty: float = 0.01,
        lambda_risk:         float = 0.3,
        lambda_turnover:     float = 0.001,
        lambda_drawdown:     float = 2.0,
        lambda_cvar:         float = 0.5,
        cvar_alpha:          float = 0.05,
        position_threshold:  float = 0.05,
        risk_window:         int   = 14,
        reward_clip:         float = 5.0,
        initial_nav:         float = 10_000.0,
    ):
        super().__init__()

        assert reward_type in ("sortino_style", "sharpe_style", "raw"), \
            f"reward_type must be 'sortino_style', 'sharpe_style', or 'raw', got: {reward_type}"

        self.reward_type         = reward_type
        self.window_size         = window_size
        self.episode_len         = episode_len
        self.leverage            = leverage
        self.taker_fee           = taker_fee
        self.slippage_bps        = slippage_bps
        self.funding_base_rate   = funding_base_rate
        self.maintenance_margin  = maintenance_margin
        self.liquidation_penalty = liquidation_penalty
        self.lambda_risk         = lambda_risk
        self.lambda_turnover     = lambda_turnover
        self.lambda_drawdown     = lambda_drawdown
        self.lambda_cvar         = lambda_cvar
        self.cvar_alpha          = cvar_alpha
        self.position_threshold  = position_threshold
        self.risk_window         = risk_window
        self.reward_clip         = reward_clip
        self.initial_nav         = initial_nav

        # ── Build price arrays from DataFrame ──
        self.feature_data = price_df[FEATURE_COLS].values.astype(np.float32)  # (T, 8)
        self.close_raw    = price_df["CloseRaw"].values.astype(np.float64)    # (T,)
        self.T            = len(self.close_raw)

        # Funding rate from data (already computed in data_utils_optimize)
        if "Funding_rate" in price_df.columns:
            self.funding_rates = price_df["Funding_rate"].values.astype(np.float64)
        else:
            self.funding_rates = np.zeros(self.T, dtype=np.float64)

        # ── Observation space ──
        # [price_window: W * 8] + [position, unrealized_pnl, nav_ratio,
        #                          margin_ratio, drawdown, loss_signal,
        #                          time_in_position] = 7
        obs_dim = window_size * N_FEATURES + 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # ── Action space: single scalar [-1, +1] ──
        # -1 = max short, 0 = flat, +1 = max long
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        # ── Internal state ──
        self.current_step        = 0
        self.episode_start       = 0
        self.position            = 0.0      # current position in [-1, 1]
        self.entry_price         = 0.0      # price at which current position was opened
        self.nav                 = float(initial_nav)
        self._return_history: list = []
        self._consecutive_losses = 0
        self._peak_nav           = float(initial_nav)
        self._time_in_position   = 0        # steps since last position change
        self._num_liquidations   = 0
        self._total_funding_paid = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        max_offset = max(0, self.T - self.window_size - self.episode_len - 1)
        self.episode_start = self.window_size + (
            self.np_random.integers(0, max_offset + 1) if max_offset > 0 else 0
        )
        self.current_step        = self.episode_start
        self.position            = 0.0
        self.entry_price         = self.close_raw[self.current_step]
        self.nav                 = float(self.initial_nav)
        self._return_history     = []
        self._consecutive_losses = 0
        self._peak_nav           = float(self.initial_nav)
        self._time_in_position   = 0
        self._num_liquidations   = 0
        self._total_funding_paid = 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # ── 1. Parse target position from action ──
        target_position = float(np.clip(action[0], -1.0, 1.0))

        t_prev = self.current_step - 1
        t_curr = self.current_step

        price_prev = self.close_raw[t_prev]
        price_curr = self.close_raw[t_curr]

        # ── 2. Compute unrealized PnL on existing position ──
        log_return = np.log(price_curr / (price_prev + 1e-12))
        unrealized_pnl = self.position * log_return * self.leverage

        # ── 3. No-Trade Zone: compare target vs current position ──
        position_change = abs(target_position - self.position)
        rebalanced = position_change >= self.position_threshold

        if rebalanced:
            new_position = target_position
        else:
            new_position = self.position
            position_change = 0.0

        # ── 4. Transaction costs (only if position changes) ──
        if rebalanced:
            notional_change = position_change * self.leverage
            fee_cost  = self.taker_fee * notional_change
            slip_cost = (self.slippage_bps / 10_000) * notional_change
            total_trade_cost = fee_cost + slip_cost
        else:
            total_trade_cost = 0.0

        # ── 5. Funding rate cost ──
        # Perpetual futures: position holder pays/receives funding
        funding_rate = self.funding_rates[t_curr] if t_curr < len(self.funding_rates) else 0.0
        funding_cost = abs(self.position) * funding_rate * self.leverage
        self._total_funding_paid += funding_cost

        # ── 6. Update NAV ──
        old_nav  = self.nav
        self.nav = old_nav * (1.0 + unrealized_pnl) - old_nav * (total_trade_cost + funding_cost)
        self.nav = max(self.nav, 1.0)  # floor at $1 to avoid negative NAV

        nav_return = (self.nav - old_nav) / (old_nav + 1e-12)

        # ── 7. Check liquidation ──
        margin_used = abs(new_position) * self.nav / (self.leverage + 1e-12)
        margin_ratio = margin_used / (self.nav + 1e-12) if abs(new_position) > 1e-6 else 0.0

        liquidated = False
        if abs(new_position) > 1e-6 and self.nav < margin_used * self.maintenance_margin:
            # Forced liquidation: close position, apply penalty
            liquidated = True
            self._num_liquidations += 1
            self.nav *= (1.0 - self.liquidation_penalty)
            new_position = 0.0
            position_change = abs(self.position)  # full close

        # ── 8. Update position state ──
        if abs(new_position - self.position) > 1e-6:
            self._time_in_position = 0
            self.entry_price = price_curr
        else:
            self._time_in_position += 1

        self.position = new_position

        # ── 9. Update peak NAV and consecutive losses ──
        self._peak_nav = max(self._peak_nav, self.nav)

        if nav_return < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # ── 10. Rolling return history ──
        self._return_history.append(nav_return)
        if len(self._return_history) > self.risk_window:
            self._return_history.pop(0)

        # ── 11. Compute reward components ──
        risk_t = self._compute_rolling_risk()
        cvar_t = self._compute_cvar()

        current_drawdown = max(0.0, 1.0 - self.nav / (self._peak_nav + 1e-12))
        dd_penalty = self.lambda_drawdown * (current_drawdown ** 2)

        reward = nav_return \
                 - self.lambda_risk * risk_t \
                 - self.lambda_turnover * position_change \
                 - dd_penalty \
                 - self.lambda_cvar * cvar_t \
                 - funding_cost

        # Extra penalty if liquidated
        if liquidated:
            reward -= 1.0

        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # ── 12. Advance state ──
        self.current_step += 1

        episode_end = self.episode_start + self.episode_len
        terminated  = self.current_step >= min(episode_end, self.T - 1)
        truncated   = False

        info = {
            "nav":               float(self.nav),
            "position":          float(self.position),
            "nav_return":        nav_return,
            "unrealized_pnl":    unrealized_pnl,
            "risk":              risk_t,
            "cvar":              cvar_t,
            "turnover":          position_change,
            "trade_cost":        total_trade_cost,
            "funding_cost":      funding_cost,
            "rebalanced":        rebalanced,
            "liquidated":        liquidated,
            "margin_ratio":      margin_ratio,
            "drawdown":          current_drawdown,
            "num_liquidations":  self._num_liquidations,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pos_str = f"{'LONG' if self.position > 0 else 'SHORT' if self.position < 0 else 'FLAT'}"
        print(
            f"step={self.current_step:4d}  NAV={self.nav:10.2f}  "
            f"pos={self.position:+.3f} ({pos_str})  "
            f"dd={1.0 - self.nav / (self._peak_nav + 1e-12):.2%}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Observation = flatten(price_window) + position_info (7 values)

        Total: 30 * 8 + 7 = 247 dimensions
        """
        start = self.current_step - self.window_size
        end   = self.current_step
        window = self.feature_data[start:end]  # (30, 8)
        window = np.clip(window, -10.0, 10.0)

        # Unrealized PnL as fraction of entry
        if abs(self.position) > 1e-6 and self.entry_price > 1e-6:
            price_now = self.close_raw[min(self.current_step, self.T - 1)]
            unrealized_pnl_pct = self.position * (price_now / self.entry_price - 1.0) * self.leverage
        else:
            unrealized_pnl_pct = 0.0

        # NAV ratio
        nav_ratio = self.nav / self.initial_nav

        # Margin ratio
        margin_used = abs(self.position) * self.nav / (self.leverage + 1e-12)
        margin_ratio = margin_used / (self.nav + 1e-12) if abs(self.position) > 1e-6 else 0.0

        # Consecutive loss signal
        loss_signal = 1.0 - np.exp(-0.5 * self._consecutive_losses)

        # Current drawdown
        current_drawdown = max(0.0, 1.0 - self.nav / (self._peak_nav + 1e-12))

        # Normalized time in position (saturates around 30 days)
        time_in_pos = 1.0 - np.exp(-0.1 * self._time_in_position)

        obs = np.concatenate([
            window.flatten(),                # 240
            [self.position],                 # 1: current position [-1, +1]
            [np.clip(unrealized_pnl_pct, -5.0, 5.0)],  # 1: unrealized PnL %
            [nav_ratio],                     # 1: NAV / NAV_0
            [margin_ratio],                  # 1: margin usage
            [current_drawdown],              # 1: drawdown from peak
            [loss_signal],                   # 1: consecutive loss signal
            [time_in_pos],                   # 1: time in current position
        ]).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _compute_rolling_risk(self) -> float:
        """Rolling risk estimate: downside std (sortino) or total std (sharpe)."""
        if len(self._return_history) < 2:
            return 0.0

        returns = np.array(self._return_history)

        if self.reward_type == "sortino_style":
            downside = returns[returns < 0.0]
            return 0.0 if downside.size < 2 else float(np.std(downside))
        elif self.reward_type == "sharpe_style":
            return float(np.std(returns))
        else:
            return 0.0

    def _compute_cvar(self) -> float:
        """
        Conditional Value-at-Risk: expected loss in the worst alpha-quantile.

        CVaR captures tail risk better than standard deviation.
        For alpha=0.05: average loss in the worst 5% of days.
        Returns 0 if not enough data or no tail losses.
        """
        if len(self._return_history) < self.risk_window:
            return 0.0

        returns = np.array(self._return_history)
        threshold = np.percentile(returns, self.cvar_alpha * 100)
        tail = returns[returns <= threshold]

        if len(tail) == 0:
            return 0.0

        # Return positive value (penalty magnitude)
        return max(0.0, -float(np.mean(tail)))
