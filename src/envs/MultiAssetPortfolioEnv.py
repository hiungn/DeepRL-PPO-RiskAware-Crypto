"""
MultiAssetPortfolioEnv.py
Gymnasium environment cho multi-asset cryptocurrency portfolio management.

MDP Formulation:
  State  s_t = [X_{t-W:t} in R^{6 x n x W}] concat [w_t in Delta^{n+cash}]
               concat [NAV_t/NAV_0] concat [loss_signal] concat [drawdown]
  Action a_t in R^{n+cash}  ->  softmax(a_t) = target weights w_t in Delta^{n+cash}
  Reward r_t = log(NAV_t/NAV_{t-1}) - lambda_risk * risk_t - lambda_turnover * turnover
               - lambda_drawdown * drawdown^2

  use_cash=True (default):
    Last weight = cash allocation (return=0, risk=0)
    Agent can "park" capital in cash during drawdown periods

  No-Trade Zone:
    Turnover is computed against PRICE-DRIFTED weights (not raw old weights).
    If turnover < rebalance_threshold -> HOLD (no trade, fee=0).
    This prevents fee bleeding from trivial rebalancing.

  risk_t: downside_std (sortino_style) | total_std (sharpe_style) | 0 (raw)

Episode design: random start offset to expose agent to diverse market regimes.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


FEATURE_COLS = ["Close", "MA7", "MA21", "RSI", "MACD", "Volatility"]
N_FEATURES   = len(FEATURE_COLS)   # 6


class MultiAssetPortfolioEnv(gym.Env):
    """
    Parameters
    ----------
    data_dict           : dict {ticker: pd.DataFrame}  - output of data_utils.load_all_data()
    reward_type         : "sortino_style" | "sharpe_style" | "raw"
    window_size         : int, lookback window W (default 30)
    episode_len         : int, steps per episode (default 365 = 1 trading year)
    fee                 : float, transaction cost per unit of turnover (default 0.001 = 0.1%)
    lambda_risk         : float, risk penalty coefficient (default 0.5)
    lambda_turnover     : float, explicit turnover penalty in reward (default 0.0015)
    lambda_drawdown     : float, quadratic drawdown penalty coefficient (default 2.0)
    rebalance_threshold : float, min turnover to trigger trade (default 0.05 = 5%)
    risk_window         : int, rolling window for risk estimation (default 14)
    reward_clip         : float, clip reward to [-reward_clip, reward_clip]
    initial_nav         : float, starting portfolio value
    use_cash            : bool, whether to include a risk-free cash asset (return=0)
    softmax_temperature : float, temperature for softmax (lower = sharper allocation, default 0.5)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_dict:           dict,
        reward_type:         str   = "sortino_style",
        window_size:         int   = 30,
        episode_len:         int   = 365,
        fee:                 float = 0.001,
        lambda_risk:         float = 0.5,
        lambda_turnover:     float = 0.0015,
        lambda_drawdown:     float = 2.0,
        rebalance_threshold: float = 0.05,
        risk_window:         int   = 14,
        reward_clip:         float = 5.0,
        initial_nav:         float = 10_000.0,
        use_cash:            bool  = True,
        softmax_temperature: float = 0.5,
    ):
        super().__init__()

        assert reward_type in ("sortino_style", "sharpe_style", "raw"), \
            f"reward_type must be 'sortino_style', 'sharpe_style', or 'raw', got: {reward_type}"

        self.reward_type         = reward_type
        self.window_size         = window_size
        self.episode_len         = episode_len
        self.fee                 = fee
        self.lambda_risk         = lambda_risk
        self.lambda_turnover     = lambda_turnover
        self.lambda_drawdown     = lambda_drawdown
        self.rebalance_threshold = rebalance_threshold
        self.risk_window         = risk_window
        self.reward_clip         = reward_clip
        self.initial_nav         = initial_nav
        self.use_cash            = use_cash
        self.softmax_temperature = softmax_temperature

        # Build aligned price matrix: shape (T, n_assets, n_features)
        tickers = sorted(data_dict.keys())
        self.tickers  = tickers
        self.n_assets = len(tickers)

        # n_actions = n_assets + 1 (cash) if use_cash else n_assets
        self.n_actions = self.n_assets + (1 if use_cash else 0)

        frames = [data_dict[t][FEATURE_COLS].values for t in tickers]
        self.price_data = np.stack(frames, axis=1).astype(np.float32)  # (T, n_assets, 6)
        self.T = self.price_data.shape[0]

        # CloseRaw: raw prices (always positive) for log return calculation
        # Do NOT use z-scored Close (can be negative -> log(negative) = NaN)
        close_matrix = np.stack(
            [data_dict[t]["CloseRaw"].values for t in tickers], axis=1
        ).astype(np.float32)   # (T, n_assets)
        self.close_matrix = close_matrix

        # Observation space:
        # [price_window: W * n_assets * 6] + [weights: n_actions] + [nav_ratio: 1]
        # + [loss_signal: 1] + [drawdown: 1]
        obs_dim = window_size * self.n_assets * N_FEATURES + self.n_actions + 1 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: R^n_actions, bounded for SB3 Gaussian policy
        # softmax maps any R^n to the simplex Delta^{n-1}
        # [-4, 4] is wide enough: softmax([-4,4]) covers the full simplex practically
        self.action_space = spaces.Box(
            low=-4.0, high=4.0, shape=(self.n_actions,), dtype=np.float32
        )

        # Internal state
        self.current_step        = 0
        self.episode_start       = 0
        self.weights             = self._initial_weights()
        self.nav                 = float(initial_nav)
        self._return_history: list = []
        self._consecutive_losses = 0
        self._peak_nav           = float(initial_nav)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Random start offset: expose agent to diverse market regimes
        max_offset = max(0, self.T - self.window_size - self.episode_len - 1)
        self.episode_start = self.window_size + (
            self.np_random.integers(0, max_offset + 1) if max_offset > 0 else 0
        )
        self.current_step        = self.episode_start
        self.weights             = self._initial_weights()
        self.nav                 = float(self.initial_nav)
        self._return_history     = []
        self._consecutive_losses = 0
        self._peak_nav           = float(self.initial_nav)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # 1. Softmax: R^n -> simplex Delta^{n-1}
        target_weights = self._softmax(action)

        # 2. Compute price-drifted weights (what weights become after price change)
        t_prev = self.current_step - 1
        t_curr = self.current_step

        close_prev = self.close_matrix[t_prev]   # (n_assets,)
        close_curr = self.close_matrix[t_curr]   # (n_assets,)

        price_ratios = close_curr / (close_prev + 1e-12)  # (n_assets,)
        if self.use_cash:
            price_ratios = np.append(price_ratios, 1.0)  # cash price unchanged

        drifted_values  = self.weights * price_ratios
        drifted_weights = drifted_values / (drifted_values.sum() + 1e-12)
        drifted_weights = drifted_weights.astype(np.float32)

        # 3. No-Trade Zone: compare target vs drifted weights
        turnover = float(np.sum(np.abs(target_weights - drifted_weights)))
        rebalanced = turnover >= self.rebalance_threshold

        if rebalanced:
            new_weights = target_weights
            cost = self.fee * turnover
        else:
            new_weights = drifted_weights
            turnover = 0.0
            cost = 0.0

        # 4. Portfolio log return (using new_weights after rebalance decision)
        asset_log_returns = np.log(close_curr / (close_prev + 1e-12))  # (n_assets,)
        if self.use_cash:
            asset_log_returns = np.append(asset_log_returns, 0.0)  # cash return = 0

        portfolio_log_return = float(np.dot(new_weights, asset_log_returns))

        # 5. Update NAV
        old_nav  = self.nav
        self.nav = old_nav * np.exp(portfolio_log_return) * (1.0 - cost)
        nav_log_return = np.log(self.nav / (old_nav + 1e-12))

        # 6. Update peak NAV and consecutive losses
        self._peak_nav = max(self._peak_nav, self.nav)

        if portfolio_log_return < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # 7. Rolling risk (computed on crypto returns only, not cash)
        crypto_weights = new_weights[:self.n_assets]
        crypto_w_sum   = crypto_weights.sum()
        if crypto_w_sum > 1e-6:
            crypto_return = float(np.dot(
                crypto_weights / crypto_w_sum,
                asset_log_returns[:self.n_assets]
            ))
        else:
            crypto_return = 0.0
        self._return_history.append(crypto_return)
        if len(self._return_history) > self.risk_window:
            self._return_history.pop(0)

        risk_t = self._compute_rolling_risk()

        # 8. Reward: log return - risk penalty - turnover penalty - drawdown penalty
        #    Drawdown penalty: quadratic in drawdown fraction
        #    When drawdown=10% -> penalty = lambda_dd * 0.01
        #    When drawdown=30% -> penalty = lambda_dd * 0.09 (9x stronger)
        #    This creates strong incentive to shift to cash during large drawdowns
        current_drawdown = max(0.0, 1.0 - self.nav / (self._peak_nav + 1e-12))
        dd_penalty = self.lambda_drawdown * (current_drawdown ** 2)

        reward = nav_log_return \
                 - self.lambda_risk * risk_t \
                 - self.lambda_turnover * turnover \
                 - dd_penalty
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # 9. Advance state
        self.weights      = new_weights
        self.current_step += 1

        episode_end = self.episode_start + self.episode_len
        terminated  = self.current_step >= min(episode_end, self.T - 1)
        truncated   = False

        # Cash allocation for reporting
        cash_weight = float(new_weights[-1]) if self.use_cash else 0.0

        info = {
            "nav":              float(self.nav),
            "portfolio_return": portfolio_log_return,
            "risk":             risk_t,
            "turnover":         turnover,
            "cost":             cost,
            "weights":          self.weights.copy(),
            "cash_weight":      cash_weight,
            "rebalanced":       rebalanced,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        w_str = ", ".join(f"{t}={self.weights[i]:.3f}" for i, t in enumerate(self.tickers))
        if self.use_cash:
            w_str += f", CASH={self.weights[-1]:.3f}"
        print(f"step={self.current_step:4d}  NAV={self.nav:10.2f}  [{w_str}]")

    # Internals

    def _initial_weights(self) -> np.ndarray:
        """Start 100% in cash if available, else equal-weight."""
        w = np.zeros(self.n_actions, dtype=np.float32)
        if self.use_cash:
            w[-1] = 1.0  # 100% cash
        else:
            w[:] = 1.0 / self.n_actions  # equal weight
        return w

    def _get_obs(self) -> np.ndarray:
        """
        Observation = flatten(price_window) + current_weights + nav_ratio
                    + loss_signal + current_drawdown
        """
        start  = self.current_step - self.window_size
        end    = self.current_step
        window = self.price_data[start:end]   # (W, n_assets, 6)
        window = np.clip(window, -10.0, 10.0)  # safety clip for outliers

        # Consecutive loss signal: exponential decay in [0, 1)
        loss_signal = 1.0 - np.exp(-0.5 * self._consecutive_losses)

        # Current drawdown from peak NAV: [0, 1)
        current_drawdown = 1.0 - self.nav / (self._peak_nav + 1e-12)
        current_drawdown = max(0.0, current_drawdown)

        obs = np.concatenate([
            window.flatten(),
            self.weights,                       # n_actions values (includes cash if use_cash)
            [self.nav / self.initial_nav],
            [loss_signal],
            [current_drawdown],
        ]).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax with temperature scaling.
        Lower temperature -> sharper distribution (more concentrated).
        tau=1.0: standard softmax.  tau=0.5: agent can reach ~90%+ in one asset.
        """
        x_scaled = x / self.softmax_temperature
        e = np.exp(x_scaled - np.max(x_scaled))
        return (e / e.sum()).astype(np.float32)

    def _compute_rolling_risk(self) -> float:
        """
        Rolling risk estimate from crypto return history.

        sortino_style: std of negative returns only (downside deviation)
                       -> penalizes only downside, consistent with Sortino ratio
        sharpe_style:  std of all returns (total deviation)
        raw:           0 (no risk penalty)
        """
        if len(self._return_history) < 2:
            return 0.0

        returns = np.array(self._return_history)

        if self.reward_type == "sortino_style":
            downside = returns[returns < 0.0]
            return 0.0 if downside.size < 2 else float(np.std(downside))

        elif self.reward_type == "sharpe_style":
            return float(np.std(returns))

        else:   # "raw"
            return 0.0
