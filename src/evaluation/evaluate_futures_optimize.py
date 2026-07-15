"""
Evaluation metrics and backtest for single-asset futures RL agent.

Differences from evaluate_portfolio.py:
  - Single-asset futures (long/short) instead of multi-asset portfolio
  - New metrics: long/short/flat %, liquidations, funding paid, win rate, profit factor
  - New baselines: long-hold, short-hold, momentum, random
  - Position history plot (long=green, short=red, flat=gray)
  - Loads VecNormalize stats alongside model for consistent evaluation
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.SingleAssetFuturesEnv_optimize import SingleAssetFuturesEnv


# ──────────────────────────────────────────────────────────────
# Metric functions (reused from original + new futures metrics)
# ──────────────────────────────────────────────────────────────

def compute_sharpe(returns: np.ndarray, rf: float = 0.0, freq: int = 365) -> float:
    excess = returns - rf / freq
    if np.std(excess) < 1e-10:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(freq))


def compute_sortino(returns: np.ndarray, rf: float = 0.0, freq: int = 365) -> float:
    mar  = rf / freq
    down = returns[returns < mar]
    if len(down) < 2:
        return 0.0
    sigma_d = float(np.std(down))
    if sigma_d < 1e-10:
        return 0.0
    return float((np.mean(returns) - mar) / sigma_d * np.sqrt(freq))


def compute_max_drawdown(nav_series: np.ndarray) -> float:
    running_max = np.maximum.accumulate(nav_series)
    drawdowns   = 1.0 - nav_series / (running_max + 1e-12)
    return float(np.max(drawdowns))


def compute_calmar(ann_return: float, mdd: float) -> float:
    return 0.0 if mdd < 1e-10 else float(ann_return / mdd)


def compute_annual_return(nav_series: np.ndarray, freq: int = 365) -> float:
    total = nav_series[-1] / nav_series[0] - 1
    n_yrs = len(nav_series) / freq
    return 0.0 if n_yrs < 1e-3 else float((1 + total) ** (1 / n_yrs) - 1)


def compute_all_metrics(
    nav_series:    np.ndarray,
    position_log:  list = None,
    trade_log:     list = None,
    freq:          int  = 365,
) -> dict:
    """Compute full metric suite including futures-specific metrics."""
    returns   = np.diff(np.log(nav_series + 1e-12))
    total_ret = nav_series[-1] / nav_series[0] - 1
    ann_ret   = compute_annual_return(nav_series, freq)
    sharpe    = compute_sharpe(returns, freq=freq)
    sortino   = compute_sortino(returns, freq=freq)
    mdd       = compute_max_drawdown(nav_series)
    calmar    = compute_calmar(ann_ret, mdd)

    m = {
        "total_return":  round(total_ret * 100, 2),
        "annual_return": round(ann_ret * 100, 2),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "max_drawdown":  round(mdd * 100, 2),
        "calmar":        round(calmar, 3),
    }

    # ── Futures-specific metrics ──
    if position_log is not None:
        positions = np.array(position_log)
        n = len(positions)
        m["long_pct"]          = round(float(np.sum(positions > 0.05)) / max(n, 1) * 100, 1)
        m["short_pct"]         = round(float(np.sum(positions < -0.05)) / max(n, 1) * 100, 1)
        m["flat_pct"]          = round(float(np.sum(np.abs(positions) <= 0.05)) / max(n, 1) * 100, 1)
        m["avg_position_size"] = round(float(np.mean(np.abs(positions))), 3)

    if trade_log is not None:
        # Win rate and profit factor from per-trade PnL
        trade_pnls = np.array(trade_log)
        if len(trade_pnls) > 0:
            wins   = trade_pnls[trade_pnls > 0]
            losses = trade_pnls[trade_pnls < 0]
            m["num_trades"]    = len(trade_pnls)
            m["win_rate"]      = round(float(len(wins)) / max(len(trade_pnls), 1) * 100, 1)
            gross_profit = float(np.sum(wins))  if len(wins) > 0   else 0.0
            gross_loss   = float(np.sum(np.abs(losses))) if len(losses) > 0 else 1e-12
            m["profit_factor"] = round(gross_profit / gross_loss, 3)

    return m


# ──────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────

def run_backtest(
    model_path:      str,
    test_df:         pd.DataFrame,
    reward_type:     str   = "sortino_style",
    initial_nav:     float = 10_000.0,
    use_lstm:        bool  = False,
    leverage:        float = 2.0,
    lambda_risk:     float = 0.3,
    lambda_drawdown: float = 2.0,
    lambda_cvar:     float = 0.5,
    vec_norm_path:   str   = None,
) -> tuple:
    """
    Backtest a trained PPO model on test data.

    Returns
    -------
    nav_arr       : np.ndarray  - NAV series
    metrics       : dict        - all metrics
    position_log  : list        - position at each step
    """
    if use_lstm:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path)
    else:
        model = PPO.load(model_path)

    n_days = len(test_df)
    env = SingleAssetFuturesEnv(
        price_df        = test_df,
        reward_type     = reward_type,
        episode_len     = n_days,
        leverage        = leverage,
        lambda_risk     = lambda_risk,
        lambda_drawdown = lambda_drawdown,
        lambda_cvar     = lambda_cvar,
        initial_nav     = initial_nav,
    )

    # Fixed start (deterministic backtest)
    obs, _ = env.reset()
    env.episode_start       = env.window_size
    env.current_step        = env.window_size
    env.position            = 0.0
    env.entry_price         = env.close_raw[env.current_step]
    env.nav                 = float(initial_nav)
    env._return_history     = []
    env._consecutive_losses = 0
    env._peak_nav           = float(initial_nav)
    env._time_in_position   = 0
    env._num_liquidations   = 0
    env._total_funding_paid = 0.0
    obs = env._get_obs()

    nav_series    = [initial_nav]
    position_log  = [0.0]
    trade_pnl_log = []      # PnL per completed trade
    trade_count   = 0
    total_funding = 0.0
    num_liquidations = 0

    # Track per-trade PnL
    _trade_entry_nav = initial_nav

    # LSTM state
    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)

    terminated = truncated = False
    while not (terminated or truncated):
        if use_lstm:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True,
            )
            episode_start = np.array([terminated or truncated])
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, info = env.step(action)

        nav_series.append(info["nav"])
        position_log.append(info["position"])
        total_funding += info["funding_cost"]

        if info.get("liquidated", False):
            num_liquidations += 1

        if info.get("rebalanced", False):
            # Record PnL of completed trade segment
            trade_pnl = info["nav"] - _trade_entry_nav
            trade_pnl_log.append(trade_pnl)
            _trade_entry_nav = info["nav"]
            trade_count += 1

    nav_arr = np.array(nav_series, dtype=float)
    metrics = compute_all_metrics(nav_arr, position_log, trade_pnl_log)

    # Add trading activity metrics
    n_steps = len(position_log) - 1
    metrics["trading_frequency"] = round(trade_count / max(n_steps, 1) * 100, 1)
    metrics["num_liquidations"]  = num_liquidations
    metrics["total_funding_pct"] = round(total_funding * 100, 2)

    return nav_arr, metrics, position_log


# ──────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────

def _run_position_baseline(
    test_df:     pd.DataFrame,
    position:    float,
    leverage:    float = 2.0,
    initial_nav: float = 10_000.0,
    funding_base_rate: float = 0.0003,
) -> tuple:
    """Generic baseline: hold a fixed position."""
    close = test_df["CloseRaw"].values
    nav   = initial_nav
    navs  = [nav]
    positions = [position]

    for t in range(1, len(close)):
        log_ret = np.log(close[t] / (close[t - 1] + 1e-12))
        pnl = position * log_ret * leverage

        # Funding cost
        ma21 = close[max(0, t - 21):t].mean() if t >= 21 else close[:t].mean()
        funding_rate = funding_base_rate * np.sign(close[t] - ma21)
        funding_cost = abs(position) * funding_rate * leverage

        nav = nav * (1.0 + pnl) - nav * funding_cost
        nav = max(nav, 1.0)
        navs.append(nav)
        positions.append(position)

    nav_arr = np.array(navs, dtype=float)
    metrics = compute_all_metrics(nav_arr, positions)
    return nav_arr, metrics


def run_long_hold_baseline(test_df, leverage=2.0, initial_nav=10_000.0):
    """100% long with leverage. Bull market benchmark."""
    return _run_position_baseline(test_df, position=1.0, leverage=leverage, initial_nav=initial_nav)


def run_short_hold_baseline(test_df, leverage=2.0, initial_nav=10_000.0):
    """100% short with leverage. Bear market benchmark."""
    return _run_position_baseline(test_df, position=-1.0, leverage=leverage, initial_nav=initial_nav)


def run_momentum_baseline(
    test_df:     pd.DataFrame,
    leverage:    float = 2.0,
    initial_nav: float = 10_000.0,
    ma_period:   int   = 21,
    funding_base_rate: float = 0.0003,
) -> tuple:
    """
    Simple momentum: long when Close > MA21, short when Close < MA21.
    This is the hardest baseline to beat.
    """
    close = test_df["CloseRaw"].values
    nav   = initial_nav
    navs  = [nav]
    positions = [0.0]

    for t in range(1, len(close)):
        # Determine position based on momentum signal
        if t >= ma_period:
            ma = close[t - ma_period:t].mean()
            position = 1.0 if close[t - 1] > ma else -1.0
        else:
            position = 0.0

        log_ret = np.log(close[t] / (close[t - 1] + 1e-12))
        pnl = position * log_ret * leverage

        # Funding cost
        ma21 = close[max(0, t - 21):t].mean() if t >= 21 else close[:t].mean()
        funding_rate = funding_base_rate * np.sign(close[t] - ma21)
        funding_cost = abs(position) * funding_rate * leverage

        # Transaction cost when position flips
        if t > 0 and len(positions) > 0 and abs(position - positions[-1]) > 0.01:
            trade_cost = 0.0004 * abs(position - positions[-1]) * leverage
        else:
            trade_cost = 0.0

        nav = nav * (1.0 + pnl) - nav * (funding_cost + trade_cost)
        nav = max(nav, 1.0)
        navs.append(nav)
        positions.append(position)

    nav_arr = np.array(navs, dtype=float)
    metrics = compute_all_metrics(nav_arr, positions)
    return nav_arr, metrics


def run_random_baseline(
    test_df:     pd.DataFrame,
    leverage:    float = 2.0,
    initial_nav: float = 10_000.0,
    seed:        int   = 42,
    funding_base_rate: float = 0.0003,
) -> tuple:
    """Random position [-1, +1] each day. Lower bound benchmark."""
    rng   = np.random.default_rng(seed)
    close = test_df["CloseRaw"].values
    nav   = initial_nav
    navs  = [nav]
    positions = [0.0]
    prev_pos  = 0.0

    for t in range(1, len(close)):
        position = float(rng.uniform(-1, 1))

        log_ret = np.log(close[t] / (close[t - 1] + 1e-12))
        pnl = position * log_ret * leverage

        # Costs
        ma21 = close[max(0, t - 21):t].mean() if t >= 21 else close[:t].mean()
        funding_rate = funding_base_rate * np.sign(close[t] - ma21)
        funding_cost = abs(position) * funding_rate * leverage
        trade_cost   = 0.0004 * abs(position - prev_pos) * leverage

        nav = nav * (1.0 + pnl) - nav * (funding_cost + trade_cost)
        nav = max(nav, 1.0)
        navs.append(nav)
        positions.append(position)
        prev_pos = position

    nav_arr = np.array(navs, dtype=float)
    metrics = compute_all_metrics(nav_arr, positions)
    return nav_arr, metrics


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot_nav_comparison(
    nav_dict:  dict,
    title:     str,
    save_path: str,
    index:     pd.DatetimeIndex = None,
):
    """Plot multiple NAV curves on one figure."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#00BFFF", "#FF6B35", "#7FFF00", "#FFD700", "#DA70D6", "#FF69B4"]
    styles = ["-", "--", "-.", ":", "-", "--"]

    for i, (label, nav) in enumerate(nav_dict.items()):
        if index is not None:
            n = min(len(nav), len(index))
            x, nav_plot = index[:n], nav[:n]
        else:
            x = np.arange(len(nav))
            nav_plot = nav
        ax.plot(
            x, nav_plot, label=label,
            color=colors[i % len(colors)],
            linestyle=styles[i % len(styles)],
            linewidth=1.8,
        )

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Date" if index is not None else "Step")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {save_path}")


def plot_position_history(
    position_log: list,
    price_series: np.ndarray,
    title:        str,
    save_path:    str,
    index:        pd.DatetimeIndex = None,
):
    """
    Plot position history with price overlay.
    Long regions in green, short in red, flat in gray.
    """
    positions = np.array(position_log)
    n = min(len(positions), len(price_series))
    positions = positions[:n]
    prices    = price_series[:n]

    if index is not None:
        x = index[:n]
    else:
        x = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Position bars (background fill)
    for i in range(len(positions)):
        if positions[i] > 0.05:
            color = "#00FF0030"   # green (long)
        elif positions[i] < -0.05:
            color = "#FF000030"   # red (short)
        else:
            color = "#88888820"   # gray (flat)

        if i < len(x) - 1:
            ax1.axvspan(x[i], x[min(i + 1, len(x) - 1)], facecolor=color, edgecolor="none")

    # Position line
    ax1.plot(x, positions, color="#00BFFF", linewidth=1.2, label="Position")
    ax1.axhline(y=0, color="white", linewidth=0.5, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Position [-1, +1]", color="#00BFFF")
    ax1.set_ylim(-1.2, 1.2)
    ax1.tick_params(axis="y", labelcolor="#00BFFF")

    # Price on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, prices, color="#FFD700", linewidth=1.0, alpha=0.7, label="Price")
    ax2.set_ylabel("Price ($)", color="#FFD700")
    ax2.tick_params(axis="y", labelcolor="#FFD700")

    ax1.set_title(title, fontsize=13, pad=10, color="white")
    ax1.set_xlabel("Date" if index is not None else "Step", color="white")
    ax1.tick_params(axis="x", colors="white")

    ax1.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=9, facecolor="#1a1a2e", labelcolor="white")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {save_path}")


def generate_report_table(metrics_dict: dict) -> pd.DataFrame:
    """Build comparison table from metrics dict."""
    metric_labels = {
        "total_return":      "Total Return (%)",
        "annual_return":     "Ann. Return (%)",
        "sharpe":            "Sharpe Ratio",
        "sortino":           "Sortino Ratio",
        "max_drawdown":      "Max Drawdown (%)",
        "calmar":            "Calmar Ratio",
        "long_pct":          "Long (%)",
        "short_pct":         "Short (%)",
        "flat_pct":          "Flat (%)",
        "avg_position_size": "Avg Position",
        "trading_frequency": "Trading Freq. (%)",
        "num_liquidations":  "Liquidations",
        "total_funding_pct": "Funding Paid (%)",
        "num_trades":        "Num Trades",
        "win_rate":          "Win Rate (%)",
        "profit_factor":     "Profit Factor",
    }

    rows = []
    for key, label in metric_labels.items():
        row = {"Metric": label}
        for name, m in metrics_dict.items():
            row[name] = m.get(key, "-")
        rows.append(row)

    return pd.DataFrame(rows).set_index("Metric")


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[evaluate_futures] Self-test...")
    nav = np.array([10000, 10100, 9950, 10200, 10150, 10500, 10300, 10800], dtype=float)
    positions = [0.0, 0.5, 0.5, 0.8, 0.3, -0.2, -0.5, 0.7]
    m = compute_all_metrics(nav, positions)
    print(f"  Metrics: {m}")
    print("  OK")
