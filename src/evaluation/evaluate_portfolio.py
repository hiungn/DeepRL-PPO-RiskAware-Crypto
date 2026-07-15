"""
Evaluation metrics and backtest for portfolio RL agent.

All metrics computed on full test period (not per-step reward).
Main functions: run_backtest(), run_*_baseline()
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

from src.envs.MultiAssetPortfolioEnv import MultiAssetPortfolioEnv


# Metric functions

def compute_sharpe(returns: np.ndarray, rf: float = 0.0, freq: int = 365) -> float:
    """Sharpe = (E[r] - rf) / std(r) * sqrt(freq)"""
    excess = returns - rf / freq
    if np.std(excess) < 1e-10:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(freq))


def compute_sortino(returns: np.ndarray, rf: float = 0.0, freq: int = 365) -> float:
    """Sortino = (E[r] - rf) / downside_std * sqrt(freq)"""
    mar     = rf / freq
    down    = returns[returns < mar]
    if len(down) < 2:
        return 0.0
    sigma_d = float(np.std(down))
    if sigma_d < 1e-10:
        return 0.0
    return float((np.mean(returns) - mar) / sigma_d * np.sqrt(freq))


def compute_max_drawdown(nav_series: np.ndarray) -> float:
    """MDD = max peak-to-trough drawdown fraction."""
    running_max = np.maximum.accumulate(nav_series)
    drawdowns   = 1.0 - nav_series / (running_max + 1e-12)
    return float(np.max(drawdowns))


def compute_calmar(ann_return: float, mdd: float) -> float:
    """Calmar = Annualized Return / MDD"""
    return 0.0 if mdd < 1e-10 else float(ann_return / mdd)


def compute_annual_return(nav_series: np.ndarray, freq: int = 365) -> float:
    """Annualized return from NAV series."""
    total = nav_series[-1] / nav_series[0] - 1
    n_yrs = len(nav_series) / freq
    return 0.0 if n_yrs < 1e-3 else float((1 + total) ** (1 / n_yrs) - 1)


def compute_allocation_entropy(weights_log: list) -> float:
    """
    Mean Shannon entropy of daily portfolio weights.
    High entropy = diversified; low entropy = concentrated.
    H = -sum(w * log(w)),  normalized by log(n) to [0, 1].
    Used in robustness experiment to compare allocation diversity.
    """
    entropies = []
    for w in weights_log:
        w = np.array(w)
        w = w[w > 1e-8]   # avoid log(0)
        if len(w) == 0:
            continue
        h = -np.sum(w * np.log(w))
        h_max = np.log(len(w)) if len(w) > 1 else 1.0
        entropies.append(h / h_max if h_max > 0 else 0.0)
    return float(np.mean(entropies)) if entropies else 0.0


def compute_all_metrics(nav_series: np.ndarray,
                        weights_log: list = None,
                        freq: int = 365) -> dict:
    """Compute full metric suite from NAV series."""
    returns   = np.diff(np.log(nav_series + 1e-12))
    total_ret = nav_series[-1] / nav_series[0] - 1
    ann_ret   = compute_annual_return(nav_series, freq)
    sharpe    = compute_sharpe(returns, freq=freq)
    sortino   = compute_sortino(returns, freq=freq)
    mdd       = compute_max_drawdown(nav_series)
    calmar    = compute_calmar(ann_ret, mdd)
    entropy   = compute_allocation_entropy(weights_log) if weights_log else None

    m = {
        "total_return":  round(total_ret * 100, 2),   # %
        "annual_return": round(ann_ret   * 100, 2),   # %
        "sharpe":        round(sharpe,  3),
        "sortino":       round(sortino, 3),
        "max_drawdown":  round(mdd      * 100, 2),    # %
        "calmar":        round(calmar,  3),
    }
    if entropy is not None:
        m["alloc_entropy"] = round(entropy, 3)
    return m


# Backtest

def run_backtest(
    model_path:          str,
    test_data:           dict,
    reward_type:         str   = "sortino_style",
    initial_nav:         float = 10_000.0,
    use_cash:            bool  = True,
    use_lstm:            bool  = False,
    lambda_risk:         float = 0.5,
    lambda_turnover:     float = 0.0015,
    lambda_drawdown:     float = 2.0,
    rebalance_threshold: float = 0.05,
    softmax_temperature: float = 0.5,
) -> tuple:
    """
    Backtest a trained PPO (MLP or LSTM) model on test_data.

    Parameters:
    use_lstm            : bool  - True = load with RecurrentPPO, use lstm_states
    lambda_risk         : float - risk penalty coefficient passed to env
    lambda_turnover     : float - turnover penalty coefficient
    lambda_drawdown     : float - quadratic drawdown penalty coefficient
    rebalance_threshold : float - minimum turnover to trigger trade

    Returns:
    nav_arr     : np.ndarray  - NAV series over test period
    metrics     : dict        - Sharpe, Sortino, MDD, Calmar, Return, entropy,
                                trading stats (turnover, frequency, fees)
    weights_log : list        - portfolio weights per day
    """
    if use_lstm:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path)
    else:
        model = PPO.load(model_path)

    n_days = len(next(iter(test_data.values())))
    env = MultiAssetPortfolioEnv(
        data_dict           = test_data,
        reward_type         = reward_type,
        window_size         = 30,
        episode_len         = n_days,
        fee                 = 0.001,
        lambda_risk         = lambda_risk,
        lambda_turnover     = lambda_turnover,
        lambda_drawdown     = lambda_drawdown,
        rebalance_threshold = rebalance_threshold,
        risk_window         = 14,
        reward_clip         = 5.0,
        initial_nav         = initial_nav,
        use_cash            = use_cash,
        softmax_temperature = softmax_temperature,
    )

    # Fixed start (no random offset for eval - reproducible)
    obs, _ = env.reset()
    env.episode_start       = env.window_size
    env.current_step        = env.window_size
    env.weights             = env._initial_weights()
    env.nav                 = float(initial_nav)
    env._return_history     = []
    env._consecutive_losses = 0
    env._peak_nav           = float(initial_nav)
    obs = env._get_obs()

    nav_series       = [initial_nav]
    weights_log      = [env.weights.copy()]
    cash_weights_log = []
    turnover_log     = []
    trade_count      = 0

    # LSTM state
    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)

    terminated = truncated = False
    while not (terminated or truncated):
        if use_lstm:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            episode_start = np.array([terminated or truncated])
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        nav_series.append(info["nav"])
        weights_log.append(info["weights"].copy())
        turnover_log.append(info["turnover"])
        if info.get("rebalanced", False):
            trade_count += 1
        if use_cash:
            cash_weights_log.append(info["cash_weight"])

    nav_arr = np.array(nav_series, dtype=float)
    metrics = compute_all_metrics(nav_arr, weights_log)

    # Trading activity metrics
    n_steps = len(turnover_log)
    total_turnover = float(np.sum(turnover_log))
    metrics["total_turnover"]    = round(total_turnover, 3)
    metrics["avg_daily_turnover"] = round(total_turnover / max(n_steps, 1), 4)
    metrics["trading_frequency"] = round(trade_count / max(n_steps, 1) * 100, 1)  # %
    metrics["total_fees_pct"]    = round(total_turnover * 0.001 * 100, 2)  # % of initial NAV

    if use_cash and cash_weights_log:
        metrics["avg_cash_weight"] = round(float(np.mean(cash_weights_log)), 3)
        metrics["max_cash_weight"] = round(float(np.max(cash_weights_log)),  3)

    return nav_arr, metrics, weights_log


# Baselines

def run_equal_weight_baseline(
    test_data:   dict,
    initial_nav: float = 10_000.0,
) -> tuple:
    """Equal-weight across all crypto assets (no cash, buy-and-hold weights)."""
    tickers = sorted(test_data.keys())
    n       = len(tickers)
    weights = np.ones(n) / n

    close_mat = np.stack([test_data[t]["CloseRaw"].values for t in tickers], axis=1)

    nav  = initial_nav
    navs = [nav]
    wlog = [weights.copy()]
    for t in range(1, len(close_mat)):
        log_ret  = np.log((close_mat[t] + 1e-12) / (close_mat[t-1] + 1e-12))
        port_ret = float(np.dot(weights, log_ret))
        nav      = nav * np.exp(port_ret)
        navs.append(nav)
        wlog.append(weights.copy())

    nav_arr = np.array(navs, dtype=float)
    metrics = compute_all_metrics(nav_arr, wlog)
    return nav_arr, metrics


def run_bnh_baseline(
    test_data:   dict,
    ticker:      str   = "BTC",
    initial_nav: float = 10_000.0,
) -> tuple:
    """100% Buy & Hold on a single asset."""
    close = test_data[ticker]["CloseRaw"].values
    nav   = initial_nav
    navs  = [nav]
    n_assets = len(test_data)
    w = np.zeros(n_assets)
    w[0] = 1.0   # all in ticker (sorted order)
    wlog = [w.copy()]
    for t in range(1, len(close)):
        log_ret = float(np.log((close[t] + 1e-12) / (close[t-1] + 1e-12)))
        nav     = nav * np.exp(log_ret)
        navs.append(nav)
        wlog.append(w.copy())

    nav_arr = np.array(navs, dtype=float)
    metrics = compute_all_metrics(nav_arr, wlog)
    return nav_arr, metrics


def run_mcap_baseline(
    test_data:   dict,
    weights:     dict  = None,
    initial_nav: float = 10_000.0,
) -> tuple:
    """
    Market-cap weighted baseline (fixed weights).
    Default: BTC=60%, ETH=30%, BNB=10% (3-asset setup).
    For 5-asset: pass custom weights dict.
    """
    tickers = sorted(test_data.keys())
    if weights is None:
        # Default 3-asset market-cap approximation
        default = {"BTC": 0.60, "ETH": 0.30, "BNB": 0.10}
        weights = default

    w = np.array([weights.get(t, 1.0 / len(tickers)) for t in tickers])
    w = w / w.sum()   # normalize to 1

    close_mat = np.stack([test_data[t]["CloseRaw"].values for t in tickers], axis=1)

    nav  = initial_nav
    navs = [nav]
    wlog = [w.copy()]
    for t in range(1, len(close_mat)):
        log_ret  = np.log((close_mat[t] + 1e-12) / (close_mat[t-1] + 1e-12))
        port_ret = float(np.dot(w, log_ret))
        nav      = nav * np.exp(port_ret)
        navs.append(nav)
        wlog.append(w.copy())

    nav_arr = np.array(navs, dtype=float)
    metrics = compute_all_metrics(nav_arr, wlog)
    return nav_arr, metrics


# Plotting & Reporting

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
            x, nav = index[:n], nav[:n]
        else:
            x = np.arange(len(nav))
        ax.plot(x, nav, label=label,
                color=colors[i % len(colors)],
                linestyle=styles[i % len(styles)],
                linewidth=1.8)

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


def generate_report_table(metrics_dict: dict) -> pd.DataFrame:
    """
    Build comparison table from metrics dict.
    metrics_dict: {"Model Name": metrics_from_compute_all_metrics, ...}
    Returns: pd.DataFrame, index=metric names, columns=model names
    """
    metric_labels = {
        "total_return":      "Total Return (%)",
        "annual_return":     "Ann. Return (%)",
        "sharpe":            "Sharpe Ratio",
        "sortino":           "Sortino Ratio",
        "max_drawdown":      "Max Drawdown (%)",
        "calmar":            "Calmar Ratio",
        "alloc_entropy":     "Alloc. Entropy",
        "trading_frequency": "Trading Freq. (%)",
        "total_fees_pct":    "Total Fees (%)",
        "total_turnover":    "Total Turnover",
        "avg_cash_weight":   "Avg Cash (%)",
        "max_cash_weight":   "Max Cash (%)",
    }
    rows = []
    for key, label in metric_labels.items():
        row = {"Metric": label}
        for name, m in metrics_dict.items():
            row[name] = m.get(key, "-")
        rows.append(row)

    return pd.DataFrame(rows).set_index("Metric")


if __name__ == "__main__":
    print("[evaluate_portfolio] Self-test...")
    nav = np.array([10000, 10100, 9950, 10200, 10150, 10500, 10300, 10800], dtype=float)
    m   = compute_all_metrics(nav)
    print(f"  Metrics: {m}")
    print("  OK")
