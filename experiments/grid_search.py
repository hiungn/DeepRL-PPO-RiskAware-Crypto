"""
Hyperparameter grid search for PPO portfolio agent (env with cash,
price-drift no-trade zone, turnover penalty).

Search space:
  learning_rate : [1e-5, 3e-5]
  gamma         : [0.97, 0.985, 0.99]
  ent_coef      : [0.01, 0.02, 0.03]
  lambda_risk   : [0.3, 0.5, 1.0]
  Total         : 54 combinations

Env config (fixed):
  use_cash            = True
  fee                 = 0.001
  lambda_turnover     = 0.0015
  rebalance_threshold = 0.05

Fixed PPO params:
  n_steps    = 2048
  batch_size = 256
  gae_lambda = 0.92
  vf_coef    = 0.7
  clip_range = 0.2
  device     = cpu

Metric: Sortino Ratio on val set

Output:
  experiments/results/exp0_hparam_results.csv
  experiments/results/exp0_hparam_best.txt

Usage:
  python experiments/grid_search.py
  python experiments/grid_search.py --steps 150000
"""

import os
import sys
import argparse
import itertools
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.utils.data_utils import load_all_data, get_split, WALK_FORWARD_SPLITS
from src.envs.MultiAssetPortfolioEnv import MultiAssetPortfolioEnv
from src.evaluation.evaluate_portfolio import run_backtest, compute_all_metrics

# Search space
SEARCH_SPACE = {
    "learning_rate": [1e-5, 3e-5],
    "gamma":         [0.97, 0.985, 0.99],
    "ent_coef":      [0.01, 0.02, 0.03],
    "lambda_risk":   [0.3, 0.5, 1.0],
}

# Fixed PPO params
FIXED_PARAMS = dict(
    n_steps    = 2048,
    batch_size = 256,
    gae_lambda = 0.92,
    vf_coef    = 0.7,
    clip_range = 0.2,
    device     = "cpu",
    verbose    = 0,
    seed       = 42,
)

# Env config
USE_CASH            = True
FEE                 = 0.001
LAMBDA_TURNOVER     = 0.0015
LAMBDA_DRAWDOWN     = 2.0
REBALANCE_THRESHOLD = 0.05
SOFTMAX_TEMPERATURE = 0.5

EVAL_REWARD   = "sortino_style"
EPISODE_LEN   = 252
EVAL_FREQ     = 50_000
N_EVAL_EPS    = 3

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "../models/hparam_search")

# Window 1 for tuning (Window 2 is hold-out)
TUNE_WINDOW   = WALK_FORWARD_SPLITS[0]   # id=1


# Helpers

def make_env(data_dict: dict, episode_len: int = EPISODE_LEN, lambda_risk: float = 0.5):
    def _inner():
        return MultiAssetPortfolioEnv(
            data_dict           = data_dict,
            reward_type         = EVAL_REWARD,
            window_size         = 30,
            episode_len         = episode_len,
            fee                 = FEE,
            lambda_risk         = lambda_risk,
            lambda_turnover     = LAMBDA_TURNOVER,
            lambda_drawdown     = LAMBDA_DRAWDOWN,
            rebalance_threshold = REBALANCE_THRESHOLD,
            risk_window         = 14,
            reward_clip         = 5.0,
            initial_nav         = 10_000.0,
            use_cash            = USE_CASH,
            softmax_temperature = SOFTMAX_TEMPERATURE,
        )
    return DummyVecEnv([_inner])


def run_one(
    lr: float,
    gamma: float,
    ent_coef: float,
    lambda_risk: float,
    train_data: dict,
    val_data:   dict,
    total_steps: int,
    run_id: int,
) -> dict:
    """Train 1 PPO model, return metrics on val set."""
    run_name = "run{:03d}_lr{:.0e}_g{}_e{}_lr{}".format(run_id, lr, gamma, ent_coef, lambda_risk)
    save_dir = os.path.join(MODELS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)

    ppo_kwargs = dict(
        learning_rate = lr,
        gamma         = gamma,
        ent_coef      = ent_coef,
        **FIXED_PARAMS,
    )

    train_env = make_env(train_data, lambda_risk=lambda_risk)
    val_env   = make_env(val_data, episode_len=len(next(iter(val_data.values()))), lambda_risk=lambda_risk)

    eval_cb = EvalCallback(
        val_env,
        best_model_save_path = save_dir,
        log_path             = save_dir,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPS,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )

    model = PPO("MlpPolicy", train_env, **ppo_kwargs)
    t0 = time.time()
    model.learn(
        total_timesteps     = total_steps,
        callback            = eval_cb,
        reset_num_timesteps = True,
    )
    elapsed = time.time() - t0

    best_path = os.path.join(save_dir, "best_model.zip")
    if not os.path.exists(best_path):
        final_path = os.path.join(save_dir, "final.zip")
        model.save(final_path)
        best_path = final_path

    try:
        nav_arr, metrics, _ = run_backtest(
            model_path          = best_path,
            test_data           = val_data,
            reward_type         = EVAL_REWARD,
            use_cash            = USE_CASH,
            lambda_risk         = lambda_risk,
            lambda_turnover     = LAMBDA_TURNOVER,
            lambda_drawdown     = LAMBDA_DRAWDOWN,
            rebalance_threshold = REBALANCE_THRESHOLD,
            softmax_temperature = SOFTMAX_TEMPERATURE,
        )
    except Exception as ex:
        print("    [WARN] Backtest failed for {}: {}".format(run_name, ex))
        metrics = {
            "total_return": 0.0, "annual_return": 0.0,
            "sharpe": 0.0, "sortino": 0.0,
            "max_drawdown": 100.0, "calmar": 0.0,
        }

    result = {
        "run_id":        run_id,
        "learning_rate": lr,
        "gamma":         gamma,
        "ent_coef":      ent_coef,
        "lambda_risk":   lambda_risk,
        "total_return":  metrics["total_return"],
        "annual_return": metrics["annual_return"],
        "sharpe":        metrics["sharpe"],
        "sortino":       metrics["sortino"],
        "max_drawdown":  metrics["max_drawdown"],
        "calmar":        metrics["calmar"],
        "trade_freq":    metrics.get("trading_frequency", 0.0),
        "total_fees":    metrics.get("total_fees_pct", 0.0),
        "train_sec":     round(elapsed, 1),
    }

    print(
        "  Run {:3d} | lr={:.0e} gamma={} ent={} lrisk={} | "
        "Sortino={:+.3f}  Sharpe={:+.3f}  "
        "Return={:+.1f}%  MDD={:.1f}%  TradeFrq={:.0f}%  "
        "[{:.0f}s]".format(
            run_id, lr, gamma, ent_coef, lambda_risk,
            metrics['sortino'], metrics['sharpe'],
            metrics['total_return'], metrics['max_drawdown'],
            metrics.get('trading_frequency', 0.0),
            elapsed,
        )
    )

    return result



# Main

def main():
    parser = argparse.ArgumentParser(description="PPO Hyperparameter Grid Search")
    parser.add_argument(
        "--steps", type=int, default=150_000,
        help="Training steps per run (default 150000)"
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    print("[exp0] Loading data...")
    all_data = load_all_data(normalize=True)
    train_data, val_data = get_split(all_data, TUNE_WINDOW)

    # Build grid
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    grid   = list(itertools.product(*values))
    total  = len(grid)

    print("\n[exp0] Grid search: {} combinations x {} steps each".format(total, args.steps))
    print("       Tuning window: {} to {} (train) | {} to {} (val)".format(
        TUNE_WINDOW['train_start'], TUNE_WINDOW['train_end'],
        TUNE_WINDOW['test_start'], TUNE_WINDOW['test_end']))
    print("       Eval metric: Sortino Ratio on val set\n")

    all_results = []
    for i, combo in enumerate(grid, start=1):
        params = dict(zip(keys, combo))
        result = run_one(
            lr          = params["learning_rate"],
            gamma       = params["gamma"],
            ent_coef    = params["ent_coef"],
            lambda_risk = params["lambda_risk"],
            train_data  = train_data,
            val_data    = val_data,
            total_steps = args.steps,
            run_id      = i,
        )
        all_results.append(result)

    # Results
    df = pd.DataFrame(all_results).sort_values("sortino", ascending=False)

    print("\n" + "=" * 75)
    print("GRID SEARCH RESULTS (sorted by Sortino Ratio, val set)")
    print("=" * 75)
    display_cols = ["run_id", "learning_rate", "gamma", "ent_coef", "lambda_risk",
                    "sortino", "sharpe", "total_return", "max_drawdown",
                    "trade_freq", "total_fees", "train_sec"]
    print(df[display_cols].to_string(index=False))

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp0_hparam_results.csv")
    df.to_csv(csv_path, index=False)
    print("\nSaved: {}".format(csv_path))

    # Best params
    best = df.iloc[0]
    best_lr      = best["learning_rate"]
    best_gamma   = best["gamma"]
    best_ent     = best["ent_coef"]
    best_lrisk   = best["lambda_risk"]

    best_txt = os.path.join(RESULTS_DIR, "exp0_hparam_best.txt")
    with open(best_txt, "w") as f:
        f.write("Best Hyperparameters (grid search - env with cash)\n")
        f.write("=" * 55 + "\n")
        f.write("Tuning window  : W{} (train {} to {})\n".format(
            TUNE_WINDOW['id'], TUNE_WINDOW['train_start'], TUNE_WINDOW['train_end']))
        f.write("Val period     : {} to {}\n".format(
            TUNE_WINDOW['test_start'], TUNE_WINDOW['test_end']))
        f.write("Steps per run  : {}\n".format(args.steps))
        f.write("Total runs     : {}\n".format(total))
        f.write("Eval metric    : Sortino Ratio\n\n")
        f.write("Env config:\n")
        f.write("  use_cash            = {}\n".format(USE_CASH))
        f.write("  fee                 = {}\n".format(FEE))
        f.write("  lambda_turnover     = {}\n".format(LAMBDA_TURNOVER))
        f.write("  rebalance_threshold = {}\n\n".format(REBALANCE_THRESHOLD))
        f.write("Best tuned params:\n")
        f.write("  learning_rate  = {}\n".format(best_lr))
        f.write("  gamma          = {}\n".format(best_gamma))
        f.write("  ent_coef       = {}\n".format(best_ent))
        f.write("  lambda_risk    = {}\n\n".format(best_lrisk))
        f.write("Val Sortino    = {:.3f}\n".format(best['sortino']))
        f.write("Val Sharpe     = {:.3f}\n".format(best['sharpe']))
        f.write("Val Return     = {:.1f}%\n".format(best['total_return']))
        f.write("Val MDD        = {:.1f}%\n".format(best['max_drawdown']))
        f.write("Val Trade Freq = {:.1f}%\n".format(best.get('trade_freq', 0.0)))
        f.write("Val Total Fees = {:.2f}%\n\n".format(best.get('total_fees', 0.0)))
        f.write("Fixed PPO params (not tuned):\n")
        for k, v in FIXED_PARAMS.items():
            if k != "verbose":
                f.write("  {} = {}\n".format(k, v))
        f.write("\n")
        f.write("Top 5 runs:\n")
        f.write(df[display_cols].head(5).to_string(index=False))

    print("Saved: {}".format(best_txt))

    print("\n" + "=" * 75)
    print("BEST HYPERPARAMETERS:")
    print("  learning_rate = {}".format(best_lr))
    print("  gamma         = {}".format(best_gamma))
    print("  ent_coef      = {}".format(best_ent))
    print("  lambda_risk   = {}".format(best_lrisk))
    print("  -> Val Sortino   = {:.3f}".format(best['sortino']))
    print("  -> Val Trade Frq = {:.1f}%".format(best.get('trade_freq', 0.0)))
    print("  -> Val Fees      = {:.2f}%".format(best.get('total_fees', 0.0)))
    print("\nNext step: update PPO_KWARGS in train_ppo_portfolio.py with these values,")
    print("then retrain: python src/agents/train_ppo_portfolio.py --reward all --window 0")


if __name__ == "__main__":
    main()
