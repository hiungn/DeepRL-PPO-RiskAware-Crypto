"""
Train PPO agent for single-asset futures trading (long/short).

Differences from train_ppo_portfolio.py:
  - Single asset (BTC by default) instead of multi-asset portfolio
  - Uses SingleAssetFuturesEnv_optimize (tanh action space [-1, +1])
  - VecNormalize for reward normalization (SOTA)
  - Adjusted PPO hyperparameters for 1D action space
  - Models saved to models_optimize/

Usage:
  # Default: BTC, sortino_style, all windows, all seeds
  python src/agents/train_futures_optimize.py

  # Single run for testing
  python src/agents/train_futures_optimize.py --reward sortino_style --window 1 --seed 42 --steps 50000

  # With LSTM
  python src/agents/train_futures_optimize.py --lstm

  # Custom leverage
  python src/agents/train_futures_optimize.py --leverage 3
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement,
)

from src.utils.data_utils_optimize import (
    load_single_asset, get_split, WALK_FORWARD_SPLITS, DEFAULT_TICKER,
)
from src.envs.SingleAssetFuturesEnv_optimize import SingleAssetFuturesEnv


# ──────────────────────────────────────────────────────────────
# PPO Hyperparameters (optimized for 1D continuous action)
# ──────────────────────────────────────────────────────────────

PPO_KWARGS = dict(
    learning_rate = 3e-4,       # higher than original 1e-5 (1D action is simpler)
    n_steps       = 2048,       # rollout buffer size
    batch_size    = 256,        # mini-batch size
    gamma         = 0.99,       # discount factor (long-horizon)
    gae_lambda    = 0.95,       # GAE lambda (slightly higher than original 0.92)
    ent_coef      = 0.01,       # entropy bonus (lower than original 0.03; 1D needs less exploration)
    vf_coef       = 0.5,        # value function loss weight
    clip_range    = 0.2,        # PPO clipping
    max_grad_norm = 0.5,        # gradient clipping for stability
    device        = "auto",     # use GPU if available
    verbose       = 1,
)

LSTM_POLICY_KWARGS = dict(
    n_lstm_layers      = 1,
    lstm_hidden_size   = 128,   # smaller than original 256 (single asset, simpler)
    shared_lstm        = False,
    enable_critic_lstm = True,
)

TOTAL_STEPS     = 500_000
EVAL_FREQ       = 50_000
CKPT_FREQ       = 100_000
LEVERAGE        = 2.0
LAMBDA_RISK     = 0.3
LAMBDA_DRAWDOWN = 2.0
LAMBDA_CVAR     = 0.5

REWARD_TYPES = ["sortino_style", "sharpe_style", "raw"]
SEEDS        = [42, 123, 777]
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "../../models_optimize")


# ──────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────

def make_env(
    price_df,
    reward_type:    str   = "sortino_style",
    episode_len:    int   = 365,
    leverage:       float = LEVERAGE,
    lambda_risk:    float = LAMBDA_RISK,
    lambda_drawdown: float = LAMBDA_DRAWDOWN,
    lambda_cvar:    float = LAMBDA_CVAR,
):
    """Create a vectorized environment with VecNormalize for reward normalization."""
    def _inner():
        return SingleAssetFuturesEnv(
            price_df            = price_df,
            reward_type         = reward_type,
            episode_len         = episode_len,
            leverage            = leverage,
            lambda_risk         = lambda_risk,
            lambda_drawdown     = lambda_drawdown,
            lambda_cvar         = lambda_cvar,
        )

    vec_env = DummyVecEnv([_inner])

    # SOTA: Reward normalization via running z-score
    # norm_obs=False because env already applies rolling z-score to features
    # norm_reward=True stabilizes PPO training on non-stationary crypto rewards
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = False,
        norm_reward = True,
        clip_reward = 10.0,
        gamma       = PPO_KWARGS["gamma"],
    )

    return vec_env


def make_eval_env(
    price_df,
    reward_type:    str   = "sortino_style",
    leverage:       float = LEVERAGE,
    lambda_risk:    float = LAMBDA_RISK,
    lambda_drawdown: float = LAMBDA_DRAWDOWN,
    lambda_cvar:    float = LAMBDA_CVAR,
):
    """Create eval env (full test period, no reward normalization)."""
    n_days = len(price_df)

    def _inner():
        return SingleAssetFuturesEnv(
            price_df            = price_df,
            reward_type         = reward_type,
            episode_len         = n_days,
            leverage            = leverage,
            lambda_risk         = lambda_risk,
            lambda_drawdown     = lambda_drawdown,
            lambda_cvar         = lambda_cvar,
        )

    vec_env = DummyVecEnv([_inner])
    # Eval env uses VecNormalize too but with training=False
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = False,
        norm_reward = True,
        clip_reward = 10.0,
        gamma       = PPO_KWARGS["gamma"],
        training    = False,
    )
    return vec_env


# ──────────────────────────────────────────────────────────────
# Model naming
# ──────────────────────────────────────────────────────────────

def model_name_str(
    reward_type: str,
    win_id:      int,
    seed:        int,
    use_lstm:    bool,
    leverage:    float,
) -> str:
    policy_tag = "lstm" if use_lstm else "mlp"
    lev_tag    = f"_lev{leverage:.0f}" if leverage != 2.0 else ""
    return f"futures_{policy_tag}_{reward_type}_w{win_id}_s{seed}{lev_tag}"


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_one(
    reward_type: str,
    split_cfg:   dict,
    train_df,
    eval_df,
    seed:        int   = 42,
    use_lstm:    bool  = False,
    total_steps: int   = TOTAL_STEPS,
    leverage:    float = LEVERAGE,
    lambda_risk: float = LAMBDA_RISK,
    lambda_cvar: float = LAMBDA_CVAR,
):
    os.makedirs(MODELS_DIR, exist_ok=True)

    win_id   = split_cfg["id"]
    name     = model_name_str(reward_type, win_id, seed, use_lstm, leverage)
    save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    policy_str = "LSTM" if use_lstm else "MLP"
    print(f"\n{'='*65}")
    print(f"Training: {reward_type} [{policy_str}]  W{win_id}  seed={seed}  leverage={leverage}x")
    print(f"  Train : {split_cfg['train_start']} to {split_cfg['train_end']}")
    print(f"  Eval  : {split_cfg['test_start']}  to {split_cfg['test_end']}")
    print(f"{'='*65}")

    train_env = make_env(
        train_df, reward_type, episode_len=365,
        leverage=leverage, lambda_risk=lambda_risk, lambda_cvar=lambda_cvar,
    )
    eval_env = make_eval_env(
        eval_df, reward_type,
        leverage=leverage, lambda_risk=lambda_risk, lambda_cvar=lambda_cvar,
    )

    if use_lstm:
        from sb3_contrib import RecurrentPPO
        lstm_kwargs = dict(**PPO_KWARGS)
        lstm_kwargs["batch_size"] = 128
        lstm_kwargs.pop("verbose")
        model = RecurrentPPO(
            "MlpLstmPolicy", train_env,
            policy_kwargs = LSTM_POLICY_KWARGS,
            verbose       = 1,
            seed          = seed,
            **lstm_kwargs,
        )
    else:
        model = PPO("MlpPolicy", train_env, seed=seed, **PPO_KWARGS)

    # Callbacks
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals = 8,
        min_evals                = 6,
        verbose                  = 1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = save_dir,
        log_path             = save_dir,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = 3,
        deterministic        = True,
        render               = False,
        callback_after_eval  = stop_cb,
    )
    ckpt_cb = CheckpointCallback(
        save_freq   = CKPT_FREQ,
        save_path   = save_dir,
        name_prefix = name,
    )

    model.learn(
        total_timesteps     = total_steps,
        callback            = [eval_cb, ckpt_cb],
        reset_num_timesteps = True,
    )

    # Save model + VecNormalize stats
    final_path = os.path.join(MODELS_DIR, f"{name}_final.zip")
    model.save(final_path)

    vec_norm_path = os.path.join(save_dir, "vec_normalize.pkl")
    train_env.save(vec_norm_path)

    print(f"\nSaved final    : {final_path}")
    print(f"Saved best     : {os.path.join(save_dir, 'best_model.zip')}")
    print(f"Saved VecNorm  : {vec_norm_path}")

    train_env.close()
    eval_env.close()

    return final_path, os.path.join(save_dir, "best_model.zip")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PPO futures agent (single asset)")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER)
    parser.add_argument("--reward", type=str, default="all",
                        choices=REWARD_TYPES + ["all"])
    parser.add_argument("--window", type=int, default=0, choices=[0, 1, 2],
                        help="0 = all windows")
    parser.add_argument("--seed", type=int, default=0,
                        help="0 = all seeds (42, 123, 777); else specific seed")
    parser.add_argument("--lstm", action="store_true",
                        help="Use LSTM (RecurrentPPO) instead of MLP")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--leverage", type=float, default=LEVERAGE)
    parser.add_argument("--lambda_risk", type=float, default=LAMBDA_RISK)
    parser.add_argument("--lambda_cvar", type=float, default=LAMBDA_CVAR)
    parser.add_argument("--models-dir", type=str, default=None)
    args = parser.parse_args()

    global MODELS_DIR
    if args.models_dir:
        MODELS_DIR = args.models_dir

    rewards = REWARD_TYPES if args.reward == "all" else [args.reward]
    windows = (WALK_FORWARD_SPLITS if args.window == 0
               else [s for s in WALK_FORWARD_SPLITS if s["id"] == args.window])
    seeds   = SEEDS if args.seed == 0 else [args.seed]

    total_runs = len(rewards) * len(windows) * len(seeds)
    policy_str = "LSTM" if args.lstm else "MLP"

    print(f"[train_futures] Ticker={args.ticker} | Policy={policy_str} | Leverage={args.leverage}x")
    print(f"[train_futures] Rewards={rewards} | Windows={[w['id'] for w in windows]} | Seeds={seeds}")
    print(f"[train_futures] Total runs: {total_runs} x {args.steps} steps")

    print("\n[train_futures] Loading data...")
    all_data = load_single_asset(ticker=args.ticker, normalize=True)

    run = 0
    for split_cfg in windows:
        train_df, eval_df = get_split(all_data, split_cfg)
        for reward_type in rewards:
            for seed in seeds:
                run += 1
                print(f"\n[Run {run}/{total_runs}]")
                train_one(
                    reward_type = reward_type,
                    split_cfg   = split_cfg,
                    train_df    = train_df,
                    eval_df     = eval_df,
                    seed        = seed,
                    use_lstm    = args.lstm,
                    total_steps = args.steps,
                    leverage    = args.leverage,
                    lambda_risk = args.lambda_risk,
                    lambda_cvar = args.lambda_cvar,
                )

    print(f"\n[train_futures] All {total_runs} runs complete.")


if __name__ == "__main__":
    main()
