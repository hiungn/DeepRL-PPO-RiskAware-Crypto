"""
Train PPO agent (MLP or LSTM policy) on MultiAssetPortfolioEnv.

Assets : BTC, ETH, BNB + optional CASH
Rewards: sortino_style, sharpe_style, raw
Policy : MLP (default) or LSTM (--lstm)

Usage:
  # All rewards, MLP, cash enabled by default
  python src/agents/train_ppo_portfolio.py

  # Sortino only, MLP
  python src/agents/train_ppo_portfolio.py --reward sortino_style

  # Sortino, LSTM
  python src/agents/train_ppo_portfolio.py --reward sortino_style --lstm

  # Single run for testing
  python src/agents/train_ppo_portfolio.py --reward sortino_style --window 1 --seed 42 --steps 50000
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
)

from src.utils.data_utils import load_all_data, get_split, WALK_FORWARD_SPLITS, DEFAULT_TICKERS
from src.envs.MultiAssetPortfolioEnv import MultiAssetPortfolioEnv

# PPO Hyperparameters
PPO_KWARGS = dict(
    learning_rate = 1e-5,
    n_steps       = 2048,
    batch_size    = 256,
    gamma         = 0.99,
    gae_lambda    = 0.92,
    ent_coef      = 0.03,       
    vf_coef       = 0.7,
    clip_range    = 0.2,
    device        = "cpu",      
    verbose       = 1,
)

LSTM_POLICY_KWARGS = dict(
    n_lstm_layers  = 1,
    lstm_hidden_size = 256,
    shared_lstm    = False,
    enable_critic_lstm = True,
)

TOTAL_STEPS     = 500_000
EVAL_FREQ       = 50_000
CKPT_FREQ       = 100_000
LAMBDA_RISK     = 0.3
LAMBDA_DRAWDOWN = 2.0
SOFTMAX_TEMP    = 0.5
EPISODE_LEN     = 365

REWARD_TYPES = ["sortino_style", "sharpe_style", "raw"]
SEEDS        = [42, 123, 777]
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "../../models")


def make_env(data_dict: dict, reward_type: str, episode_len: int,
             use_cash: bool, lambda_risk: float = LAMBDA_RISK,
             lambda_turnover: float = 0.0015,
             lambda_drawdown: float = LAMBDA_DRAWDOWN,
             rebalance_threshold: float = 0.05,
             softmax_temperature: float = SOFTMAX_TEMP):
    def _inner():
        return MultiAssetPortfolioEnv(
            data_dict           = data_dict,
            reward_type         = reward_type,
            window_size         = 30,
            episode_len         = episode_len,
            fee                 = 0.001,
            lambda_risk         = lambda_risk,
            lambda_turnover     = lambda_turnover,
            lambda_drawdown     = lambda_drawdown,
            rebalance_threshold = rebalance_threshold,
            risk_window         = 14,
            reward_clip         = 5.0,
            initial_nav         = 10_000.0,
            use_cash            = use_cash,
            softmax_temperature = softmax_temperature,
        )
    return DummyVecEnv([_inner])


def model_name_str(reward_type: str, win_id: int, seed: int,
                   use_cash: bool, use_lstm: bool) -> str:
    policy_tag = "lstm" if use_lstm else "mlp"
    cash_tag   = "_cash" if use_cash else ""
    return f"ppo_{policy_tag}_{reward_type}_w{win_id}_s{seed}{cash_tag}"


def train_one(
    reward_type: str,
    split_cfg:   dict,
    train_data:  dict,
    eval_data:   dict,
    seed:        int   = 42,
    use_cash:    bool  = False,
    use_lstm:    bool  = False,
    total_steps: int   = TOTAL_STEPS,
    lambda_risk: float = LAMBDA_RISK,
):
    os.makedirs(MODELS_DIR, exist_ok=True)

    win_id   = split_cfg["id"]
    name     = model_name_str(reward_type, win_id, seed, use_cash, use_lstm)
    save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    policy_str = "LSTM" if use_lstm else "MLP"
    cash_str   = " +CASH" if use_cash else ""
    print(f"\n{'='*65}")
    print(f"Training: {reward_type} [{policy_str}]  W{win_id}  seed={seed}{cash_str}")
    print(f"  Train : {split_cfg['train_start']} to {split_cfg['train_end']}")
    print(f"  Eval  : {split_cfg['test_start']}  to {split_cfg['test_end']}")
    print(f"{'='*65}")

    train_env = make_env(train_data, reward_type, EPISODE_LEN, use_cash, lambda_risk)
    eval_env  = make_env(
        eval_data, reward_type,
        episode_len = len(next(iter(eval_data.values()))),
        use_cash    = use_cash,
        lambda_risk = lambda_risk,
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

    final_path = os.path.join(MODELS_DIR, f"{name}_final.zip")
    model.save(final_path)
    print(f"\nSaved final : {final_path}")
    print(f"Saved best  : {os.path.join(save_dir, 'best_model.zip')}")
    return final_path, os.path.join(save_dir, "best_model.zip")


def main():
    parser = argparse.ArgumentParser(description="Train PPO portfolio agent")
    parser.add_argument("--reward", type=str, default="all",
                        choices=REWARD_TYPES + ["all"])
    parser.add_argument("--window", type=int, default=0, choices=[0, 1, 2],
                        help="0 = all windows")
    parser.add_argument("--seed", type=int, default=0,
                        help="0 = all seeds (42, 123, 777); else specific seed")
    parser.set_defaults(cash=True)
    parser.add_argument("--no-cash", dest="cash", action="store_false",
                        help="Disable CASH asset (cash is enabled by default)")
    parser.add_argument("--lstm", action="store_true",
                        help="Use LSTM (RecurrentPPO) instead of MLP")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--lambda_risk", type=float, default=LAMBDA_RISK)
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Override models output directory")
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
    cash_str   = " +CASH" if args.cash else ""

    print(f"[train] Policy={policy_str} | Assets=BTC/ETH/BNB{cash_str}")
    print(f"[train] Rewards={rewards} | Windows={[w['id'] for w in windows]} | Seeds={seeds}")
    print(f"[train] Total runs: {total_runs} x {args.steps} steps")

    print("\n[train] Loading data...")
    all_data = load_all_data(normalize=True, tickers=DEFAULT_TICKERS)

    run = 0
    for split_cfg in windows:
        train_data, eval_data = get_split(all_data, split_cfg)
        for reward_type in rewards:
            for seed in seeds:
                run += 1
                print(f"\n[Run {run}/{total_runs}]")
                train_one(
                    reward_type = reward_type,
                    split_cfg   = split_cfg,
                    train_data  = train_data,
                    eval_data   = eval_data,
                    seed        = seed,
                    use_cash    = args.cash,
                    use_lstm    = args.lstm,
                    total_steps = args.steps,
                    lambda_risk = args.lambda_risk,
                )

    print(f"\n[train] All {total_runs} runs complete.")


if __name__ == "__main__":
    main()
