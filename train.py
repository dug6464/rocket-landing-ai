"""
🚀 train.py — Train the rocket lander with PPO
===============================================
Entry point for Stable-Baselines3 PPO training.
Imports clean modules: env, callbacks.

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464

Run: python train.py
     python train.py --timesteps 2000000
     python train.py --envs 8
"""

import argparse
import os
from stable_baselines3 import PPO

from env import make_env
from callbacks import build_callbacks, MODEL_DIR, LOG_DIR

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train the rocket lander PPO agent")
parser.add_argument("--timesteps", type=int,   default=1_000_000, help="Total training timesteps")
parser.add_argument("--envs",      type=int,   default=4,         help="Parallel environments")
parser.add_argument("--seed",      type=int,   default=42,        help="Random seed")
parser.add_argument("--lr",        type=float, default=3e-4,      help="Learning rate")
args = parser.parse_args()

os.makedirs(MODEL_DIR, exist_ok=True)


def build_agent(env) -> PPO:
    """Construct the PPO agent with a 3-layer MLP policy."""
    return PPO(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = args.lr,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = {"net_arch": [256, 256, 128]},
        tensorboard_log = LOG_DIR,
        verbose         = 1,
    )


if __name__ == "__main__":
    print("🚀  Rocket Lander — PPO Training")
    print(f"    Timesteps    : {args.timesteps:,}")
    print(f"    Environments : {args.envs}")
    print(f"    Seed         : {args.seed}")
    print(f"    TensorBoard  : tensorboard --logdir {LOG_DIR}")
    print("-" * 48)

    train_env = make_env(n_envs=args.envs, seed=args.seed)
    model     = build_agent(train_env)

    model.learn(
        total_timesteps = args.timesteps,
        callback        = build_callbacks(seed=args.seed),
        progress_bar    = True,
    )

    final = os.path.join(MODEL_DIR, "rocket_ppo_final")
    model.save(final)
    print(f"\n✅  Training complete — model saved to {final}.zip")
    print("    Run  python watch.py  to see it land! 🛸")
