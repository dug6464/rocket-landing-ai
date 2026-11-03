"""
📡 callbacks.py — Training callbacks
======================================
Custom SB3 callbacks used during PPO training:
  • RocketEvalCallback  — saves the best model, logs to console
  • ProgressCallback    — prints a summary every N steps
  • build_callbacks()   — convenience builder used by train.py

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import VecEnv

from env import make_eval_env

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR      = "models"
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_model")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
LOG_DIR        = "logs"


class ProgressCallback(BaseCallback):
    """
    Prints a one-line training summary every `print_freq` steps.
    Shows timestep, mean episode reward, and mean episode length.
    """

    def __init__(self, print_freq: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._rewards   = []

    def _on_step(self) -> bool:
        # Collect episode rewards from Monitor info
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.print_freq == 0 and self._rewards:
            mean_r = np.mean(self._rewards[-50:])
            print(f"  ⏱  {self.num_timesteps:>8,} steps │ mean reward (last 50 eps): {mean_r:>8.1f}")

        return True


def build_callbacks(
    eval_freq: int = 25_000,
    checkpoint_freq: int = 100_000,
    n_eval_episodes: int = 10,
    seed: int = 0,
) -> CallbackList:
    """
    Build and return the full callback stack for PPO training.

    Args:
        eval_freq:        Evaluate every N timesteps.
        checkpoint_freq:  Save a checkpoint every N timesteps.
        n_eval_episodes:  Episodes per evaluation run.
        seed:             Seed for the evaluation environment.
    Returns:
        SB3 CallbackList ready to pass to model.learn().
    """
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    eval_env = make_eval_env(seed=seed)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_DIR,
        log_path             = LOG_DIR,
        eval_freq            = eval_freq,
        n_eval_episodes      = n_eval_episodes,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = checkpoint_freq,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "rocket_ppo",
        verbose     = 0,
    )

    progress_cb = ProgressCallback(print_freq=50_000)

    return CallbackList([eval_cb, checkpoint_cb, progress_cb])
