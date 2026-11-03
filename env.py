"""
🌍 env.py — Environment setup & reward-shaping wrapper
=======================================================
Centralises all environment creation so every script
(train, evaluate, watch) uses identical settings.

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464
"""

import gymnasium as gym
from gymnasium import Env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv

from reward import RewardShaper

# ── Default environment ID ────────────────────────────────────────────────────
ENV_ID = "LunarLander-v3"


class ShapedLunarLander(gym.Wrapper):
    """
    Wraps LunarLander-v3 and injects shaped rewards every step.

    Observation layout (Gymnasium docs):
        [0] x position      [4] angle (tilt)
        [1] y position      [5] angular velocity
        [2] x velocity      [6] left leg contact
        [3] y velocity      [7] right leg contact
    """

    def __init__(self, env: Env, reward_weights: dict = None):
        super().__init__(env)
        self._shaper = RewardShaper(weights=reward_weights)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._shaper.reset(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = self._shaper.compute(obs, reward)
        info["raw_reward"]    = reward
        info["shaped_reward"] = shaped
        return obs, shaped, terminated, truncated, info


def _make_single_env(seed: int = 42, reward_weights: dict = None, render_mode: str = None):
    """Factory that returns a callable producing one wrapped env."""
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env = ShapedLunarLander(env, reward_weights=reward_weights)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def make_env(
    n_envs: int = 4,
    seed: int = 42,
    reward_weights: dict = None,
) -> VecEnv:
    """
    Create a vectorised training environment.

    Args:
        n_envs:         Number of parallel environments.
        seed:           Base random seed (each env gets seed + rank).
        reward_weights: Optional override for RewardShaper weights.
    Returns:
        Vectorised SB3 environment ready for PPO.
    """
    return make_vec_env(
        _make_single_env(seed=seed, reward_weights=reward_weights),
        n_envs=n_envs,
    )


def make_eval_env(seed: int = 0, reward_weights: dict = None) -> VecEnv:
    """Single-env vectorised environment used for evaluation callbacks."""
    return make_vec_env(
        _make_single_env(seed=seed, reward_weights=reward_weights),
        n_envs=1,
    )


def make_render_env() -> Env:
    """Unwrapped (non-vectorised) env with human rendering for watch.py."""
    env = gym.make(ENV_ID, render_mode="human")
    env = ShapedLunarLander(env)
    return env
