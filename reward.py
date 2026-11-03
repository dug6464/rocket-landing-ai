"""
🎯 reward.py — Reward shaping logic
=====================================
Standalone reward shaper used by ShapedLunarLander (env.py).
All reward weights live here so they're easy to tune in one place.

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464
"""

import numpy as np


# ── Tunable reward weights ────────────────────────────────────────────────────
WEIGHTS = {
    "proximity":    0.5,   # reward for closing distance to pad per frame
    "tilt":        -0.3,   # penalty per radian of lean
    "horiz_vel":   -0.1,   # penalty for horizontal drift
    "soft_descent": 0.2,   # bonus for slow downward movement near ground
    "landing":    100.0,   # one-time bonus for touching down safely
    "crash":     -100.0,   # one-time penalty for crashing
}

# Threshold: y-position below which soft-descent bonus activates
GROUND_THRESHOLD = 0.3


class RewardShaper:
    """
    Computes shaped reward signals every environment step.

    Usage:
        shaper = RewardShaper()
        shaper.reset()
        delta = shaper.compute(obs_prev, obs_curr, base_reward)
    """

    def __init__(self, weights: dict = None):
        self.w = {**WEIGHTS, **(weights or {})}
        self._prev_dist = 0.0

    def reset(self, initial_obs: np.ndarray = None):
        """Call at the start of each episode."""
        if initial_obs is not None:
            self._prev_dist = float(np.sqrt(initial_obs[0] ** 2 + initial_obs[1] ** 2))
        else:
            self._prev_dist = 0.0

    def compute(self, obs: np.ndarray, base_reward: float) -> float:
        """
        Args:
            obs:         Current observation vector (8 elements).
            base_reward: Raw reward from the environment.
        Returns:
            Total shaped reward for this step.
        """
        x,  y  = float(obs[0]), float(obs[1])
        vx, vy = float(obs[2]), float(obs[3])
        tilt   = float(obs[4])

        # Proximity bonus: reward closing distance to pad (at origin)
        dist_now        = float(np.sqrt(x ** 2 + y ** 2))
        proximity       = self.w["proximity"] * (self._prev_dist - dist_now)
        self._prev_dist = dist_now

        # Tilt penalty
        tilt_penalty = self.w["tilt"] * abs(tilt)

        # Horizontal velocity penalty
        vx_penalty = self.w["horiz_vel"] * abs(vx)

        # Soft descent bonus (only near the ground, moving downward)
        if y < GROUND_THRESHOLD and vy < 0:
            descent = self.w["soft_descent"] * min(-vy, 1.0)
        else:
            descent = 0.0

        return base_reward + proximity + tilt_penalty + vx_penalty + descent

    def breakdown(self, obs: np.ndarray, base_reward: float) -> dict:
        """Return each component separately — useful for plotting / debugging."""
        x,  y  = float(obs[0]), float(obs[1])
        vx, vy = float(obs[2]), float(obs[3])
        tilt   = float(obs[4])
        dist   = float(np.sqrt(x ** 2 + y ** 2))

        return {
            "base":      base_reward,
            "proximity": self.w["proximity"] * (self._prev_dist - dist),
            "tilt":      self.w["tilt"] * abs(tilt),
            "horiz_vel": self.w["horiz_vel"] * abs(vx),
            "descent":   self.w["soft_descent"] * min(-vy, 1.0) if (y < GROUND_THRESHOLD and vy < 0) else 0.0,
        }
