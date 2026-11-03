"""
🧪 test_reward.py — Unit tests for the RewardShaper
=====================================================
Validates that each reward component fires correctly
and that shaped rewards are numerically sane.

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464

Run: python test_reward.py
     python -m pytest test_reward.py -v
"""

import numpy as np
import pytest
from reward import RewardShaper, GROUND_THRESHOLD


def _obs(x=0.0, y=0.5, vx=0.0, vy=0.0, tilt=0.0,
         ang_vel=0.0, left=0.0, right=0.0) -> np.ndarray:
    """Helper to build a fake observation array."""
    return np.array([x, y, vx, vy, tilt, ang_vel, left, right], dtype=np.float32)


class TestRewardShaper:

    def setup_method(self):
        self.shaper = RewardShaper()
        self.shaper.reset(_obs(x=0.5, y=0.5))

    def test_proximity_bonus_when_closing(self):
        """Moving toward the pad should give positive proximity bonus."""
        obs = _obs(x=0.4, y=0.4)   # closer than initial (0.5, 0.5)
        reward = self.shaper.compute(obs, base_reward=0.0)
        assert reward > 0.0, "Should earn bonus for closing distance"

    def test_proximity_penalty_when_drifting_away(self):
        """Moving away from the pad should give negative proximity delta."""
        obs = _obs(x=0.9, y=0.9)
        reward = self.shaper.compute(obs, base_reward=0.0)
        assert reward < 0.0, "Should be penalised for moving away"

    def test_tilt_penalty_applied(self):
        """Leaning rocket should receive a tilt penalty."""
        obs_straight = _obs(tilt=0.0)
        obs_leaning  = _obs(tilt=1.0)
        self.shaper.reset(obs_straight)
        r_straight = self.shaper.compute(obs_straight, 0.0)
        self.shaper.reset(obs_leaning)
        r_leaning  = self.shaper.compute(obs_leaning, 0.0)
        assert r_leaning < r_straight, "Leaning should give lower reward"

    def test_soft_descent_bonus_near_ground(self):
        """Slow downward movement near ground should earn descent bonus."""
        obs = _obs(y=GROUND_THRESHOLD - 0.05, vy=-0.5)
        reward = self.shaper.compute(obs, base_reward=0.0)
        breakdown = self.shaper.breakdown(obs, 0.0)
        assert breakdown["descent"] > 0.0, "Descent bonus should activate near ground"

    def test_no_descent_bonus_high_altitude(self):
        """Descent bonus should NOT activate far from ground."""
        obs = _obs(y=1.0, vy=-0.5)
        breakdown = self.shaper.breakdown(obs, 0.0)
        assert breakdown["descent"] == 0.0

    def test_base_reward_preserved(self):
        """Base environment reward should always be included in total."""
        obs = _obs()
        self.shaper.reset(obs)
        for base in [100.0, -100.0, 0.0]:
            result = self.shaper.compute(obs, base_reward=base)
            assert result == pytest.approx(base, abs=5.0), \
                "Base reward should dominate at origin with no distance change"

    def test_custom_weights(self):
        """Custom weights should override defaults."""
        shaper = RewardShaper(weights={"tilt": -1.0})
        obs = _obs(tilt=1.0)
        shaper.reset(obs)
        breakdown = shaper.breakdown(obs, 0.0)
        assert breakdown["tilt"] == pytest.approx(-1.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
