"""
🚀 Rocket Lander — Reinforcement Learning with PPO + Reward Shaping
===================================================================
Trains an AI agent to land a rocket using:
  • Gymnasium's LunarLander-v3 physics simulator
  • PPO (Proximal Policy Optimization) — the same family used by Tesla for robot training
  • Custom reward shaping to steer learning faster

Run: python train.py
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os

# ──────────────────────────────────────────────
# 1. REWARD SHAPING WRAPPER
#    The raw LunarLander reward is OK, but we
#    add extra signal so the agent learns faster:
#      +  proximity bonus  → encourages moving toward pad
#      +  tilt penalty     → penalises leaning dangerously
#      +  velocity penalty → rewards gentle descent speed
# ──────────────────────────────────────────────
class ShapedLunarLander(gym.Wrapper):
    """
    Wraps LunarLander-v3 and injects extra reward signals
    on top of the built-in score:

    Observation index map (from Gymnasium docs):
        0  x position          (–1 … 1)
        1  y position          (–1 … 1)
        2  x velocity
        3  y velocity
        4  angle (tilt)
        5  angular velocity
        6  left leg contact    (bool)
        7  right leg contact   (bool)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_x = 0.0
        self._prev_y = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_x = float(obs[0])
        self._prev_y = float(obs[1])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x, y = float(obs[0]), float(obs[1])
        vx, vy = float(obs[2]), float(obs[3])
        tilt = float(obs[4])

        # ── Proximity bonus ──────────────────────────────────────────────
        # Distance to the landing pad (always at origin, x=0, y=0)
        dist_now  = np.sqrt(x**2 + y**2)
        dist_prev = np.sqrt(self._prev_x**2 + self._prev_y**2)
        proximity_bonus = 0.5 * (dist_prev - dist_now)   # +reward if getting closer

        # ── Tilt penalty ─────────────────────────────────────────────────
        # Penalise leaning; abs(tilt) near π means sideways/upside-down
        tilt_penalty = -0.3 * abs(tilt)

        # ── Velocity penalty (horizontal) ────────────────────────────────
        # Discourage wild horizontal drift
        vx_penalty = -0.1 * abs(vx)

        # ── Soft descent bonus ───────────────────────────────────────────
        # Reward controlled downward velocity when close to the ground
        if y < 0.3 and vy < 0:
            descent_bonus = 0.2 * min(-vy, 1.0)   # up to +0.2 for slow descent
        else:
            descent_bonus = 0.0

        shaped_reward = reward + proximity_bonus + tilt_penalty + vx_penalty + descent_bonus

        self._prev_x = x
        self._prev_y = y

        return obs, shaped_reward, terminated, truncated, info


# ──────────────────────────────────────────────
# 2. ENVIRONMENT FACTORY
# ──────────────────────────────────────────────
def make_env(rank: int = 0, seed: int = 42):
    """Create a single wrapped + monitored environment."""
    def _init():
        env = gym.make("LunarLander-v3")
        env = ShapedLunarLander(env)
        env = Monitor(env, filename=None)
        env.reset(seed=seed + rank)
        return env
    return _init


# ──────────────────────────────────────────────
# 3. TRAINING CONFIG
# ──────────────────────────────────────────────
N_ENVS          = 4          # parallel environments (adjust to your CPU count)
TOTAL_TIMESTEPS = 1_000_000  # ~15–20 min on a modern laptop; raise to 2M for perfection
LOG_DIR         = "logs"
MODEL_SAVE_DIR  = "models"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


# ──────────────────────────────────────────────
# 4. BUILD AGENT  (PPO with a 3-layer MLP)
#    PPO is the algorithm Tesla uses (FSD / Optimus).
#    It clips gradient updates so training stays stable.
# ──────────────────────────────────────────────
def build_agent(env) -> PPO:
    return PPO(
        policy          = "MlpPolicy",         # multi-layer perceptron brain
        env             = env,
        learning_rate   = 3e-4,                # Adam step size
        n_steps         = 2048,                # steps per rollout per env
        batch_size      = 64,                  # mini-batch size for gradient updates
        n_epochs        = 10,                  # passes over each rollout
        gamma           = 0.99,                # future-reward discount
        gae_lambda      = 0.95,                # GAE smoothing (reduces variance)
        clip_range      = 0.2,                 # PPO trust-region clip
        ent_coef        = 0.01,                # entropy bonus → keeps exploring
        vf_coef         = 0.5,                 # value function loss weight
        max_grad_norm   = 0.5,                 # gradient clipping
        policy_kwargs   = {
            "net_arch": [256, 256, 128],       # 3 hidden layers
        },
        tensorboard_log = LOG_DIR,
        verbose         = 1,
    )


# ──────────────────────────────────────────────
# 5. CALLBACKS
#    • EvalCallback  → saves the best model automatically
#    • CheckpointCallback → saves snapshots every 100k steps
# ──────────────────────────────────────────────
def build_callbacks():
    eval_env = make_vec_env(make_env(seed=0), n_envs=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_PATH,
        log_path             = LOG_DIR,
        eval_freq            = 25_000,         # evaluate every 25k steps
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = CHECKPOINT_PATH,
        name_prefix = "rocket_ppo",
    )

    return [eval_cb, checkpoint_cb]


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀  Rocket Lander — PPO + Reward Shaping")
    print(f"    Environments : {N_ENVS} parallel")
    print(f"    Total steps  : {TOTAL_TIMESTEPS:,}")
    print(f"    Models saved : {MODEL_SAVE_DIR}/")
    print(f"    TensorBoard  : tensorboard --logdir {LOG_DIR}")
    print("-" * 50)

    # Vectorised parallel environments
    train_env = make_vec_env(make_env(seed=42), n_envs=N_ENVS)

    # Build PPO agent
    model = build_agent(train_env)

    # Train!
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = build_callbacks(),
        progress_bar    = True,
    )

    # Save final model
    final_path = os.path.join(MODEL_SAVE_DIR, "rocket_ppo_final")
    model.save(final_path)
    print(f"\n✅  Training complete! Final model saved to: {final_path}.zip")
    print(f"    Run  python watch.py  to see your rocket fly! 🛸")
