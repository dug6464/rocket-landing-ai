"""
🔢 train_numpy.py — Train using the from-scratch NumPy PPO
============================================================
Uses numpy_ppo.NumpyPPO (no PyTorch) to train the lunar lander.
Slower than SB3 but lets you see every weight update in plain Python.

Use train.py for serious training; this file is for learning the algorithm.

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464

Run: python train_numpy.py
     python train_numpy.py --episodes 3000
"""

import argparse
import numpy as np
import gymnasium as gym

from env import ShapedLunarLander, ENV_ID
from numpy_ppo import NumpyPPO

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train rocket lander with NumPy PPO")
parser.add_argument("--episodes",    type=int,   default=2000,  help="Training episodes")
parser.add_argument("--rollout",     type=int,   default=512,   help="Steps per rollout")
parser.add_argument("--lr",          type=float, default=1e-3,  help="Learning rate")
parser.add_argument("--print-every", type=int,   default=100,   help="Log every N episodes")
parser.add_argument("--seed",        type=int,   default=42)
args = parser.parse_args()

np.random.seed(args.seed)


def collect_rollout(env, agent: NumpyPPO, n_steps: int) -> dict:
    """Run the agent for n_steps steps and collect a rollout buffer."""
    obs_buf, act_buf, prob_buf = [], [], []
    rew_buf, val_buf, done_buf = [], [], []

    obs, _ = env.reset()
    for _ in range(n_steps):
        action, prob = agent.select_action(obs)
        value        = agent.get_value(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_buf.append(obs); act_buf.append(action)
        prob_buf.append(prob); rew_buf.append(reward)
        val_buf.append(value); done_buf.append(float(done))

        obs = next_obs if not done else env.reset()[0]

    advantages, returns = agent.compute_gae(rew_buf, val_buf, done_buf)
    return {"obs": obs_buf, "actions": act_buf, "old_probs": prob_buf,
            "advantages": advantages, "returns": returns}


if __name__ == "__main__":
    env   = gym.make(ENV_ID)
    env   = ShapedLunarLander(env)
    obs_s = env.observation_space.shape[0]   # 8
    act_s = env.action_space.n               # 4

    agent = NumpyPPO(obs_dim=obs_s, action_dim=act_s, lr=args.lr)

    print("🔢  NumPy PPO — Rocket Lander (educational)")
    print(f"    Obs dim : {obs_s}  |  Action dim : {act_s}")
    print(f"    Episodes: {args.episodes}  |  Rollout steps: {args.rollout}")
    print("-" * 48)

    episode_rewards = []
    ep = 0
    while ep < args.episodes:
        rollout = collect_rollout(env, agent, args.rollout)
        loss    = agent.update(rollout)

        # Estimate episode reward from rollout
        chunk_reward = float(np.sum(rollout["returns"][:args.rollout]) / max(1, args.rollout // 200))
        episode_rewards.append(chunk_reward)
        ep += max(1, args.rollout // 200)

        if ep % args.print_every < (args.rollout // 200):
            recent = np.mean(episode_rewards[-20:])
            print(f"  Ep ~{ep:>5} │ mean reward (last 20): {recent:>8.1f} │ loss: {loss:.4f}")

    env.close()
    print(f"\n✅  NumPy training done — {ep} episodes")
    print("    (For real performance, use: python train.py)")
