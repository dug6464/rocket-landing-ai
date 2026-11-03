"""
🛸 watch.py — Watch your trained rocket land in real time
==========================================================
Renders the agent using the trained PPO model with a live window.

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464

Run: python watch.py
     python watch.py --model models/rocket_ppo_final
     python watch.py --episodes 20
"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO

from env import make_render_env

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Watch the trained rocket lander")
parser.add_argument("--model",    default="models/best_model/best_model")
parser.add_argument("--episodes", type=int, default=10)
args = parser.parse_args()


def load_model(path: str) -> PPO:
    for p in [path, "models/rocket_ppo_final"]:
        if os.path.exists(p + ".zip"):
            print(f"🧠  Loading: {p}.zip")
            return PPO.load(p)
    print("❌  No model found. Run  python train.py  first.")
    raise SystemExit(1)


if __name__ == "__main__":
    model = load_model(args.model)
    env   = make_render_env()

    rewards, successes = [], []
    print(f"\n🚀  Launching {args.episodes} landing attempts …\n")
    print(f"{'Ep':>4}  {'Reward':>10}  {'Steps':>6}  Result")
    print("─" * 36)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total, steps, done = 0.0, 0, False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated

        success = (bool(obs[6]) or bool(obs[7])) and total > 200
        label   = "🟢 LANDED" if success else ("🟡 OK" if total > 0 else "🔴 CRASH")
        rewards.append(total); successes.append(success)
        print(f"{ep:>4}  {total:>10.1f}  {steps:>6}  {label}")

    env.close()
    print(f"\n  Mean: {np.mean(rewards):.1f}  │  "
          f"Best: {np.max(rewards):.1f}  │  "
          f"Success: {100*np.mean(successes):.0f}% ({sum(successes)}/{args.episodes})")
