"""
🛸 Rocket Lander — Watch your trained AI fly
=============================================
Loads the best saved model and renders it landing rockets in real-time.

👨‍💻 Developed by Tolosa  |  GitHub: https://github.com/dug6464

Run: python watch.py
     python watch.py --model models/rocket_ppo_final   (pick a specific checkpoint)
     python watch.py --episodes 20                     (run more episodes)
"""

import argparse
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Watch the trained rocket lander")
parser.add_argument("--model",    default="models/best_model/best_model",
                    help="Path to saved model (without .zip)")
parser.add_argument("--episodes", type=int, default=10,
                    help="Number of episodes to render")
parser.add_argument("--no-shaped-reward", action="store_true",
                    help="Use raw environment (no wrapper) for evaluation")
args = parser.parse_args()

# ── Load model ─────────────────────────────────────────────────────────────────
model_path = args.model
if not os.path.exists(model_path + ".zip"):
    alt = "models/rocket_ppo_final"
    if os.path.exists(alt + ".zip"):
        print(f"⚠️  {model_path}.zip not found — falling back to {alt}.zip")
        model_path = alt
    else:
        print("❌  No trained model found. Run  python train.py  first.")
        raise SystemExit(1)

print(f"🧠  Loading model: {model_path}.zip")
model = PPO.load(model_path)

# ── Environment ────────────────────────────────────────────────────────────────
env = gym.make("LunarLander-v3", render_mode="human")

# ── Statistics ─────────────────────────────────────────────────────────────────
rewards   = []
successes = []   # episode ended with both legs on pad

print(f"\n🚀  Launching {args.episodes} landing attempts …  (close the window to stop)\n")
print(f"{'Episode':>8}  {'Reward':>10}  {'Steps':>7}  {'Result'}")
print("─" * 42)

for ep in range(1, args.episodes + 1):
    obs, _ = env.reset()
    total_reward = 0.0
    steps        = 0
    done         = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps        += 1
        done = terminated or truncated

    landed  = bool(obs[6] or obs[7])   # leg contact sensor
    success = landed and total_reward > 200
    result  = "🟢 LANDED" if success else ("🟡 SURVIVED" if total_reward > 0 else "🔴 CRASHED")

    rewards.append(total_reward)
    successes.append(success)

    print(f"{ep:>8}  {total_reward:>10.1f}  {steps:>7}  {result}")

env.close()

# ── Summary ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 42)
print(f"  Mean reward : {np.mean(rewards):.1f}")
print(f"  Best reward : {np.max(rewards):.1f}")
print(f"  Success rate: {100 * np.mean(successes):.0f}%  ({sum(successes)}/{args.episodes} landings)")
