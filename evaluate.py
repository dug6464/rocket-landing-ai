"""
📊 evaluate.py — Full evaluation suite
========================================
Loads a trained model and runs a comprehensive evaluation:
  • Episode reward statistics (mean, std, min, max)
  • Success rate (both-legs landing + reward > 200)
  • Per-episode breakdown table
  • Reward distribution histogram
  • Saves a PNG report

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464

Run: python evaluate.py
     python evaluate.py --model models/rocket_ppo_final --episodes 50
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
import gymnasium as gym

from env import ShapedLunarLander, ENV_ID

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate a trained rocket lander")
parser.add_argument("--model",    default="models/best_model/best_model")
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--seed",     type=int, default=99)
parser.add_argument("--render",   action="store_true", help="Show window while evaluating")
args = parser.parse_args()


def load_model(path: str) -> PPO:
    for p in [path, "models/rocket_ppo_final"]:
        if os.path.exists(p + ".zip"):
            print(f"🧠  Loading: {p}.zip")
            return PPO.load(p)
    print("❌  No model found. Run python train.py first.")
    raise SystemExit(1)


def run_evaluation(model: PPO) -> list[dict]:
    render_mode = "human" if args.render else None
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = ShapedLunarLander(env)

    results = []
    print(f"\n{'Ep':>4}  {'Reward':>10}  {'Steps':>6}  {'L-Leg':>6}  {'R-Leg':>6}  Result")
    print("─" * 56)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        total, steps = 0.0, 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated

        left, right = bool(obs[6]), bool(obs[7])
        success = (left or right) and total > 200
        label   = "🟢 LANDED" if success else ("🟡 OK" if total > 0 else "🔴 CRASH")

        results.append({"reward": total, "steps": steps,
                         "left": left, "right": right, "success": success})
        print(f"{ep:>4}  {total:>10.1f}  {steps:>6}  {'✓' if left else '✗':>6}  {'✓' if right else '✗':>6}  {label}")

    env.close()
    return results


def print_summary(results: list[dict]):
    rewards   = [r["reward"]  for r in results]
    successes = [r["success"] for r in results]
    print("\n" + "═" * 56)
    print("  📈  EVALUATION SUMMARY")
    print("═" * 56)
    print(f"  Episodes    : {len(results)}")
    print(f"  Mean reward : {np.mean(rewards):>8.1f}")
    print(f"  Std reward  : {np.std(rewards):>8.1f}")
    print(f"  Best        : {np.max(rewards):>8.1f}")
    print(f"  Worst       : {np.min(rewards):>8.1f}")
    print(f"  Success rate: {100*np.mean(successes):>7.1f}%  ({sum(successes)}/{len(results)})")
    print("═" * 56)


def save_report(results: list[dict]):
    rewards   = [r["reward"]  for r in results]
    successes = [r["success"] for r in results]

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2)

    # Reward over episodes
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(rewards, color="#00b4d8", linewidth=1.8, marker="o", markersize=3)
    ax1.axhline(np.mean(rewards), color="#ffd166", linestyle="--", label=f"Mean: {np.mean(rewards):.1f}")
    ax1.axhline(200, color="#06d6a0", linestyle="--", linewidth=1, label="Solved (200)")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
    ax1.set_title("Reward per Episode"); ax1.legend(); ax1.grid(alpha=0.3)

    # Histogram
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(rewards, bins=15, color="#4361ee", edgecolor="white", alpha=0.85)
    ax2.axvline(np.mean(rewards), color="#ffd166", linestyle="--", label=f"Mean: {np.mean(rewards):.1f}")
    ax2.axvline(200, color="#06d6a0", linestyle="--", label="Solved")
    ax2.set_xlabel("Reward"); ax2.set_ylabel("Count")
    ax2.set_title(f"Reward Distribution  (success {100*np.mean(successes):.0f}%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("🚀  Rocket Lander — Evaluation Report  |  dug / @dug6464",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "evaluation_report.png"
    plt.savefig(out, dpi=150)
    print(f"\n📊  Report saved → {out}")
    plt.show()


if __name__ == "__main__":
    model   = load_model(args.model)
    results = run_evaluation(model)
    print_summary(results)
    save_report(results)
