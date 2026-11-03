"""
📊 Rocket Lander — Plot training progress
==========================================
Reads the evaluations.npz file saved by EvalCallback and plots
mean reward over training steps.

👨‍💻 Developed by Tolosa  |  GitHub: https://github.com/dug6464

Run: python plot_progress.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

EVAL_LOG = "logs/evaluations.npz"

if not os.path.exists(EVAL_LOG):
    print(f"❌  {EVAL_LOG} not found. Train first (python train.py) or wait for the first eval checkpoint.")
    raise SystemExit(1)

data = np.load(EVAL_LOG)
timesteps   = data["timesteps"]          # x-axis
results     = data["results"]            # shape: (n_evals, n_eval_episodes)
mean_reward = results.mean(axis=1)
std_reward  = results.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(timesteps, mean_reward, color="#00b4d8", linewidth=2, label="Mean reward")
ax.fill_between(timesteps,
                mean_reward - std_reward,
                mean_reward + std_reward,
                alpha=0.25, color="#00b4d8", label="±1 std dev")

# Reference lines
ax.axhline(200, color="#06d6a0", linestyle="--", linewidth=1.2, label="Solved (≥200)")
ax.axhline(0,   color="#ef233c", linestyle="--", linewidth=0.8, alpha=0.5)

ax.set_xlabel("Training timesteps", fontsize=12)
ax.set_ylabel("Episode reward", fontsize=12)
ax.set_title("🚀  Rocket Lander — PPO Training Progress", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()

out = "training_progress.png"
plt.savefig(out, dpi=150)
print(f"✅  Plot saved to {out}")
plt.show()
