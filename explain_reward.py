"""
🔬 Rocket Lander — Reward shaping visualiser
=============================================
Generates a chart explaining how the shaped reward is composed
during a simulated landing trajectory. Educational / demo use.

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464

Run: python explain_reward.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Simulate a 200-step landing trajectory ────────────────────────────────────
np.random.seed(7)
steps = 200

# Smooth trajectory from (0.6, 0.9) → (0.0, 0.0)
t     = np.linspace(0, 1, steps)
x     = 0.6 * (1 - t) + np.random.normal(0, 0.02, steps)
y     = 0.9 * (1 - t)**0.7 + np.random.normal(0, 0.01, steps)
vx    = np.gradient(x) * steps
vy    = np.gradient(y) * steps
tilt  = 0.4 * np.sin(3 * np.pi * t) * (1 - t)   # wobble that settles

# ── Reward components ──────────────────────────────────────────────────────────
dist   = np.sqrt(x**2 + y**2)
prox   =  0.5 * np.maximum(-np.gradient(dist) * steps, 0)   # positive when closing
tp     = -0.3 * np.abs(tilt)
vxp    = -0.1 * np.abs(vx)
desc   = np.where((y < 0.3) & (vy < 0), 0.2 * np.minimum(-vy, 1.0), 0.0)
base   = -0.3 + 3 * (1 - dist) * t              # crude base reward approximation

total  = base + prox + tp + vxp + desc

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
fig.suptitle("🚀  Reward Shaping — What the Agent Earns Each Frame",
             fontsize=14, fontweight="bold")

# Panel 1: trajectory metrics
ax = axes[0]
ax.plot(x,    label="x position",   color="#4cc9f0")
ax.plot(y,    label="y position",   color="#f72585")
ax.plot(tilt, label="tilt (angle)", color="#ffd166", linestyle="--")
ax.set_ylabel("Value")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_title("Rocket state over time")

# Panel 2: reward components
ax = axes[1]
ax.plot(prox, label="Proximity bonus (+)",  color="#06d6a0")
ax.plot(tp,   label="Tilt penalty (–)",     color="#ef233c")
ax.plot(vxp,  label="Horiz. vel. penalty (–)", color="#f77f00")
ax.plot(desc, label="Soft descent bonus (+)", color="#7209b7")
ax.axhline(0, color="white", linewidth=0.6, alpha=0.5)
ax.set_ylabel("Shaped Δreward")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_title("Shaped reward components")

# Panel 3: cumulative reward
ax = axes[2]
ax.plot(np.cumsum(total), color="#00b4d8", linewidth=2, label="Cumulative reward")
ax.fill_between(range(steps), 0, np.cumsum(total), alpha=0.2, color="#00b4d8")
ax.axhline(200, color="#06d6a0", linestyle="--", linewidth=1.2, label="Solved threshold (200)")
ax.set_xlabel("Timestep")
ax.set_ylabel("Cumulative reward")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_title("Cumulative episode reward")

plt.tight_layout()
out = "reward_explanation.png"
plt.savefig(out, dpi=150)
print(f"✅  Saved to {out}")
plt.show()
