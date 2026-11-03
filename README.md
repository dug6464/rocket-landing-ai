# 🚀 Rocket Landing AI

Train a neural network to land a rocket using **Proximal Policy Optimization (PPO)** — the same reinforcement learning technique used by Tesla to train its robots — running entirely on your laptop with free open-source tools.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-orange)
![Stable Baselines3](https://img.shields.io/badge/SB3-PPO-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎬 What it does

The AI starts with zero knowledge. Through thousands of trial landings it discovers — entirely on its own — how to:

- Point downward
- Slow its descent
- Correct horizontal drift
- Touch down gently on the landing pad

All guided by a **reward function**, not hand-coded rules.

---

## 🧠 Key Concept: Reward Shaping

Instead of telling the rocket *how* to fly, we score what it does every frame:

| Signal | Points |
|--------|--------|
| Moving closer to pad | +0.5 per unit |
| Upright posture | up to +0.3 |
| Controlled horizontal speed | up to +0.1 |
| Soft descent near ground | up to +0.2 |
| Safe landing | **+100** |
| Crash | **–100** |

The AI maximises its total score across thousands of episodes. That's it.

---

## 📁 Project Structure

```
rocket-landing-ai/
├── train.py           ← Train the agent (start here)
├── watch.py           ← Render your trained rocket in real-time
├── plot_progress.py   ← Plot the reward curve after training
├── explain_reward.py  ← Visual breakdown of the reward function
└── requirements.txt   ← All dependencies
```

---

## ⚡ Quick Start

### 1. Install dependencies

> **Windows users:** Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first (needed for Box2D physics).

```bash
pip install -r requirements.txt
```

### 2. Train the AI (~15–20 min on a laptop)

```bash
python train.py
```

Watch live stats in TensorBoard (open a second terminal):

```bash
tensorboard --logdir logs
# then visit http://localhost:6006
```

### 3. Watch it land

```bash
python watch.py
```

### 4. Plot the learning curve

```bash
python plot_progress.py
```

---

## 📈 Expected Training Progress

| Timesteps | Mean Reward | Behaviour |
|-----------|-------------|-----------|
| 0 | –200 | Random crashes |
| 100k | –50 | Starts pointing down |
| 300k | +50 | Reaches the pad area |
| 600k | +150 | Occasional soft landings |
| **1M** | **+200–250** | **Consistent landings ✅** |

---

## 🤖 The Algorithm: PPO

**Proximal Policy Optimization** (PPO) is the workhorse of modern robot training:

```
1. Collect experience (rollouts across parallel environments)
2. Compute advantage: how much better was this action vs. average?
3. Update the neural network — but CLIP the update so it never
   changes too drastically:  clip(π_new / π_old, 0.8, 1.2)
4. Repeat
```

Tesla uses PPO (or a close variant) for Full Self-Driving and Optimus robot motor control. This project runs the same math at laptop scale.

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| [Gymnasium](https://gymnasium.farama.org/) | LunarLander-v3 physics simulator |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | PPO implementation |
| [PyTorch](https://pytorch.org/) | Neural network backend |
| [TensorBoard](https://www.tensorflow.org/tensorboard) | Training visualisation |
| [Matplotlib](https://matplotlib.org/) | Reward curve plots |

---

## 🎛️ Tuning Tips

| What to change | File | Effect |
|----------------|------|--------|
| `N_ENVS` | `train.py` | More cores → faster training |
| `TOTAL_TIMESTEPS` | `train.py` | Raise to 2M for near-perfect |
| `net_arch` | `train.py` PPO kwargs | Larger network = more expressive |
| `ent_coef` | `train.py` PPO kwargs | Higher = more exploration |
| Reward weights | `ShapedLunarLander.step()` | Tune landing style |

---

## 📄 License

MIT — do whatever you want with it.
