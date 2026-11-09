# 🚀 Rocket Landing AI

Train a neural network to land a rocket using **Proximal Policy Optimization (PPO)** — the same reinforcement learning technique used by Tesla to train its robots — running on your laptop, in the cloud, or deployed as an interactive web simulator.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dug6464/rocket-landing-ai/blob/master/rocket_lander_colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-orange)
![Stable Baselines3](https://img.shields.io/badge/SB3-PPO-green)
![Gradio](https://img.shields.io/badge/Web_App-Gradio-orange?logo=gradio)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> 👨‍💻 **Developed by dug** &nbsp;|&nbsp; ⭐ GitHub: [@dug6464](https://github.com/dug6464)

---

## 🎬 What it does

The AI starts with zero knowledge. Through trial-and-error simulation, it learns to:
- Orient itself vertically
- Modulate main and lateral thrusters
- Counter balance against crosswinds
- Touch down gently on the designated landing pad

All driven by **Reward Shaping**, transforming raw state dynamics into informative learning gradients.

---

## 🧠 Key Concept: Reward Shaping

Instead of telling the rocket *how* to fly, we score what it does each frame:

| Signal | Adjustment |
|--------|------------|
| Closing distance to pad | `+0.5` per unit |
| Maintaining upright tilt | up to `+0.3` |
| Controlled horizontal drift | up to `+0.1` |
| Soft vertical touchdown velocity | up to `+0.2` |
| **Safe landing** | **`+100`** |
| **Crash or tip-over** | **`-100`** |

---

## 📁 Modular Project Structure

```
rocket-landing-ai/
├── app.py                     ← 🌐 Interactive Gradio web application
├── train.py                   ← 🚀 Stable-Baselines3 PPO training script
├── env.py                     ← 🌍 Environment factory & ShapedLunarLander wrapper
├── reward.py                  ← 🎯 RewardShaper component logic & weights
├── callbacks.py               ← 📡 Training callbacks & automated checkpointing
├── evaluate.py                ← 📊 Comprehensive statistical evaluation suite
├── watch.py                   ← 🛸 Real-time desktop visualization window
├── test_reward.py             ← 🧪 Unit tests for reward calculations
├── numpy_ppo.py               ← 🔢 Complete PPO from scratch (pure NumPy)
├── train_numpy.py             ← 🎓 Educational scratch-training pipeline
├── upload_model.py            ← 🤗 Hugging Face Hub / Space deployment script
├── rocket_lander_colab.ipynb  ← 📓 1-Click Google Colab notebook
├── Dockerfile                 ← 🐳 Production Docker image definition
├── docker-compose.yml         ← 🐳 Docker compose for 1-command startup
├── requirements.txt           ← 📦 Project dependencies
└── README.md                  ← 📖 Project documentation
```

---

## ⚡ Quick Start

### 1. Install dependencies

> **Windows users:** Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first (for Box2D physics).

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Web Simulator

```bash
python app.py
```
Open **http://localhost:7860** in your browser to launch simulations with custom gravity and crosswinds!

### 3. Train the AI (~15–20 min on a laptop)

```bash
python train.py --timesteps 1000000 --envs 4
```

Watch live progress in TensorBoard:
```bash
tensorboard --logdir logs
```

### 4. Watch Desktop Simulation & Evaluate

```bash
python watch.py
python evaluate.py --episodes 30
```

---

## 🚢 Deployment Guide (4 Ways)

### 1. 🌐 Hugging Face Spaces (Web App)
Deploy the live browser simulator for free:
```bash
# Set your token or login
huggingface-cli login

# Deploy as an interactive web Space
python upload_model.py --repo-id dug6464/rocket-landing-ai --type space
```
Your simulator will be live at: `https://huggingface.co/spaces/dug6464/rocket-landing-ai`

### 2. 🤗 Hugging Face Model Hub (Model Weights)
Upload your trained neural network weights so others can import it with 1 line of code:
```bash
python upload_model.py --repo-id dug6464/rocket-landing-ai --type model
```

### 3. 📓 Google Colab (Zero-Install Cloud Training)
Run without installing anything on your PC:
Click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dug6464/rocket-landing-ai/blob/master/rocket_lander_colab.ipynb) to open the interactive notebook, train on cloud GPUs, and render videos directly in the browser.

### 4. 🐳 Docker & Docker Compose
Containerized deployment with zero dependency headaches:
```bash
# Build & start the container
docker compose up --build
```
Access the web app at `http://localhost:7860`.

---

## 📈 Expected Training Milestones

| Timesteps | Mean Reward | Observed Behavior |
|-----------|-------------|-------------------|
| 0 | –200 | Erratic thruster bursts, frequent crashes |
| 100k | –50 | Orients bottom thruster downward |
| 300k | +50 | Navigates towards pad coordinates |
| 600k | +150 | Smooth descent, occasional soft touchdown |
| **1M** | **+200–250** | **Precise, upright landing sequence ✅** |

---

## 📄 License

MIT — do whatever you want with it.

---

## 👨‍💻 Author & Developer

<table>
  <tr>
    <td align="center">
      <strong>dug</strong><br/>
      Creator & Developer<br/><br/>
      <a href="https://github.com/dug6464">
        <img src="https://img.shields.io/badge/GitHub-dug6464-181717?style=for-the-badge&logo=github" alt="GitHub: dug6464"/>
      </a><br/><br/>
      <sub>🤖 Reinforcement Learning • 🚀 Robotics & Control • 🧠 Deep Learning</sub>
    </td>
  </tr>
</table>

**Tech stack used in this project:**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?style=flat)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat&logo=gradio&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)

---

<p align="center">
  Developed with ❤️ by <strong>dug</strong> &nbsp;•&nbsp;
  <a href="https://github.com/dug6464">GitHub: dug6464</a>
</p>
