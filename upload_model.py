"""
🤗 upload_model.py — Push model & Space to Hugging Face
========================================================
Uploads your trained rocket lander weights and/or web app to Hugging Face.

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464

Usage:
    # 1. Login to Hugging Face (or pass --token):
    huggingface-cli login

    # 2. Upload model to Model Hub:
    python upload_model.py --repo-id dug6464/rocket-landing-ai --type model

    # 3. Deploy full interactive app to Hugging Face Spaces:
    python upload_model.py --repo-id dug6464/rocket-landing-ai --type space
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo

parser = argparse.ArgumentParser(description="Upload Rocket Lander to Hugging Face")
parser.add_argument("--repo-id", default="dug6464/rocket-landing-ai", help="HF repository ID (username/repo-name)")
parser.add_argument("--type", choices=["model", "space"], default="model", help="Type of repository on HF")
parser.add_argument("--model-path", default="models/best_model/best_model.zip", help="Path to model file")
parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="Hugging Face API token")
args = parser.parse_args()


def create_model_card(repo_id: str) -> str:
    return f"""---
language: en
tags:
- reinforcement-learning
- ppo
- gymnasium
- lunar-lander
- rocket-landing
- stable-baselines3
license: mit
---

# 🚀 Rocket Landing AI — PPO Model

Trained neural network for landing a rocket in Gymnasium's `LunarLander-v3` physics environment using **Proximal Policy Optimization (PPO)** and custom reward shaping.

**👨‍💻 Developed by dug** &nbsp;|&nbsp; ⭐ **GitHub:** [@dug6464](https://github.com/dug6464/rocket-landing-ai)

## 📦 How to Use

```python
from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO
import gymnasium as gym

# 1. Download model
model_path = hf_hub_download(repo_id="{repo_id}", filename="rocket_ppo_best.zip")

# 2. Load model
model = PPO.load(model_path)

# 3. Run simulation
env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
```
"""


def main():
    api = HfApi(token=args.token)
    
    print(f"🚀 Hugging Face Deployment: {args.repo_id} ({args.type})")
    print("-" * 50)

    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type=args.type,
            space_sdk="gradio" if args.type == "space" else None,
            exist_ok=True,
            token=args.token,
        )
        print(f"✅ Repository ready on Hugging Face: https://huggingface.co/{args.type}s/{args.repo_id}")
    except Exception as e:
        print(f"Note on repo creation: {e}")

    if args.type == "model":
        model_file = args.model_path
        if not os.path.exists(model_file):
            alt = "models/rocket_ppo_final.zip"
            if os.path.exists(alt):
                model_file = alt
            else:
                print(f"⚠️ Warning: No trained model found at {model_file}. Run 'python train.py' first.")
                return

        print(f"Uploading model: {model_file}...")
        api.upload_file(
            path_or_fileobj=model_file,
            path_in_repo="rocket_ppo_best.zip",
            repo_id=args.repo_id,
            repo_type="model",
            token=args.token,
        )

        # Upload Model Card README
        card_content = create_model_card(args.repo_id)
        temp_card = "TEMP_MODEL_CARD.md"
        with open(temp_card, "w", encoding="utf-8") as f:
            f.write(card_content)
        
        api.upload_file(
            path_or_fileobj=temp_card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            token=args.token,
        )
        if os.path.exists(temp_card):
            os.remove(temp_card)

        print(f"\n🎉 Model successfully uploaded to: https://huggingface.co/{args.repo_id}")

    elif args.type == "space":
        print(f"Uploading Space application files...")
        files_to_upload = [
            "app.py",
            "env.py",
            "reward.py",
            "requirements.txt",
        ]
        
        # Include model if available
        if os.path.exists(args.model_path):
            files_to_upload.append(args.model_path)

        for fpath in files_to_upload:
            if os.path.exists(fpath):
                print(f"  • Uploading {fpath}...")
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=fpath,
                    repo_id=args.repo_id,
                    repo_type="space",
                    token=args.token,
                )

        print(f"\n🎉 Space live at: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
