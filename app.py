"""
🚀 app.py — Interactive Rocket Landing Web App (Gradio)
=========================================================
Web interface to watch the AI land rockets in real-time.
Ready for deployment on Hugging Face Spaces or local browser.

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464

Run locally:
    python app.py
"""

import os
import tempfile
import numpy as np
import gymnasium as gym
import imageio
import gradio as gr
from stable_baselines3 import PPO

from env import ShapedLunarLander, ENV_ID

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATHS = [
    "models/best_model/best_model.zip",
    "models/rocket_ppo_final.zip",
]

def get_available_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return p.replace(".zip", "")
    return None

# ── Simulation & Video Generation ─────────────────────────────────────────────
def simulate_landing(seed: int, gravity: float, wind_power: float):
    """
    Runs an episode in LunarLander-v3, records frames, and computes telemetry.
    Returns: (video_path, summary_text, telemetry_plot_path)
    """
    model_path = get_available_model()
    model = None
    if model_path:
        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")

    # Set up environment with custom physics parameters
    env = gym.make(
        ENV_ID,
        render_mode="rgb_array",
        gravity=gravity,
        enable_wind=(wind_power > 0.0),
        wind_power=wind_power,
    )
    env = ShapedLunarLander(env)

    obs, _ = env.reset(seed=int(seed) if seed is not None else None)
    frames = []
    
    positions_x = []
    positions_y = []
    velocities_y = []
    tilts = []
    rewards_history = []
    
    done = False
    total_reward = 0.0
    steps = 0
    max_steps = 1000

    while not done and steps < max_steps:
        frame = env.render()
        frames.append(frame)

        # Telemetry logging
        positions_x.append(float(obs[0]))
        positions_y.append(float(obs[1]))
        velocities_y.append(float(obs[3]))
        tilts.append(float(obs[4]))

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Fallback heuristic if trained model is not yet compiled
            # P-controller for landing:
            x, y, vx, vy, angle, ang_vel = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
            if abs(angle) > 0.15:
                action = 1 if angle > 0 else 3  # adjust tilt
            elif vy < -0.4:
                action = 2  # main booster
            else:
                action = 0  # hover / do nothing

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        rewards_history.append(total_reward)
        steps += 1
        done = terminated or truncated

    env.close()

    # Determine outcome
    left_leg, right_leg = bool(obs[6]), bool(obs[7])
    is_success = (left_leg or right_leg) and total_reward > 150
    status = "🟢 SAFE LANDING" if is_success else ("🟡 TOUCHDOWN / ROUGH" if total_reward > 0 else "🔴 CRASHED")

    # Render video
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"landing_{np.random.randint(10000, 99999)}.mp4")
    
    try:
        imageio.mimsave(video_path, frames, fps=30)
    except Exception:
        # Fallback to GIF if mp4 encoder (ffmpeg) not installed
        video_path = video_path.replace(".mp4", ".gif")
        imageio.mimsave(video_path, frames, fps=25)

    summary = f"""### Flight Summary
- **Outcome:** {status}
- **Total Reward:** `{total_reward:.2f}`
- **Flight Duration:** `{steps}` frames (~`{steps/30:.1f}`s)
- **Left Leg Contact:** `{'Yes' if left_leg else 'No'}` | **Right Leg Contact:** `{'Yes' if right_leg else 'No'}`
- **Model Used:** `{model_path if model else 'Autonomous Fallback Heuristic (Untrained)'}`
"""
    return video_path, summary

# ── Gradio Web UI ─────────────────────────────────────────────────────────────
custom_css = """
.container { max-width: 900px; margin: auto; padding-top: 20px; }
.header-card { text-align: center; margin-bottom: 20px; }
.dev-badge { font-weight: bold; font-size: 1.1em; color: #00b4d8; }
"""

with gr.Blocks(title="🚀 Rocket Landing AI — by dug", css=custom_css) as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            """
            # 🚀 Rocket Landing AI — Interactive Flight Simulator
            
            > **👨‍💻 Developed by dug** &nbsp;|&nbsp; ⭐ **GitHub:** [@dug6464](https://github.com/dug6464/rocket-landing-ai)
            
            Train a neural network with **PPO + Reward Shaping** to touch down smoothly on the launch pad.
            Run real-time flight simulations with variable gravity and crosswinds right in your browser!
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Simulation Environment")
                seed_input = gr.Number(value=42, label="Simulation Seed (Terrain Variation)", precision=0)
                gravity_slider = gr.Slider(minimum=-12.0, maximum=-2.0, value=-10.0, step=0.5, label="Gravity Force (m/s²)")
                wind_slider = gr.Slider(minimum=0.0, maximum=15.0, value=0.0, step=1.0, label="Crosswind Intensity")
                launch_btn = gr.Button("🚀 Launch Simulation", variant="primary", size="lg")

            with gr.Column(scale=1):
                video_output = gr.Video(label="Landing Trajectory Video", autoplay=True)
                stats_output = gr.Markdown()

        launch_btn.click(
            fn=simulate_landing,
            inputs=[seed_input, gravity_slider, wind_slider],
            outputs=[video_output, stats_output],
        )

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #888; font-size: 0.9em;">
            Developed with ❤️ by <strong>dug</strong> • Source code & models on <a href="https://github.com/dug6464/rocket-landing-ai" target="_blank">GitHub: @dug6464</a>
            </div>
            """
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
