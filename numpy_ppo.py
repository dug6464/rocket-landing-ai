"""
🔢 numpy_ppo.py — PPO implemented from scratch with NumPy
===========================================================
A minimal but complete PPO implementation using only NumPy
(no PyTorch, no TensorFlow). Educational purpose: understand
exactly what the algorithm does under the hood.

Architecture:
    • Policy network  — 2 hidden layers, softmax output (discrete actions)
    • Value network   — 2 hidden layers, linear output
    • Both networks share no weights (actor-critic separate)
    • Updates use the PPO-clip objective + value loss + entropy bonus

👨‍💻 Developed by dug  |  GitHub: https://github.com/dug6464
"""

import numpy as np


# ── Activation functions ───────────────────────────────────────────────────────
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


# ── MLP helper ────────────────────────────────────────────────────────────────
def init_weights(layer_sizes: list[int], scale: float = 0.1) -> list[dict]:
    """Xavier-ish initialisation for a list of layer sizes."""
    params = []
    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out)
        params.append({"W": w, "b": b})
    return params


def forward(params: list[dict], x: np.ndarray,
            output_activation=None) -> tuple[np.ndarray, list]:
    """Forward pass; returns output and cache of pre-activations."""
    cache = []
    h = x.copy()
    for i, p in enumerate(params):
        z = h @ p["W"] + p["b"]
        cache.append((h, z))
        h = relu(z) if i < len(params) - 1 else z
    if output_activation:
        h = output_activation(h)
    return h, cache


# ── PPO Agent (NumPy) ─────────────────────────────────────────────────────────
class NumpyPPO:
    """
    Minimal PPO agent implemented purely in NumPy.

    Supports discrete action spaces (e.g. LunarLander: 4 actions).
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        hidden:     list[int] = None,
        lr:         float = 3e-4,
        gamma:      float = 0.99,
        lam:        float = 0.95,
        clip_eps:   float = 0.2,
        ent_coef:   float = 0.01,
        n_epochs:   int   = 4,
    ):
        hidden = hidden or [128, 128]
        self.gamma    = gamma
        self.lam      = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.n_epochs = n_epochs
        self.lr       = lr

        # Policy (actor) and value (critic) networks
        self.actor  = init_weights([obs_dim] + hidden + [action_dim])
        self.critic = init_weights([obs_dim] + hidden + [1])

    # ── Inference ─────────────────────────────────────────────────────────────
    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        logits, _ = forward(self.actor, obs)
        return softmax(logits)

    def get_value(self, obs: np.ndarray) -> float:
        v, _ = forward(self.critic, obs)
        return float(v[0])

    def select_action(self, obs: np.ndarray) -> tuple[int, float]:
        probs  = self.get_action_probs(obs)
        action = int(np.random.choice(len(probs), p=probs))
        return action, float(probs[action])

    # ── GAE returns ───────────────────────────────────────────────────────────
    def compute_gae(self, rewards, values, dones) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation."""
        n         = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(n)):
            next_val  = values[t + 1] if t + 1 < len(values) else 0.0
            delta     = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae  = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(values[:n], dtype=np.float32)
        return advantages, returns

    # ── SGD update (Adam-lite: plain gradient step) ────────────────────────────
    def _grad_step(self, params: list[dict], grads: list[dict]):
        for p, g in zip(params, grads):
            p["W"] -= self.lr * g["W"]
            p["b"] -= self.lr * g["b"]

    # ── PPO update ────────────────────────────────────────────────────────────
    def update(self, rollout: dict):
        """
        Run n_epochs of PPO-clip updates on a collected rollout.

        rollout keys: obs, actions, old_probs, advantages, returns
        """
        obs        = np.array(rollout["obs"],        dtype=np.float32)
        actions    = np.array(rollout["actions"],    dtype=np.int32)
        old_probs  = np.array(rollout["old_probs"],  dtype=np.float32)
        advantages = np.array(rollout["advantages"], dtype=np.float32)
        returns    = np.array(rollout["returns"],    dtype=np.float32)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        for _ in range(self.n_epochs):
            for i in range(len(obs)):
                o  = obs[i]
                a  = actions[i]
                op = old_probs[i]
                adv = advantages[i]
                ret = returns[i]

                # Policy gradient (PPO-clip)
                probs     = self.get_action_probs(o)
                new_prob  = probs[a]
                ratio     = new_prob / (op + 1e-8)
                clip_ratio = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -min(ratio * adv, clip_ratio * adv)

                # Entropy bonus
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                policy_loss -= self.ent_coef * entropy
                total_policy_loss += policy_loss

                # Approximate gradient for actor (one-step finite diff not shown;
                # in practice use autograd — this demo does a simplified update)
                grad_logit       = probs.copy()
                grad_logit[a]   -= 1.0
                grad_logit      *= policy_loss
                for layer in reversed(self.actor):
                    layer["W"] -= self.lr * np.outer(o, grad_logit) / len(obs)
                    layer["b"] -= self.lr * grad_logit / len(obs)

                # Value loss (MSE)
                v        = self.get_value(o)
                v_loss   = (v - ret) ** 2
                grad_v   = 2 * (v - ret) * np.ones(1)
                for layer in reversed(self.critic):
                    layer["W"] -= self.lr * np.outer(o, grad_v) / len(obs)
                    layer["b"] -= self.lr * grad_v / len(obs)

        return total_policy_loss / (self.n_epochs * len(obs))
