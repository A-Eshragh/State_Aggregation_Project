"""
Train an SAC expert on Hopper-v5 that reaches >3500 reward.
Save the model for use as BC teacher in the DFL pipeline.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps", type=int, default=300_000)
    p.add_argument("--save_path", type=str, default="sac_hopper")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make("Hopper-v5")
    eval_env = gym.make("Hopper-v5")

    # Hyperparameters from rl-baselines3-zoo Hopper SAC config (proven to reach 3500+)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        seed=args.seed,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"./{args.save_path}_best",
        log_path="./sac_hopper_logs",
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    print(f"Training SAC for {args.total_steps} steps...")
    model.learn(total_timesteps=args.total_steps, callback=eval_cb,
                progress_bar=False, log_interval=100)
    model.save(args.save_path)

    # Final eval
    rewards = []
    for ep in range(10):
        obs, _ = eval_env.reset(seed=ep)
        total = 0
        for t in range(1000):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(a)
            total += r
            if term or trunc:
                break
        rewards.append(total)
    print(f"Final eval: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
    print(f"Saved to: {args.save_path}.zip")


if __name__ == "__main__":
    main()
