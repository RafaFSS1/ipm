import argparse
import csv
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

from custom_carracing_env import CustomCarRacingEnv


def make_env_original(render_mode="human"):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init


def make_env_custom(render_mode="human"):
    def _init():
        env = CustomCarRacingEnv(render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init


def evaluate(model_path, env_type, episodes=3, render_mode="human"):
    print(f"\nEvaluating model: {model_path}")
    print(f"Environment type: {env_type}\n")

    # Escolher o tipo de ambiente
    if env_type == "custom":
        env_fn = make_env_custom(render_mode=render_mode)
    else:
        env_fn = make_env_original(render_mode=render_mode)

    # Wrap igual ao treino
    env = DummyVecEnv([env_fn])
    env = VecTransposeImage(env)

    model = PPO.load(model_path)

    episode_rewards = []  # <- lista para guardar rewards por episódio

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # VecEnv: 4 valores, renomear para NÃO colidir com episode_rewards
            obs, step_rewards, dones, infos = env.step(action)

            ep_reward += float(step_rewards[0])
            done = bool(dones[0])

        print(f"Episode {ep + 1} reward: {ep_reward:.2f}")
        episode_rewards.append(ep_reward)

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"\nAverage reward over {episodes} episodes: {avg_reward:.2f}")

    # Guardar resultados num CSV
    csv_file = "evaluation_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "environment", "episodes", "average_reward"])
        writer.writerow([model_path, env_type, episodes, avg_reward])

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, choices=["original", "custom"], required=True,
                        help="Which environment to evaluate on.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model ZIP file.")
    parser.add_argument("--episodes", type=int, default=3)

    args = parser.parse_args()

    evaluate(args.model, args.env, args.episodes)
