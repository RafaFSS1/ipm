import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


def make_env_original():
    """
    Environment original: CarRacing-v3 SEM reward shaping.
    """
    def _init():
        env = gym.make("CarRacing-v3", render_mode=None)
        env = Monitor(env)
        return env
    return _init


def main():
    os.makedirs("./checkpoints_original/", exist_ok=True)
    os.makedirs("./models_original/best/", exist_ok=True)
    os.makedirs("./logs_original/", exist_ok=True)
    os.makedirs("./tensorboard_original/", exist_ok=True)

    # ---------- ENV DE TREINO ----------
    train_env = DummyVecEnv([make_env_original()])
    train_env = VecTransposeImage(train_env)

    # ---------- ENV DE AVALIAÇÃO ----------
    eval_env = DummyVecEnv([make_env_original()])
    eval_env = VecTransposeImage(eval_env)

    # ---------- CALLBACKS ----------
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints_original/",
        name_prefix="ppo_carracing_original",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_original/best/",
        log_path="./logs_original/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    # ---------- MODELO PPO ----------
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        tensorboard_log="./tensorboard_original/"
    )

    TOTAL_TIMESTEPS = 500_000

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save("ppo_carracing_original_final")


if __name__ == "__main__":
    main()
