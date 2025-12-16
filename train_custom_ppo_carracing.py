import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from custom_carracing_env import CustomCarRacingEnv


def make_env():
    """
    Função de fábrica para criar uma instância do environment customizado.
    Necessária para o DummyVecEnv.
    """
    def _init():
        # 1. Cria o Gym normal
        env = gym.make("CarRacing-v3", render_mode=None)
        # 2. Aplica o Wrapper Manualmente
        env = CustomCarRacingEnv(env)
        # 3. Monitor
        env = Monitor(env)
        return env
    return _init


def main():
    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs("./models/best/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./tensorboard/", exist_ok=True)

    # ---------- ENV DE TREINO ----------
    # DummyVecEnv espera uma lista de funções que criam envs
    train_env = DummyVecEnv([make_env()])
    # PPO com CnnPolicy espera observações no formato (C, H, W),
    # por isso fazemos a transposição dos canais
    train_env = VecTransposeImage(train_env)

    # ---------- ENV DE AVALIAÇÃO ----------
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecTransposeImage(eval_env)

    # ---------- CALLBACKS ----------
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,              # guarda modelo a cada 50k steps
        save_path="./checkpoints/",
        name_prefix="ppo_carracing",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
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
        tensorboard_log="./tensorboard/"
    )

    # ---------- TREINO ----------
    TOTAL_TIMESTEPS = 500_000  # ↑ aumentámos para um valor mais sério

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )

    # ---------- GUARDAR MODELO FINAL ----------
    model.save("ppo_carracing_custom_final")


if __name__ == "__main__":
    main()
