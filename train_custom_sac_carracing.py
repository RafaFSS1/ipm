import os

import gymnasium as gym
from stable_baselines3 import SAC
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
        # 1. Cria o Gym normal (CONTÍNUO para SAC)
        env = gym.make("CarRacing-v3", render_mode=None, continuous=True)
        # 2. Aplica o Wrapper Manualmente
        env = CustomCarRacingEnv(env)
        # 3. Monitor
        env = Monitor(env)
        return env
    return _init


def main():
    os.makedirs("./checkpoints_sac/", exist_ok=True)
    os.makedirs("./models_sac/best/", exist_ok=True)
    os.makedirs("./logs_sac/", exist_ok=True)
    os.makedirs("./tensorboard_sac/", exist_ok=True)

    # ---------- ENV DE TREINO ----------
    train_env = DummyVecEnv([make_env()])
    train_env = VecTransposeImage(train_env)

    # ---------- ENV DE AVALIAÇÃO ----------
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecTransposeImage(eval_env)

    # ---------- CALLBACKS ----------
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints_sac/",
        name_prefix="sac_carracing",
        save_replay_buffer=True,       # SAC => faz sentido guardar replay buffer
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_sac/best/",
        log_path="./logs_sac/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    # ---------- MODELO SAC ----------
    model = SAC(
        policy="CnnPolicy",
        env=train_env,
        verbose=1,

        # Off-policy / replay
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        train_freq=1,          # atualiza a cada step
        gradient_steps=1,

        # SAC core
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,

        # Entropia
        ent_coef="auto",
        target_entropy="auto",

        # Exploração contínua (muito útil no CarRacing)
        use_sde=True,
        sde_sample_freq=4,

        tensorboard_log="./tensorboard_sac/"
    )

    # ---------- TREINO ----------
    TOTAL_TIMESTEPS = 500_000  # para SAC costuma compensar subir (1M-4M), mas fica igual ao teu

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )

    # ---------- GUARDAR MODELO FINAL ----------
    model.save("sac_carracing_custom_final")


if __name__ == "__main__":
    main()
