import os
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from custom_carracing_env import CustomCarRacingEnv


def make_env():
    def _init():
        env = gym.make("CarRacing-v3", render_mode=None, continuous=True)
        env = CustomCarRacingEnv(env)
        env = Monitor(env)
        return env
    return _init


def main():
    os.makedirs("./checkpoints_sac/", exist_ok=True)
    os.makedirs("./models_sac/best/", exist_ok=True)
    os.makedirs("./logs_sac/", exist_ok=True)
    os.makedirs("./tensorboard_sac/", exist_ok=True)

    # ---------- ENV ----------
    train_env = DummyVecEnv([make_env()])
    train_env = VecTransposeImage(train_env)

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecTransposeImage(eval_env)

    # ---------- CALLBACKS ----------
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints_sac/",
        name_prefix="sac_carracing",
        save_replay_buffer=False,   # ⚠️ grande speed-up
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_sac/best/",
        log_path="./logs_sac/",
        eval_freq=25_000,           # menos overhead
        deterministic=True,
        render=False
    )

    # ---------- MODELO SAC (OTIMIZADO) ----------
    model = SAC(
        policy="CnnPolicy",
        env=train_env,
        verbose=1,

        # Off-policy
        buffer_size=300_000,
        learning_starts=5_000,      # começa a aprender mais cedo
        batch_size=64,              # ↓↓↓ enorme speed-up
        train_freq=4,               # 4× menos updates
        gradient_steps=1,

        # SAC core
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,

        # Entropia
        ent_coef="auto",
        target_entropy="auto",

        # Exploração
        use_sde=False,              # ⚠️ desliga SDE (CPU killer)

        tensorboard_log="./tensorboard_sac/"
    )

    # ---------- TREINO ----------
    TOTAL_TIMESTEPS = 500_000

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    model.save("sac_carracing_custom_final")


if __name__ == "__main__":
    main()
