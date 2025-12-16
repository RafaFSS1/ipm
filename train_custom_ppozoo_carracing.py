import os
import gymnasium as gym
import torch as th
import torch.nn as nn
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from gymnasium.wrappers import ResizeObservation, FrameSkip

# IMPORTANTE: O teu ficheiro com a lógica partilhada
from custom_carracing_env import CustomCarRacingEnv

# --- 1. FUNÇÃO DE SCHEDULE (Decaimento Linear) ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Função que faz a learning rate descer linearmente de 'initial_value' até 0.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# --- 2. CONSTRUTOR DO AMBIENTE ---
def make_zoo_env(rank: int, seed: int = 0):
    def _init():
        # A. Cria o ambiente base
        env = gym.make("CarRacing-v2", render_mode="rgb_array")
        
        # B. APLICA A TUA LÓGICA DE RELVA PRIMEIRO!
        env = CustomCarRacingEnv(env)
        
        # C. FrameSkip (Otimização)
        try:
            env = FrameSkip(env, skip=2)
        except:
            pass 
            
        # D. Resize (Otimização) - 64x64 RGB
        env = ResizeObservation(env, shape=(64, 64))

        # E. Monitor
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    # --- CONFIGURAÇÃO DE DIRETÓRIOS ---
    # Usamos sufixo "_zoo" para separar do outro treino
    checkpoint_dir = "./checkpoints_zoo/"
    best_model_dir = "./models_zoo/best/"
    log_dir = "./logs_zoo/"
    tb_log_dir = "./tensorboard_zoo/"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # --- 3. AMBIENTE DE TREINO (PARALELO) ---
    num_cpu = 8 
    print(f"--- A iniciar {num_cpu} ambientes de treino (Zoo Config) ---")
    
    # SubprocVecEnv para treino rápido
    vec_env = SubprocVecEnv([make_zoo_env(i) for i in range(num_cpu)])
    vec_env = VecFrameStack(vec_env, n_stack=2)
    # Normalizamos rewards no treino para o PPO aprender melhor
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # --- 4. AMBIENTE DE AVALIAÇÃO (EVAL) ---
    print("--- A configurar ambiente de avaliação... ---")
    # Usamos DummyVecEnv (apenas 1 env) para avaliação, para não gastar muito CPU
    eval_env = DummyVecEnv([make_zoo_env(999)]) # Seed diferente
    eval_env = VecFrameStack(eval_env, n_stack=2)
    
    # NOTA IMPORTANTE: Não usamos VecNormalize no Eval Env!
    # Porquê? Porque queremos que o 'best_model' seja decidido com base na 
    # pontuação REAL (0-1000) e não na normalizada. Assim sabes se ele fez 900 pontos.

    # --- 5. CALLBACKS (CHECKPOINTS + BEST MODEL) ---
    
    # Callback 1: Guardar o melhor modelo com base na performance real
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=50_000,        # Testar a cada 50k steps
        n_eval_episodes=5,       # Fazer média de 5 corridas
        deterministic=True,      # Usar modo determinístico (sem ruído)
        render=False
    )

    # Callback 2: Guardar checkpoints periódicos (segurança)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,       # Guardar backup a cada 100k steps
        save_path=checkpoint_dir,
        name_prefix="ppo_zoo_backup"
    )

    # --- 6. MODELO PPO (HIPERPARÂMETROS TUNADOS) ---
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=linear_schedule(1e-4),
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.0, 
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=nn.GELU,
            net_arch=dict(pi=[256], vf=[256])
        ),
        tensorboard_log=tb_log_dir
    )

    # --- 7. TREINAR ---
    print("--- A iniciar treino Zoo (com Saves)... ---")
    TOTAL_TIMESTEPS = 2_000_000 

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            # Passamos a lista com os dois callbacks
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTreino interrompido! A guardar modelo de segurança...")

    # --- 8. GUARDAR FINAL ---
    model.save("ppo_carracing_zoo_final")
    vec_env.save("vec_normalize_zoo.pkl") 
    print("Treino concluído. Ficheiros guardados.")

if __name__ == "__main__":
    main()