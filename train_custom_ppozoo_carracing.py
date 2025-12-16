import os
import gymnasium as gym
import torch as th
import torch.nn as nn
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from gymnasium.wrappers import ResizeObservation
from utils import FrameSkip

# IMPORTANTE: O teu ficheiro com a lógica partilhada
from custom_carracing_env import CustomCarRacingEnv

# --- 1. FUNÇÃO DE SCHEDULE (Decaimento Linear) ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Learning Rate começa em 'initial_value' e desce até 0 no fim do treino.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# --- 2. CONSTRUTOR DO AMBIENTE ---
def make_zoo_env(rank: int, seed: int = 0):
    def _init():
        # A. Cria o ambiente base
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        
        # B. APLICA A TUA LÓGICA DE RELVA PRIMEIRO!
        env = CustomCarRacingEnv(env)
        
        # C. FrameSkip (Otimização)
        try:
            env = FrameSkip(env, skip=1)
        except:
            pass 

        # D. Resize (Otimização) - 84x84 RGB
        env = ResizeObservation(env, shape=(84, 84))

        # E. Monitor
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    # --- CONFIGURAÇÃO DE DIRETÓRIOS ---
    checkpoint_dir = "./checkpoints_zoo/"
    best_model_dir = "./models_zoo/best/"
    log_dir = "./logs_zoo/"
    tb_log_dir = "./tensorboard_zoo/"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # --- 3. AMBIENTE DE TREINO (PARALELO) ---
    num_cpu = 2 
    print(f"--- A iniciar {num_cpu} ambientes (Configuração Rápida 500k) ---")
    
    vec_env = SubprocVecEnv([make_zoo_env(i) for i in range(num_cpu)])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    # Clip reward a 10.0 ajuda a estabilizar treinos curtos
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # --- 4. AMBIENTE DE AVALIAÇÃO ---
    eval_env = DummyVecEnv([make_zoo_env(999)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_reward=10.0)
    eval_env.training = False 
    eval_env.norm_reward = False
    eval_env = VecTransposeImage(eval_env)

    # --- 5. CALLBACKS (AJUSTADOS PARA 500K) ---
    
    # Se queres avaliar a cada 25k steps REAIS:
    eval_freq_real = 10_000 // num_cpu  
    
    # Se queres guardar backup a cada 50k steps REAIS:
    save_freq_real = 50_000 // num_cpu
    
    # Avaliar a cada 25k steps (Total de 20 avaliações durante o treino)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=eval_freq_real,        
        n_eval_episodes=5,       
        deterministic=True,      
        render=False
    )

    # Guardar backup a cada 50k steps (Total de 10 backups)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_real,       
        save_path=checkpoint_dir,
        name_prefix="ppo_zoo_backup"
    )

    # --- 6. MODELO PPO (AJUSTADO PARA VELOCIDADE) ---
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
        
        # AQUI ESTÁ A MUDANÇA PRINCIPAL:
        # Começamos com 3e-4 (mais rápido) em vez de 1e-4
        #learning_rate=linear_schedule(3e-4),
        learning_rate=3e-4,  

        n_steps=2048,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.01,
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
    TOTAL_TIMESTEPS = 500_000 
    print(f"--- A treinar por {TOTAL_TIMESTEPS} steps... ---")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTreino interrompido! A guardar modelo de segurança...")

    # --- 8. GUARDAR FINAL ---
    model.save("ppo_carracing_zoo_500k")
    vec_env.save("vec_normalize_zoo_500k.pkl") 
    print("Treino curto concluído.")

if __name__ == "__main__":
    main()