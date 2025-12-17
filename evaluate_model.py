import argparse
import csv
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC  # <--- Adicionado SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from gymnasium.wrappers import ResizeObservation
# from utils import FrameSkip # Mantive comentado como no teu original

# Importar a tua lógica partilhada
from custom_carracing_env import CustomCarRacingEnv

# =========================================================
# 1. ORIGINAL (O Puro - 96x96)
# =========================================================
def make_original_env(render_mode="human"):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        return env
    return _init

# =========================================================
# 2. CUSTOM (O Teu Simples - 96x96 + Wrapper)
# =========================================================
def make_custom_env(render_mode="human"):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        # Aplica APENAS a tua lógica de recompensas
        env = CustomCarRacingEnv(env) 
        return env
    return _init

# =========================================================
# 3. ZOO (O Otimizado - Stack + Wrapper)
# =========================================================
def make_zoo_env(render_mode="human"):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = CustomCarRacingEnv(env) 
        # Mantém as configurações que tinhas (sem resize/skip por agora)
        return env
    return _init

# =========================================================
# FUNÇÃO DE AVALIAÇÃO
# =========================================================
def evaluate(model_path, env_type, algo="ppo", episodes=5, render_mode="human"):
    print(f"\n>>> A CARREGAR MODELO ({algo.upper()}): {model_path}")
    print(f">>> AMBIENTE SELECIONADO: {env_type.upper()}\n")

    # --- LÓGICA DE SELEÇÃO DE AMBIENTE ---
    if env_type == "original":
        # Original: 96x96. Precisa de Transpose.
        env = DummyVecEnv([make_original_env(render_mode)])
        env = VecTransposeImage(env)
        
    elif env_type == "custom":
        # Custom/SAC: 96x96. Igual ao original + Wrapper.
        env = DummyVecEnv([make_custom_env(render_mode)])
        env = VecTransposeImage(env)
    
    elif env_type == "zoo":
        # Zoo: Pode ter FrameStack.
        env = DummyVecEnv([make_zoo_env(render_mode)])
        # IMPORTANTE: Se o Zoo foi treinado com Stack, tem de ter Stack aqui.
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env) # Adicionei Transpose aqui caso o Stack não o faça sozinho corretamente no pipeline

    else:
        print("Erro: Tipo de ambiente desconhecido.")
        return

    # --- CARREGAR MODELO (PPO ou SAC) ---
    try:
        if algo.lower() == "sac":
            model = SAC.load(model_path)
        else:
            model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Não encontrei o ficheiro '{model_path}'")
        return
    except Exception as e:
        print(f"ERRO ao carregar modelo: {e}")
        return

    total_rewards = []
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]
            
        print(f"Episódio {ep + 1}: {ep_reward:.1f} pontos")
        total_rewards.append(ep_reward)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"\n---------------------------------------")
    print(f"RESULTADO ({algo.upper()} @ {env_type}): {avg_reward:.1f} (+/- {std_reward:.1f})")
    print(f"---------------------------------------\n")
    
    env.close()

    # --- GUARDAR NO CSV ---
    csv_file = "evaluation_results.csv"
    file_exists = os.path.isfile(csv_file)

    try:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            # Cabeçalho atualizado
            if not file_exists:
                writer.writerow(["algorithm", "model", "environment", "episodes", "average_reward", "std_dev"])
            
            # Escreve a linha de dados
            writer.writerow([algo.upper(), model_path, env_type, episodes, round(avg_reward, 2), round(std_reward, 2)])
            print(f"Resultados guardados em '{csv_file}'.")
    except Exception as e:
        print(f"Aviso: Não foi possível guardar no CSV. Erro: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True, help="Caminho do .zip")
    
    # Argumento NOVO para escolher o algoritmo
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], 
                        help="Algoritmo a usar: 'ppo' (default) ou 'sac'")

    parser.add_argument("--env", type=str, required=True, 
                        choices=["original", "custom", "zoo"], 
                        help="Escolhe: original, custom ou zoo")
    
    parser.add_argument("--episodes", type=int, default=3, help="Número de voltas")
    parser.add_argument("--no-render", action="store_true", help="Não abrir janela (rápido)")

    args = parser.parse_args()
    
    mode = "rgb_array" if args.no_render else "human"
    
    evaluate(args.model, args.env, args.algo, args.episodes, mode)