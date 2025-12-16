import argparse
import csv
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from gymnasium.wrappers import ResizeObservation
from utils import FrameSkip

# Importar a tua lógica partilhada
from custom_carracing_env import CustomCarRacingEnv

# =========================================================
# 1. ORIGINAL (O Puro - 96x96)
# =========================================================
def make_original_env(render_mode="human"):
    def _init():
        # Apenas o ambiente nativo. Sem wrappers de relva, sem resize.
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        return env
    return _init

# =========================================================
# 2. PPO (O Teu Customizado Simples - 96x96)
# =========================================================
def make_ppo_env(render_mode="human"):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        # Aplica APENAS a tua lógica de recompensas
        env = CustomCarRacingEnv(env) 
        return env
    return _init

# =========================================================
# 3. ZOO (O Otimizado - 64x64 + Stack)
# =========================================================
def make_zoo_env(render_mode="human"):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        
        # 1. Penalidade na imagem original (96x96)
        env = CustomCarRacingEnv(env) 
        
        # 2. FrameSkip (igual ao treino Zoo)
        try:
            env = FrameSkip(env, skip=1)
        except:
            pass
            
        # 3. Resize para 84x84 (igual ao treino Zoo)
        env = ResizeObservation(env, shape=(84, 84))
        return env
    return _init

# =========================================================
# FUNÇÃO DE AVALIAÇÃO
# =========================================================
def evaluate(model_path, env_type, episodes=5, render_mode="human"):
    print(f"\n>>> A CARREGAR MODELO: {model_path}")
    print(f">>> AMBIENTE SELECIONADO: {env_type.upper()}\n")

    # --- LÓGICA DE SELEÇÃO DE AMBIENTE ---
    if env_type == "original":
        # Original: 96x96. Precisa de Transpose (HWC -> CHW) para o PPO.
        env = DummyVecEnv([make_original_env(render_mode)])
        env = VecTransposeImage(env)
        
    elif env_type == "ppo":
        # PPO Custom: 96x96. Igual ao original, mas com as tuas penalidades ativas.
        env = DummyVecEnv([make_ppo_env(render_mode)])
        env = VecTransposeImage(env)
    
    elif env_type == "zoo":
        # Zoo: 64x64. Usa FrameStack em vez de Transpose.
        env = DummyVecEnv([make_zoo_env(render_mode)])
        env = VecFrameStack(env, n_stack=4)

    else:
        print("Erro: Tipo de ambiente desconhecido.")
        return

    # --- CARREGAR E TESTAR ---
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Não encontrei o ficheiro '{model_path}'")
        return

    total_rewards = []
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            # deterministic=True força o modelo a usar o que sabe melhor
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]
            
        print(f"Episódio {ep + 1}: {ep_reward:.1f} pontos")
        total_rewards.append(ep_reward)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"\n---------------------------------------")
    print(f"MÉDIA FINAL ({env_type}): {avg_reward:.1f} (+/- {std_reward:.1f})")
    print(f"---------------------------------------\n")
    
    env.close()

    # --- GUARDAR NO CSV (COMO NO ORIGINAL) ---
    csv_file = "evaluation_results.csv"
    file_exists = os.path.isfile(csv_file)

    try:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            # Se o ficheiro não existe, escreve o cabeçalho primeiro
            if not file_exists:
                writer.writerow(["model", "environment", "episodes", "average_reward", "std_dev"])
            
            # Escreve a linha de dados
            writer.writerow([model_path, env_type, episodes, round(avg_reward, 2), round(std_reward, 2)])
            print(f"Resultados guardados em '{csv_file}'.")
    except Exception as e:
        print(f"Aviso: Não foi possível guardar no CSV. Erro: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True, help="Caminho do .zip")
    
    # Agora aceita os 3 tipos:
    parser.add_argument("--env", type=str, required=True, 
                        choices=["original", "ppo", "zoo"], 
                        help="Escolhe: original, ppo (o teu simples) ou zoo (otimizado)")
    
    parser.add_argument("--episodes", type=int, default=3, help="Número de voltas")
    parser.add_argument("--no-render", action="store_true", help="Não abrir janela (rápido)")

    args = parser.parse_args()
    
    mode = "rgb_array" if args.no_render else "human"
    
    evaluate(args.model, args.env, args.episodes, mode)