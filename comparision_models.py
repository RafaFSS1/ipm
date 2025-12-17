import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

# ==============================================================================
# 1. WRAPPERS DE ANÁLISE (O Espião e o Juiz)
# ==============================================================================

class SoftMetricWrapper(gym.Wrapper):
    """
    MODO SOFT: Deixa o carro andar até ao fim.
    Mede: Percentagem de Relva, ZigZag e Pontuação Pura.
    """
    def __init__(self, env):
        super().__init__(env)
        self.grass_steps = 0
        self.total_steps = 0
        self.total_zigzag = 0.0
        self.last_steering = 0.0

    def reset(self, **kwargs):
        self.grass_steps = 0
        self.total_steps = 0
        self.total_zigzag = 0.0
        self.last_steering = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        
        # Métrica ZigZag (Estabilidade)
        self.total_zigzag += abs(action[0] - self.last_steering)
        self.last_steering = action[0]

        # Métrica Relva (Disciplina)
        roi = obs[60:65, 46:50]
        if np.mean(roi[:, :, 1]) > np.mean(roi[:, :, 0]) + 0.15:
            self.grass_steps += 1
            
        return obs, reward, terminated, truncated, info

    def get_stats(self):
        return {
            "Relva_Pct": (self.grass_steps / self.total_steps * 100) if self.total_steps > 0 else 0,
            "ZigZag_Index": (self.total_zigzag / self.total_steps) if self.total_steps > 0 else 0
        }

class HardSafetyWrapper(gym.Wrapper):
    """
    MODO HARD: Regras de Morte Ativas.
    Mede: Taxa de Sobrevivência.
    """
    def __init__(self, env):
        super().__init__(env)
        self.grass_counter = 0

    def reset(self, **kwargs):
        self.grass_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        steering, gas, brake = action
        custom_reward = reward
        
        # Penalidade Relva
        roi = obs[60:65, 46:50]
        diff = np.mean(roi[:, :, 1]) - np.mean(roi[:, :, 0])
        if diff > 0.15:
            custom_reward -= 0.1 
            penalty = diff * 4.0 
            penalty = min(penalty, 2.0) 
            self.grass_counter += 1
        else:
            self.grass_counter = 0
        
        # Cerca Elétrica (Morte)
        if self.grass_counter > 50:
            terminated = True
            custom_reward -= 10.0
        
        # Penalidade Travagem Excessiva
        custom_reward -= 0.05 * brake

        # Penalidade ZigZag
        steering_diff = abs(steering - self.last_steering)
        custom_reward -= 0.05 * steering_diff
        self.last_steering = steering
            
        return obs, custom_reward, terminated, truncated, info

# ==============================================================================
# 2. MOTOR DE TESTE
# ==============================================================================

def make_soft_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = SoftMetricWrapper(env)
    return env

def make_hard_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = HardSafetyWrapper(env)
    return env

def run_test_battery(model, name, mode="soft", n_episodes=5):
    """
    Corre uma bateria de testes. Aplica Stacks automaticamente se for 'Zoo'.
    """
    print(f"   -> A correr modo {mode.upper()} ({n_episodes} voltas)...")
    
    # 1. Escolha do Ambiente
    if mode == "soft":
        env = DummyVecEnv([make_soft_env])
    else:
        env = DummyVecEnv([make_hard_env])

    # 2. Lógica de STACKS (Otimizada para o teu caso)
    # Apenas o Zoo usa Stacks. Os outros (Original, Custom Normal, SAC) são Raw.
    if "Zoo" in name:
        env = VecFrameStack(env, n_stack=4)
        print("      (FrameStack=4 Aplicado)")
    
    env = VecTransposeImage(env)
    
    results = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]
        
        if mode == "soft":
            stats = env.envs[0].get_stats()
            results.append({
                "Soft_Reward": ep_reward,
                "Relva_Pct": stats["Relva_Pct"],
                "ZigZag": stats["ZigZag_Index"]
            })
        else:
            # Sobreviveu se fez > 800 pontos no modo Hard
            survived = 1 if ep_reward > 800 else 0
            results.append({
                "Hard_Reward": ep_reward,
                "Sobreviveu": survived
            })
            
    env.close()
    
    # Calcular Estatísticas (Média e Desvio Padrão)
    df_temp = pd.DataFrame(results)
    stats_out = df_temp.mean().to_dict()
    
    for key in ["Soft_Reward", "Hard_Reward", "ZigZag"]:
        if key in df_temp.columns:
            stats_out[f"{key}_Std"] = df_temp[key].std()
            
    return stats_out

# ==============================================================================
# 3. MAIN - AQUI ESTÃO OS 4 MODELOS
# ==============================================================================
def main():
    # --- CONFIGURAÇÃO DOS CAMINHOS ---
    # Tens de garantir que estes 4 ficheiros existem!
    models_config = {
        
        # 1. PPO Original (Gym Padrão)
        "PPO_Original_Gym": "models_original/best_model.zip", 
        
        # 2. PPO Custom (Gym com as tuas penalidades, mas arquitetura normal)
        "PPO_Custom_Gym":   "models_custom_normal/best_model.zip", 
        
        # 3. PPO Zoo (Gym Custom + 2 CPU + Stacks)
        "PPO_Zoo_Custom":   "models_zoo/best/best_model.zip", 
        
        # 4. SAC Custom (Gym Custom + SAC)
        "SAC_Custom":       "models_sac/best_model.zip" 
    }

    N_EPISODES = 5 # Aumenta para 10 se quiseres mais precisão
    final_report = []

    print(f"--- BENCHMARK QUARTETO: ORIGINAL vs CUSTOM vs ZOO vs SAC ---")

    for name, path in models_config.items():
        if not os.path.exists(path):
            print(f"\n[ERRO] Ficheiro não encontrado para: {name}")
            print(f"       Caminho: {path}")
            continue
            
        print(f"\n>> Carregando: {name}")
        try:
            # Carregar o Modelo Certo
            if "SAC" in name.upper():
                model = SAC.load(path, device="cpu")
            else:
                model = PPO.load(path, device="cpu")
            
            # Teste Soft (Qualidade)
            soft_stats = run_test_battery(model, name, mode="soft", n_episodes=N_EPISODES)
            # Teste Hard (Sobrevivência)
            hard_stats = run_test_battery(model, name, mode="hard", n_episodes=N_EPISODES)
            
            combined = {"Modelo": name}
            combined.update(soft_stats)
            combined.update(hard_stats)
            final_report.append(combined)
            
        except Exception as e:
            print(f"   [CRASH] Erro ao testar {name}: {e}")

    if not final_report: 
        print("Nenhum modelo testado.")
        return

    # --- GERAR TABELA ---
    df = pd.DataFrame(final_report)
    df["Sobreviveu"] = df["Sobreviveu"] * 100
    df = df.round(2)
    
    cols = ["Modelo", "Soft_Reward", "Soft_Reward_Std", "Hard_Reward", "Relva_Pct", "ZigZag", "Sobreviveu"]
    df = df[[c for c in cols if c in df.columns]]

    print("\n" + "="*100)
    print(" RESULTADOS FINAIS (4 MODELOS) ")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    df.to_csv("benchmark_quartet.csv", index=False)

    # --- GRÁFICOS COMPARATIVOS ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Performance Pura (Média + Erro)
    sns.barplot(data=df, x="Modelo", y="Soft_Reward", ax=axes[0,0], palette="viridis", hue="Modelo", legend=False)
    axes[0,0].errorbar(x=range(len(df)), y=df["Soft_Reward"], yerr=df["Soft_Reward_Std"], fmt='none', c='black', capsize=5)
    axes[0,0].set_title("Pontuação Pura (Ambiente Soft)")
    axes[0,0].axhline(900, color='r', linestyle='--')

    # 2. Segurança (Relva %)
    sns.barplot(data=df, x="Modelo", y="Relva_Pct", ax=axes[0,1], palette="Reds", hue="Modelo", legend=False)
    axes[0,1].set_title("Uso de Relva (%) - Menor é melhor")

    # 3. Estabilidade (ZigZag)
    sns.barplot(data=df, x="Modelo", y="ZigZag", ax=axes[1,0], palette="Blues", hue="Modelo", legend=False)
    axes[1,0].set_title("Índice ZigZag (Estabilidade) - Menor é melhor")

    # 4. Sobrevivência (Hard Mode)
    sns.barplot(data=df, x="Modelo", y="Sobreviveu", ax=axes[1,1], palette="RdYlGn", hue="Modelo", legend=False)
    axes[1,1].set_title("Taxa de Sobrevivência (Hard Mode) %")
    axes[1,1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("benchmark_quartet_charts.png")
    plt.show()

if __name__ == "__main__":
    main()