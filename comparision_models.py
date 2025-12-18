import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

# ==============================================================================
# 1. WRAPPERS (Lógica ESTRITA do teu ficheiro + Fix dos Zeros)
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
        red_channel_mean   = np.mean(roi[:, :, 0]) / 255.0
        green_channel_mean = np.mean(roi[:, :, 1]) / 255.0
        
        diff = green_channel_mean - red_channel_mean
        if diff > 0.15:
            self.grass_steps += 1
        
        # --- FIX: INJETAR NO INFO PARA NÃO DAR ZERO ---
        # Guardamos as stats aqui porque no próximo frame o reset() limpa tudo
        if terminated or truncated:
            grass_pct = (self.grass_steps / self.total_steps * 100) if self.total_steps > 0 else 0
            zigzag_idx = (self.total_zigzag / self.total_steps) if self.total_steps > 0 else 0
            
            info["episode_metrics"] = {
                "Relva_Pct": grass_pct,
                "ZigZag_Index": zigzag_idx
            }

        return obs, reward, terminated, truncated, info

class HardSafetyWrapper(gym.Wrapper):
    """
    MODO HARD: Regras de Morte Ativas (Copiado do teu ficheiro).
    """
    def __init__(self, env):
        super().__init__(env)
        self.grass_counter = 0
        self.last_steering = 0.0

    def reset(self, **kwargs):
        self.grass_counter = 0
        self.last_steering = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        steering, gas, brake = action
        custom_reward = reward
        
        # Penalidade Relva (A tua lógica complexa)
        roi = obs[60:65, 46:50]
        red_channel_mean   = np.mean(roi[:, :, 0]) / 255.0
        green_channel_mean = np.mean(roi[:, :, 1]) / 255.0
        
        diff = green_channel_mean - red_channel_mean
        if diff > 0.15:
            penalty = diff * 4.0 
            penalty = min(penalty, 2.0) 
            self.grass_counter += 1
            custom_reward -= penalty
        else:
            self.grass_counter = 0
        
        # Cerca Elétrica (Morte aos 50 frames)
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
# 2. MOTOR DE TESTE (Com correção de leitura de Info)
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
    print(f"   -> A correr modo {mode.upper()} ({n_episodes} voltas)...")
    
    if mode == "soft":
        env = DummyVecEnv([make_soft_env])
    else:
        env = DummyVecEnv([make_hard_env])

    if "Zoo" in name:
        env = VecFrameStack(env, n_stack=4)
        print("      (FrameStack=4 Aplicado)")
    
    env = VecTransposeImage(env)
    
    results = []
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        captured_metrics = None # Variável para guardar o info
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            ep_reward += reward[0]
            
            # Capturar métricas do wrapper antes que desapareçam
            if done and "episode_metrics" in infos[0]:
                captured_metrics = infos[0]["episode_metrics"]
        
        if mode == "soft":
            # Usar métricas capturadas (ou 0 se algo falhar)
            r_pct = captured_metrics["Relva_Pct"] if captured_metrics else 0
            z_idx = captured_metrics["ZigZag_Index"] if captured_metrics else 0
            
            results.append({
                "Soft_Reward": ep_reward,
                "Relva_Pct": r_pct,
                "ZigZag": z_idx
            })
            print(f"      Volta {i+1}: Reward={ep_reward:.0f} | Relva={r_pct:.1f}%")
        else:
            survived = 1 if ep_reward > 600 else 0
            results.append({
                "Hard_Reward": ep_reward,
                "Sobreviveu": survived
            })
            
    env.close()
    
    df_temp = pd.DataFrame(results)
    stats_out = df_temp.mean().to_dict()
    
    # Calcular Desvio Padrão
    for key in ["Soft_Reward", "Hard_Reward", "ZigZag"]:
        if key in df_temp.columns:
            stats_out[f"{key}_Std"] = df_temp[key].std()
            
    return stats_out

# ==============================================================================
# 3. MAIN (Com novos gráficos)
# ==============================================================================
def main():
    models_config = {
        "PPO_Original_Gym": "models_original/best/best_model.zip", 
        "PPO_Custom_Gym":   "models/best/best_after_change_rewards.zip", 
        "PPO_Zoo_Custom":   "models_zoo/best/best_model_beforerewardchange.zip", 
        "SAC_Custom":       "models_sac/best/best_model.zip" 
    }

    N_EPISODES = 5
    final_report = []

    print(f"--- BENCHMARK QUARTETO (Strict User Rewards) ---")

    for name, path in models_config.items():
        if not os.path.exists(path):
            print(f"\n[ERRO] Ficheiro não encontrado: {path}")
            continue
            
        print(f"\n>> Carregando: {name}")
        try:
            if "SAC" in name.upper():
                model = SAC.load(path, device="cpu")
            else:
                model = PPO.load(path, device="cpu")
            
            soft_stats = run_test_battery(model, name, mode="soft", n_episodes=N_EPISODES)
            hard_stats = run_test_battery(model, name, mode="hard", n_episodes=N_EPISODES)
            
            combined = {"Modelo": name}
            combined.update(soft_stats)
            combined.update(hard_stats)
            final_report.append(combined)
            
        except Exception as e:
            print(f"   [CRASH] Erro ao testar {name}: {e}")

    if not final_report: return

    df = pd.DataFrame(final_report)
    df["Sobreviveu"] = df["Sobreviveu"] * 100
    df = df.round(2)
    
    cols = ["Modelo", "Soft_Reward", "Soft_Reward_Std", "Hard_Reward", "Hard_Reward_Std", "Relva_Pct", "ZigZag", "Sobreviveu"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    print("\n" + "="*100)
    print(" RESULTADOS FINAIS ")
    print("="*100)
    print(df.to_string(index=False))
    
    df.to_csv("benchmark_quartet_strict.csv", index=False)
    
    # --- GRÁFICOS (Layout 2x3 para incluir Hard Reward) ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # 1. Soft Reward
    sns.barplot(data=df, x="Modelo", y="Soft_Reward", ax=axes[0,0], hue="Modelo", legend=False, palette="viridis")
    axes[0,0].errorbar(x=range(len(df)), y=df["Soft_Reward"], yerr=df.get("Soft_Reward_Std", 0), fmt='none', c='black', capsize=5)
    axes[0,0].set_title("Soft Reward (Pontos Originais)")
    axes[0,0].axhline(900, color='r', linestyle='--')

    # 2. Hard Reward (ADICIONADO)
    sns.barplot(data=df, x="Modelo", y="Hard_Reward", ax=axes[0,1], hue="Modelo", legend=False, palette="magma")
    axes[0,1].errorbar(x=range(len(df)), y=df["Hard_Reward"], yerr=df.get("Hard_Reward_Std", 0), fmt='none', c='black', capsize=5)
    axes[0,1].set_title("Hard Reward (As tuas Penalidades)")

    # 3. Sobrevivência
    sns.barplot(data=df, x="Modelo", y="Sobreviveu", ax=axes[0,2], hue="Modelo", legend=False, palette="RdYlGn")
    axes[0,2].set_title("Sobrevivência (%)")
    axes[0,2].set_ylim(0, 100)

    # 4. Relva
    sns.barplot(data=df, x="Modelo", y="Relva_Pct", ax=axes[1,0], hue="Modelo", legend=False, palette="Reds")
    axes[1,0].set_title("Relva %")

    # 5. ZigZag
    sns.barplot(data=df, x="Modelo", y="ZigZag", ax=axes[1,1], hue="Modelo", legend=False, palette="Blues")
    axes[1,1].set_title("ZigZag Index")

    # Apagar plot vazio
    fig.delaxes(axes[1,2])
    
    plt.tight_layout()
    plt.savefig("benchmark_quartet_charts.png")
    plt.show()

if __name__ == "__main__":
    main()