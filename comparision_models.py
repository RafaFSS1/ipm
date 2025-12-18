import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

# ==============================================================================
# 1. WRAPPER UNIFICADO (O Juiz Duplo)
# ==============================================================================

class DualAnalyticWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Acumuladores
        self.grass_counter = 0
        self.last_steering = 0.0
        
        # Stats do Episódio
        self.episode_original_reward = 0.0
        self.episode_custom_reward = 0.0
        self.episode_grass_steps = 0
        self.episode_total_steps = 0
        self.episode_zigzag = 0.0
        
        # Estado de "Morte Virtual"
        self.simulated_death = False

    def reset(self, **kwargs):
        self.grass_counter = 0
        self.last_steering = 0.0
        
        self.episode_original_reward = 0.0
        self.episode_custom_reward = 0.0
        self.episode_grass_steps = 0
        self.episode_total_steps = 0
        self.episode_zigzag = 0.0
        
        self.simulated_death = False
        
        return self.env.reset(**kwargs)

    def step(self, action):
        # 1. Passo Real (Ambiente Original - O carro nunca morre por regras custom)
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        steering, gas, brake = action
        self.episode_total_steps += 1
        self.episode_original_reward += original_reward

        # --- CÁLCULO DE MÉTRICAS GERAIS ---
        
        # ZigZag
        self.episode_zigzag += abs(steering - self.last_steering)
        self.last_steering = steering

        # Deteção de Relva (Canais normalizados 0-1)
        roi = obs[60:65, 46:50]
        red = np.mean(roi[:, :, 0]).astype(float) / 255.0
        green = np.mean(roi[:, :, 1]).astype(float) / 255.0
        diff = green - red
        
        is_grass = diff > 0.15
        if is_grass:
            self.episode_grass_steps += 1
            self.grass_counter += 1
        else:
            self.grass_counter = 0

        # --- CÁLCULO DA REWARD CUSTOM (SIMULADA) ---
        # Só calculamos se o carro ainda estiver "vivo" nas regras custom
        if not self.simulated_death:
            
            # Começamos com a reward base deste step
            step_custom_reward = original_reward
            
            # 1. Penalidade Relva (Diff * 4)
            if is_grass:
                penalty = diff * 4.0
                penalty = min(penalty, 2.0)
                step_custom_reward -= penalty
                # Nota: Não apliquei o -0.1 fixo extra porque no teu código anterior
                # parecia que querias ou um ou outro. Se quiseres os dois, descomenta:
                # step_custom_reward -= 0.1

            # 2. Penalidade Travão
            step_custom_reward -= 0.05 * brake
            
            # 3. Penalidade ZigZag
            steering_diff = abs(steering - self.last_steering) # Já calculado mas ok
            step_custom_reward -= 0.05 * steering_diff
            
            # 4. Verificar Morte Súbita (> 50 frames)
            if self.grass_counter > 50:
                self.simulated_death = True
                step_custom_reward -= 10.0 # Penalidade final de morte
                # A partir de agora, self.episode_custom_reward deixa de ser atualizado
            
            self.episode_custom_reward += step_custom_reward

        # --- SALVAR NO INFO ---
        if terminated or truncated:
            grass_pct = (self.episode_grass_steps / self.episode_total_steps * 100) if self.episode_total_steps > 0 else 0
            zigzag_idx = (self.episode_zigzag / self.episode_total_steps) if self.episode_total_steps > 0 else 0
            
            info["episode_metrics"] = {
                "Original_Score": self.episode_original_reward,
                "Custom_Score": self.episode_custom_reward, # Congelado na morte
                "Relva_Pct": grass_pct,
                "ZigZag_Index": zigzag_idx,
                "Died_Virtual": 1 if self.simulated_death else 0, # Se morreu virtualmente
                "Death_Step": self.episode_total_steps if self.simulated_death else -1 # Debug
            }

        return obs, original_reward, terminated, truncated, info

# ==============================================================================
# 2. MOTOR DE AVALIAÇÃO
# ==============================================================================

def make_eval_env():
    # Usamos o ambiente original + O nosso Wrapper Duplo
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = DualAnalyticWrapper(env)
    return env

def evaluate_model(model, name, n_episodes=10):
    print(f"\n>> Avaliando: {name} ({n_episodes} episódios)...")
    
    env = DummyVecEnv([make_eval_env])

    # Lógica Stack para o Zoo
    if "Zoo" in name:
        env = VecFrameStack(env, n_stack=4)
        print("   [Info] FrameStack=4 Ativo")
    
    env = VecTransposeImage(env)
    
    results = []
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        
        # Variável para capturar as métricas do wrapper
        metrics = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            
            if done and "episode_metrics" in infos[0]:
                metrics = infos[0]["episode_metrics"]
        
        # Se por acaso falhar (segurança), mete zeros
        if metrics is None: metrics = {}
        
        data_point = {
            "Modelo": name,
            "Reward Original": metrics.get("Original_Score", 0),
            "Reward Custom": metrics.get("Custom_Score", 0),
            "Relva %": metrics.get("Relva_Pct", 0),
            "ZigZag": metrics.get("ZigZag_Index", 0),
            "Morreu (Simulado)": metrics.get("Died_Virtual", 0)
        }
        results.append(data_point)
        
        # Print bonito
        status = "MORREU" if data_point["Morreu (Simulado)"] else "VIVO"
        print(f"   Volta {i+1}: Orig={data_point['Reward Original']:.0f} | Custom={data_point['Reward Custom']:.0f} | {status}")

    env.close()
    return results

# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    # CAMINHOS - AJUSTA AQUI
    models_config = {
        "PPO_Original": "final_models/ppo_original.zip", 
        "PPO_Custom":   "final_models/ppo_custom.zip", 
        "PPO_Zoo":      "final_models/ppo_zoo_custom.zip", 
        "SAC_Custom":   "final_models/sac_custom.zip" 
    }

    N_EPISODES = 10 # Pediste 10 voltas
    all_data = []

    print(f"--- BENCHMARK UNIFICADO ({N_EPISODES} Voltas) ---")

    for name, path in models_config.items():
        if not os.path.exists(path):
            print(f"Saltar {name} (Não encontrado)")
            continue
            
        try:
            if "SAC" in name.upper():
                model = SAC.load(path, device="cpu")
            else:
                model = PPO.load(path, device="cpu")
                
            data = evaluate_model(model, name, n_episodes=N_EPISODES)
            all_data.extend(data)
        except Exception as e:
            print(f"Erro {name}: {e}")

    if not all_data: return

    # --- PROCESSAMENTO DE DADOS ---
    df = pd.DataFrame(all_data)
    
    # Calcular Sobrevivência Real (%)
    # Se Morreu (Simulado) for 0, então sobreviveu.
    # Invertemos a lógica para calcular % de Sobrevivência
    df["Sobreviveu"] = 1 - df["Morreu (Simulado)"]
    
    # Agrupar médias e desvios
    summary = df.groupby("Modelo").agg({
        "Reward Original": ["mean", "std"],
        "Reward Custom": ["mean", "std"],
        "Relva %": "mean",
        "ZigZag": "mean",
        "Sobreviveu": "mean" # Média de 0s e 1s dá a percentagem (ex: 0.8 = 80%)
    }).round(2)
    
    # Ajustar nome das colunas e converter %
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.rename(columns={"Sobreviveu_mean": "Taxa_Sobrevivência"}, inplace=True)
    summary["Taxa_Sobrevivência"] *= 100

    print("\n" + "="*100)
    print(" TABELA FINAL (10 Episódios no Mesmo Ambiente) ")
    print("="*100)
    print(summary.to_string())
    
    df.to_csv("benchmark_unified_raw.csv", index=False)
    summary.to_csv("benchmark_unified_summary.csv")

    # --- GRÁFICOS ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Comparação Direta de Rewards (Original vs Custom)
    # Vamos criar um DF longo para poder usar o 'hue'
    df_melt = df.melt(id_vars="Modelo", value_vars=["Reward Original", "Reward Custom"], var_name="Tipo", value_name="Pontos")
    sns.barplot(data=df_melt, x="Modelo", y="Pontos", hue="Tipo", ax=axes[0,0], palette="muted")
    axes[0,0].set_title("Original vs Custom Reward (Mesma Volta)")
    axes[0,0].axhline(900, color='r', linestyle='--', alpha=0.5)

    # 2. Taxa de Sobrevivência
    # Precisamos de calcular a % para o gráfico
    surv_plot = df.groupby("Modelo")["Sobreviveu"].mean().reset_index()
    surv_plot["Sobreviveu"] *= 100
    sns.barplot(data=surv_plot, x="Modelo", y="Sobreviveu", ax=axes[0,1], palette="RdYlGn")
    axes[0,1].set_title("Taxa de Sobrevivência Virtual (%)")
    axes[0,1].set_ylim(0, 100)

    # 3. Relva %
    sns.barplot(data=df, x="Modelo", y="Relva %", ax=axes[1,0], palette="Reds")
    axes[1,0].set_title("Uso de Relva (%)")

    # 4. ZigZag
    sns.barplot(data=df, x="Modelo", y="ZigZag", ax=axes[1,1], palette="Blues")
    axes[1,1].set_title("Estabilidade (ZigZag)")

    # Limpar gráficos extra
    fig.delaxes(axes[0,2])
    fig.delaxes(axes[1,2])

    plt.tight_layout()
    plt.savefig("benchmark_unified_charts.png")
    plt.show()

if __name__ == "__main__":
    main()