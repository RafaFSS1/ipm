import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomCarRacingEnv(gym.Wrapper):
    """
    Wrapper para modificar o reward do CarRacing-v3.

    Objetivo:
    - Incentivar o carro a manter-se na pista
    - Desencorajar travagens excessivas
    - Suavizar o steering (menos zig-zags)
    - Incentivar a eficiência (não “passear” sem propósito)
    """

    def __init__(self, env):
        # Cria o ambiente base
        super().__init__(env)
        # Mantemos o espaço de observação e ação tal como no original
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Para medir zig-zag precisamos de lembrar o steering anterior
        self.last_steering = 0.0
        self.grass_counter = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_steering = 0.0
        self.grass_counter = 0
        return obs, info

    def step(self, action):
        # Ação padrão no CarRacing: [steering, gas, brake]
        obs, reward, terminated, truncated, info = self.env.step(action)

        steering, gas, brake = action

        # Começamos com o reward original
        custom_reward = reward

        # ---------- 1) Penalização por sair da pista (relva) ----------
        # 1. Recorte (ROI) - Pequena área debaixo/frente do carro
        # Ajuste ligeiro para tentar evitar o corpo vermelho do carro
        roi = obs[60:65, 46:50]

        # 2. Separar Canais e Normalizar
        # Axis 0 e 1 são altura e largura, Axis 2 são as cores (0=R, 1=G, 2=B)
        red_channel_mean   = np.mean(roi[:, :, 0]) / 255.0
        green_channel_mean = np.mean(roi[:, :, 1]) / 255.0
        
        # 3. A Lógica da Diferença ("Green Dominance")
        # diff será:
        # Perto de 0.0 se for estrada (cinzento)
        # Positivo (> 0.15) se for relva
        # Negativo se for o próprio carro (vermelho)
        diff = green_channel_mean - red_channel_mean

        # 4. Debug (Muito importante agora no início)
        # print(f"Green: {green_channel_mean:.3f} | Red: {red_channel_mean:.3f} | Diff: {diff:.3f}")

        # 5. Penalização
        # Se o verde for significativamente maior que o vermelho (ex: 5% maior)
        if diff > 0.15:
            
            # Com multiplicador 4.0:
            # - Se diff for 0.25 (mínimo relva) -> Punição = -1.0
            # - Se diff for 0.50 (máximo relva) -> Punição = -2.0
            #penalty = diff * 4.0  
            self.grass_counter += 1 # Está na relva, aumenta o contador
            # Mantemos o clip em 2.0 por segurança, caso apareça um pico estranho
            #penalty = min(penalty, 2.0)
            penalty=0.1
            
            custom_reward -= penalty
        else:
            # Voltou para a estrada! Zera o contador.
            self.grass_counter = 0
        
        if self.grass_counter > 50: #50 steps na relva
            terminated = True  # GAME OVER. Mata o episódio.
            custom_reward -= 10.0 # "Death Penalty" (Castigo final por desistir)
            
        # ---------- 2) Penalização por travagem forte ----------
        # Queremos desincentivar brake constante a 1.0
        custom_reward -= 0.05 * brake

        # ---------- 3) Penalização por zig-zag (mudança brusca de steering) ----------
        steering_diff = abs(steering - self.last_steering)
        custom_reward -= 0.05 * steering_diff
        self.last_steering = steering

        # ---------- 4) Pequena penalização por passo ----------
        # Isto força o agente a ser eficiente (não ficar a “cozinhar” parado)
        # custom_reward -= 0.01
        # ja existe uma penalização por passo no ambiente original

        return obs, custom_reward, terminated, truncated, info
    

