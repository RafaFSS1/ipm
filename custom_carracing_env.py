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

    def __init__(self, render_mode=None):
        # Cria o ambiente base
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        super().__init__(env)

        # Mantemos o espaço de observação e ação tal como no original
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Para medir zig-zag precisamos de lembrar o steering anterior
        self.last_steering = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_steering = 0.0
        return obs, info

    def step(self, action):
        # Ação padrão no CarRacing: [steering, gas, brake]
        obs, reward, terminated, truncated, info = self.env.step(action)

        steering, gas, brake = action

        # Começamos com o reward original
        custom_reward = reward

        # ---------- 1) Penalização por sair da pista (relva) ----------
        # Heurística simples: muita cor verde -> provavelmente relva
        green_channel = obs[:, :, 1]  # canal G da imagem RGB
        green_ratio = np.mean(green_channel) / 255.0

        # Ajusta este limiar em experiências futuras
        if green_ratio > 0.35:
            custom_reward -= 0.1  # penalização extra por estar na relva

        # ---------- 2) Penalização por travagem forte ----------
        # Queremos desincentivar brake constante a 1.0
        custom_reward -= 0.05 * brake

        # ---------- 3) Penalização por zig-zag (mudança brusca de steering) ----------
        steering_diff = abs(steering - self.last_steering)
        custom_reward -= 0.02 * steering_diff
        self.last_steering = steering

        # ---------- 4) Pequena penalização por passo ----------
        # Isto força o agente a ser eficiente (não ficar a “cozinhar” parado)
        custom_reward -= 0.01

        return obs, custom_reward, terminated, truncated, info
