import gymnasium as gym

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip):
        """Devolve apenas um a cada 'skip' frames"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repete a ação, soma a reward e salta frames"""
        total_reward = 0.0
        for i in range(self._skip):
            # Acumula a reward e repete a mesma ação
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info