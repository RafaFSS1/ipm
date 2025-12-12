from custom_carracing_env import CustomCarRacingEnv


def main():
    # Para debug visual usamos render_mode="human"
    env = CustomCarRacingEnv(render_mode="human")

    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Ainda estamos a usar ações aleatórias (só para testar integração)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()


if __name__ == "__main__":
    main()
