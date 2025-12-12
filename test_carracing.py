import gymnasium as gym

def main():
    env = gym.make("CarRacing-v3", render_mode="human")

    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    main()
