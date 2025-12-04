import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.HumanRendering(env)
    # env = gym.wrappers.RecordVideo(env, video_folder="video")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    print("Initial observation:", obs)

    while True:
        action = env.action_space.sample()
        print("Step %d: took action %d" % (total_steps, action))
        obs, reward, done, _, _ = env.step(action)
        print("Step %d: received reward %.2f" % (total_steps, reward))

        total_reward += reward
        total_steps += 1
        if done:
            break

    print(f"Episode done in {total_steps} steps, total reward {total_reward:.2f}")
    env.close()
