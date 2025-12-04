import gymnasium as gym
from enum import Enum
import numpy as np

# Gymnasium<1.1.0 doesn't expose AutoresetMode, which ale_py>=0.11 expects.
try:
    from gymnasium.vector import AutoresetMode  # type: ignore
except ImportError:
    class AutoresetMode(str, Enum):
        NEXT_STEP = "NextStep"
        SAME_STEP = "SameStep"

    import gymnasium.vector as gym_vector

    gym_vector.AutoresetMode = AutoresetMode

import ale_py

gym.register_envs(ale_py)



if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = gym.wrappers.HumanRendering(env)
    # env = gym.wrappers.RecordVideo(env, video_folder="video")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    print("Initial observation:", obs)

    ram_state = env.unwrapped.ale.getRAM()  # numpy array length 128

    print(f"RAM state shape: {ram_state.shape}, dtype: {ram_state.dtype}") 

    while True:
        action = env.action_space.sample()
        #print("Step %d: took action %d" % (total_steps, action))
        obs, reward, done, _, _ = env.step(action)
        ram_state = env.unwrapped.ale.getRAM()  # numpy array length 128
        #print("Step %d: received reward %.2f" % (total_steps, reward))

        paddle_x = ram_state[72]   #   index for paddle 
        ball_x   = ram_state[99]   #   ball horizontal position
        ball_y   = ram_state[101]  #   ball vertical position
        score    = ram_state[105]  #  score  
        features = np.array([paddle_x, ball_x, ball_y, score], dtype=float)
        print(f"Step {total_steps}, Action {action}, Reward {reward}: Features: Paddle X: {paddle_x}, Ball X: {ball_x}, Ball Y: {ball_y}, Score: {score}")


        total_reward += reward
        total_steps += 1
        if done:
            break

    print(f"Episode done in {total_steps} steps, total reward {total_reward:.2f}")
    env.close()
