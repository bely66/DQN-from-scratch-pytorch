import os

import gym
from gym.wrappers import record_video

gym.logger.set_level(gym.logger.DEBUG)

env = gym.make("Pong-v0", render_mode="rgb_array")
num_actions = env.action_space.sample()
print("Number of actions:", num_actions)
env = gym.wrappers.RecordVideo(env, 'video', video_length=500)

env.reset()
for t in range(1):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rew, done, _ = env.step(action)
        print("Reward for Frame:", rew)



env.close()

