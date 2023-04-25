import gym
from stable_baselines3 import DQN

env = gym.make("Pong-v0", render_mode="rgb_array")
video_path = "pong_dqn.mp4"
env = gym.wrappers.RecordVideo(env, 'video', video_length=500)

model = DQN.load("pong_dqn_model.zip")


obs = env.reset()
terminated = False
while not terminated:
    action, _ = model.predict(obs)
    obs, reward, terminated, info = env.step(action)
    env.render()

env.close()