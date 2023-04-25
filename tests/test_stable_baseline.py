import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create Pong environment
env = gym.make('PongNoFrameskip-v4')

# Define and train the DQN agent
model = DQN('MlpPolicy', env, learning_starts=1000,verbose=1, buffer_size=1000)
model.learn(total_timesteps=20000)

# Evaluate the trained agent
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

print("Model Mean Reward:", mean_reward)
# Save the trained model
model.save("pong_dqn_model")