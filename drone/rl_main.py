import torch
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from DroneEnv import DroneEnv  # Import your environment from DroneEnv.py

# Create a function to plot the rewards
def plot_rewards(rewards, title="Training Rewards"):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title(title)
    plt.show()

# Create an instance of your environment
env = DroneEnv(reward_type="stabilize")

# Define a simple policy network
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])])

# Create the A2C agent
agent = A2C("MlpPolicy", env, gamma=0.95, policy_kwargs=policy_kwargs, verbose=1)

# Callback for evaluation
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=True, render=False)

# Train the agent and plot rewards
total_timesteps = 10000
agent.learn(total_timesteps=total_timesteps, callback=eval_callback)
mean_rewards, _ = zip(*eval_callback.evaluations_results)
plot_rewards(mean_rewards, title="Stabilize Phase Rewards")

# Save the trained agent
agent.save("a2c_stabilize")

# STAGE 2 - vertical
env.reward_type = "vertical"
agent = A2C.load("a2c_stabilize", env=env)
agent.learn(total_timesteps=total_timesteps, callback=eval_callback)
mean_rewards, _ = zip(*eval_callback.evaluations_results)
plot_rewards(mean_rewards, title="Vertical Phase Rewards")
agent.save("a2c_vertical")

# STAGE 3 - position
env.reward_type = "position"
agent = A2C.load("a2c_vertical", env=env)
agent.learn(total_timesteps=total_timesteps, callback=eval_callback)
mean_rewards, _ = zip(*eval_callback.evaluations_results)
plot_rewards(mean_rewards, title="Position Phase Rewards")
agent.save("a2c_position")

# Test the model
obs = env.reset()
reward_list = []
for _ in range(1000):
    action, _states = agent.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    reward_list.append(rewards)
    env.render()

print(f"MEAN REWARD IN TEST: {sum(reward_list)/len(reward_list)}")
