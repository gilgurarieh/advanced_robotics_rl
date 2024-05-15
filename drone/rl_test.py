import torch
from stable_baselines3 import PPO
from DroneEnv import DroneEnv  # Import your environment from DroneEnv.py

# Create an instance of your environment
env = DroneEnv()

# Define a simple policy network
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])])

# Create the PPO agent
ppo_agent = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# Train the agent
ppo_agent.learn(total_timesteps=1000)

# Save the trained agent
ppo_agent.save("ppo_drone_agent")
