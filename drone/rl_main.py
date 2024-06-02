import torch
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from DroneEnv import DroneEnv  # Import your environment from DroneEnv.py

# Create a function to plot the rewards
def plot_rewards(rewards, title="Training Rewards"):
    plt.plot(rewards)
    plt.xlabel('Evaluations')
    plt.ylabel('Mean Reward')
    plt.title(title)
    plt.savefig(title+".png")

# Create an instance of your environment
env = DroneEnv(reward_type="mixed")

# Define a simple policy network
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[128, 128], vf=[128, 128])])

# Callback for evaluation
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=True, render=False)

# Train the agent
total_timesteps = 10000
env.reward_type = "mixed"

# # agent = A2C("MlpPolicy", env, gamma=0.95, policy_kwargs=policy_kwargs, verbose=1)
# a2c_agent = A2C.load("a2c_action_test_200k", env=env)
# policy_weights = a2c_agent.policy.state_dict()
# ppo_model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1)
# ppo_model.policy.load_state_dict(policy_weights)

rewards_list = []
k_timesteps = 350

for i in range(0, 11):
    # if not first - load the previous model
    model_name = "ppo_action_test_"+str(k_timesteps)+"k"
    ppo_model = PPO.load(model_name, env=env)
    ppo_model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    k_timesteps += 10
    save_name = "ppo_action_test_"+str(k_timesteps)+"k"
    ppo_model.save(save_name)
    # Extract evaluation results and plot rewards
    mean_rewards = [result[0] for result in eval_callback.evaluations_results]
    rewards_list.append(mean_rewards)
    plot_rewards(mean_rewards, title=save_name)

print(f"rewards: {rewards_list}")


# # Test the model
# env = DroneEnv(reward_type="mixed")
# agent = A2C.load("a2c_action_test_200k", env=env)
# obs, _ = env.reset()
# terminated = False
# while not terminated:
#     action, _states = agent.predict(obs, deterministic=True)
#     obs, rewards, terminated, truncated, info = env.step(action)
#     env.render()
# env.close()

