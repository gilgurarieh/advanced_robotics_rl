from DroneEnv import DroneEnv

drone_env = DroneEnv()

initial_state, _ = drone_env.reset()
print(f"initial state: {initial_state}")

# try to step the environment
action = [1, 1, 1, 1]
print(f"chosen action: {action}")

new_state, reward, _, _, _ = drone_env.step(action)
print(f"new state: {new_state}, reward: {reward}")
