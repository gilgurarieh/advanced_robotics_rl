
from DroneEnv import DroneEnv
import time

drone_env = DroneEnv()


initial_state, _ = drone_env.reset()
start_time = time.time()
# print(f"initial state: {initial_state}")

# try to step the environment
action = [1, 1, 1, 1]
# print(f"chosen action: {action}")

new_state, reward, _, _, _ = drone_env.step(action)
# print(f"new state: {new_state}, reward: {reward}")

print(f"elapsed time: {time.time() - start_time}")

# env = DroneEnv()
# new_target_position = [0.5, 0.5, 0.2]  # New XYZ coordinates
# env.update_target_position(new_target_position)

drone_env.close()