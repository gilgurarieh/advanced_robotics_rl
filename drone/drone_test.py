
from DroneEnv import DroneEnv
import time

drone_env = DroneEnv(reward_type="vertical")


initial_state, _ = drone_env.reset()
start_time = time.time()
# print(f"initial state: {initial_state}")

# try to step the environment
terminated=False

while not terminated:
    action = [1, 1, 1, 1]
    new_state, reward, terminated, _, _ = drone_env.step(action)

print(f"elapsed time: {time.time() - start_time}")

# env = DroneEnv()
# new_target_position = [0.5, 0.5, 0.2]  # New XYZ coordinates
# env.update_target_position(new_target_position)

drone_env.close()