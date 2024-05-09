# imports
from CartPoleEnv import CartPoleEnv
import random

# initialize the environment
env = CartPoleEnv(action_type="continuous")

env.reset()

# control using actions

done = False
while not done:
    action = random.uniform(-1.0, 1.0)
    print(f"action = {action}")
    _, _, done, _, _ = env.step(action)
env.close()


