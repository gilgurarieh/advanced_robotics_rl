import numpy as np
import gymnasium
from gymnasium.utils import seeding
from gymnasium import spaces
import time
import sys

sys.path.append('VREP_RemoteAPIs')
import sim

from Drone_model import DroneModel


class DroneEnv(gymnasium.Env):
    def __init__(self, reward_type):
        '''
        :param reward_type: "stabilize", "vertical", "position"
        '''
        super(DroneEnv, self).__init__()
        self.action_scaling = 10
        self.reward_type = reward_type
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        drone_params = 4 + 3 + 6  # orientation quaternion + position + velocities
        target_params = 3  # position only
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(drone_params + target_params,),
                                            dtype=np.float32)

        self.seed()
        self.step_count = 0
        self.max_steps = 100

        # Connect to VREP (CoppeliaSim)
        sim.simxFinish(-1)
        self.client_ID = sim.simxStart('127.0.0.1', 19997, True, False, 5000, 5)
        if self.client_ID > -1:
            print('Connected to remote API server.')
            sim.simxSynchronous(self.client_ID, True)
            sim.simxStartSimulation(self.client_ID, sim.simx_opmode_oneshot)
            sim.simxSynchronousTrigger(self.client_ID)

            self.drone_sim_model = DroneModel()
            self.drone_sim_model.initializeSimModel(self.client_ID)
            self.current_state_dict = {}
            self.state = self.get_current_state()
        else:
            print('Failed connecting to remote API server! Try it again ...')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = self.action_scaling * action

        # Apply the action to the CoppeliaSim model
        action_list = action.tolist()
        self.drone_sim_model.setPropellerThrust(action_list)

        # Trigger the simulation step
        sim.simxSynchronousTrigger(self.client_ID)
        sim.simxGetPingTime(self.client_ID)

        new_state = self.get_current_state()
        terminated, termination_reason = self.check_terminal_state()
        reward = self.calculate_reward(terminated, termination_reason)

        if self.step_count >= self.max_steps:
            truncated = True
        else:
            truncated = False
            self.step_count += 1

        info = {}

        # Print step information
        print(f"Step count: {self.step_count}")
        print(f"Action: {action_list}")
        # Print pose and velocities
        target_pos = self.current_state_dict["target_pos"]
        drone_pos = self.current_state_dict["drone_pos"]
        drone_ori = self.current_state_dict["drone_ori"]
        drone_v_lin = self.current_state_dict["drone_v_lin"]
        drone_v_ang = self.current_state_dict["drone_v_ang"]

        print(f"Target Position: {target_pos}")
        print(f"Position: {drone_pos}")
        print(f"Orientation: {drone_ori}")
        print(f"Velocities - Linear: {drone_v_lin}")
        print(f"Velocities - Angular: {drone_v_ang}")
        print(f"Step reward: {reward}")
        print("\n")

        return new_state, reward, terminated, truncated, info

    def get_current_state(self):
        clientID = self.drone_sim_model.client_ID
        heli = self.drone_sim_model.heli_handle
        target = self.drone_sim_model.target_handle
        opmode = sim.simx_opmode_blocking

        _, drone_quat = sim.simxGetObjectQuaternion(clientID, heli, -1, opmode)
        _, drone_pos = sim.simxGetObjectPosition(clientID, heli, -1, opmode)
        _, drone_ori = sim.simxGetObjectOrientation(clientID, heli, -1, opmode)
        _, v_lin, v_ang = sim.simxGetObjectVelocity(clientID, heli, opmode)
        # _, target_pos = sim.simxGetObjectPosition(clientID, target, -1, opmode)
        target_pos = [-0.550000011920929, 0.6000000238418579, 0.5]

        self.current_state_dict = {
            "drone_pos": drone_pos,
            "drone_ori": drone_ori,
            "drone_v_lin": v_lin,
            "drone_v_ang": v_ang,
            "target_pos": target_pos
        }

        state_list = drone_quat + drone_pos + v_lin + v_ang + target_pos
        state = np.array(state_list)
        return state

    def calculate_reward(self, terminated, termination_reason):
        reward_scaling = 10
        state_dict = self.current_state_dict
        alpha = state_dict["drone_ori"][0]
        beta = state_dict["drone_ori"][1]
        height = state_dict["drone_pos"][2]
        target_height = state_dict["target_pos"][2]

        # Calculate linear and angular velocities
        linear_velocity = np.linalg.norm(state_dict["drone_v_lin"])
        angular_velocity = np.linalg.norm(state_dict["drone_v_ang"])

        # Calculate reward components
        r_stabilize = -abs(alpha)/np.pi -abs(beta)/np.pi -linear_velocity/10 -angular_velocity/10
        r_vertical = -abs(height-target_height) / 0.3  # normalize by maximum dist
        drone_pos = state_dict["drone_pos"]
        target_pos = state_dict["target_pos"]
        r_position = -np.sqrt((drone_pos[0]-target_pos[0])**2 + (drone_pos[1]-target_pos[1])**2)

        if self.reward_type == "stabilize":
            reward = r_stabilize
            if terminated and termination_reason in ["angle_too_big", "reached_ground"]:
                reward -= 100

        elif self.reward_type == "vertical":
            reward = 0.3*r_stabilize + 0.7*r_vertical
            if terminated and termination_reason in ["reached_ground", "vertical"]:
                reward -= 100

        elif self.reward_type == "position":
            reward = 0.1*r_stabilize + 0.2*r_vertical + 0.7*r_position
            if terminated and termination_reason in ["reached_ground", "vertical", "horizontal"]:
                reward -= 100
        else:
            reward = 0

        return reward

    def check_terminal_state(self):
        '''
        :return:
        terminated: bool
        termination_reason: "horizontal",
                            "vertical",
                            "reached_ground",
                            "angle_too_big"
        '''
        state_dict = self.current_state_dict
        pos = np.array(state_dict["drone_pos"])
        alpha = np.array(state_dict["drone_ori"][0])
        beta = np.array(state_dict["drone_ori"][1])
        target_pos = np.array(state_dict["target_pos"])

        horizontal_distance = np.sqrt((pos[0] - target_pos[0]) ** 2 + (pos[1] - target_pos[1]) ** 2)
        vertical_distance = np.abs(target_pos[2]-pos[2])
        if horizontal_distance > 0.5:
            print("Terminated: too far from target")
            return True, "horizontal"
        if vertical_distance > 0.4:
            print("Terminated: too far from target")
            return True, "vertical"

        if pos[2] <= 0.1:
            print("Terminated: reached ground")
            return True, "reached_ground"

        threshold = 45  # degrees
        if abs(alpha) > np.radians(threshold) or abs(beta) > np.radians(threshold):
            print("Terminated: angle too big")
            return True, "angle_too_big"

        return False, None

    def reset(self, seed=None):
        # Stop the simulation
        sim.simxStopSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_blocking)
        sim.simxGetPingTime(self.drone_sim_model.client_ID)
        time.sleep(0.01)

        # Re-enable synchronization
        sim.simxSynchronous(self.client_ID, True)
        sim.simxStartSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_oneshot)
        sim.simxSynchronousTrigger(self.client_ID)

        self.step_count = 0

        # Set initial position and orientation
        initial_pos = (-0.55, 0.6, 0.51)
        initial_ori = (0.0, 0.0, 0.0)
        opmode = sim.simx_opmode_blocking
        sim.simxSetObjectPosition(self.drone_sim_model.client_ID, self.drone_sim_model.heli_handle, -1, initial_pos,
                                  opmode)
        sim.simxSetObjectOrientation(self.drone_sim_model.client_ID, self.drone_sim_model.heli_handle, -1, initial_ori,
                                     opmode)

        # Wait for a moment to ensure the state is set
        time.sleep(1)  # Sleep for 1 second to let the simulation stabilize

        # Fetch the initial state from CoppeliaSim again
        state = self.get_current_state()
        self.state = state
        return state, {}

    def render(self):
        return None

    def close(self):
        sim.simxStopSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_blocking)
        sim.simxFinish(-1)
        print('Close the environment')
        return None


if __name__ == "__main__":
    env = DroneEnv()
    env.reset()

    for _ in range(500):
        action = env.action_space.sample()
        env.step(action)

    env.close()
