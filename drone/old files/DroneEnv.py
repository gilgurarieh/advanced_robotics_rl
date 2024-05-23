import numpy as np
import gymnasium
from gymnasium.utils import seeding
from gymnasium import spaces
import time
# from dqrobotics import DQ
import sys

sys.path.append('VREP_RemoteAPIs')
import sim

from Drone_model import DroneModel
# import utils


class DroneEnv(gymnasium.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        # define some internal values
        self.action_scaling = 50

        # define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        drone_params = 4 + 3 + 6  # orientation quaternion + position + velocities
        target_params = 3 # position only
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(drone_params+target_params,), dtype=np.float32)

        # initialize random seed
        self.seed()

        # initialize step count for truncation
        self.step_count = 0
        self.max_steps = 20

        # Connect to VREP (CoppeliaSim)
        sim.simxFinish(-1)  # close all opened connections
        self.client_ID = sim.simxStart('127.0.0.1', 19997, True, False, 5000, 5)  # Connect to CoppeliaSim
        if self.client_ID > -1:  # connected
            print('Connect to remote API server.')
            sim.simxSynchronous(self.client_ID, True)
            sim.simxStartSimulation(self.client_ID, sim.simx_opmode_oneshot)
            sim.simxSynchronousTrigger(self.client_ID)

            # Initialize drone model
            self.drone_sim_model = DroneModel()
            self.drone_sim_model.initializeSimModel(self.client_ID)

            # Fetch the initial state from CoppeliaSim
            self.current_state_dict = {}
            self.state = self.get_current_state()
        else:
            print('Failed connecting to remote API server! Try it again ...')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        :param action: (ActType) – an action provided by the agent to update the environment state.
        :return: observation (obs_type), reward (float), terminated (bool), truncated (bool), info (dict)
        '''
        # Convert the action to a NumPy array (if it isn't already)
        action = np.array(action, dtype=np.float32)


        # Scale the action using the action scaling factor
        action = self.action_scaling * action
        # print(f"action: {action}")

        # Apply the action to the CoppeliaSim model
        action_list = action.tolist()
        self.drone_sim_model.setPropellerThrust(action_list)

        # Trigger the simulation step
        sim.simxSynchronousTrigger(self.client_ID)
        # Ensure the step is complete
        sim.simxGetPingTime(self.client_ID)

        # Get new state:
        new_state = self.get_current_state()

        # Check if state is terminated
        terminated = self.check_terminal_state()

        # Get reward
        reward = self.calculate_reward()  # Normal reward calculation based on the latest state

        # Determine if the episode is truncated
        if self.step_count >= self.max_steps:
            truncated = True
        else:
            truncated = False
            self.step_count = self.step_count + 1

        # Compile all information into a dictionary
        info = {}

        if terminated:
            reward = -50

        # update state - maybe we don't need it!
        if not terminated and not truncated:
            self.state = new_state

        print(f"step count: {self.step_count}")
        print(f"Action: {action_list}")
        print(f"step reward: {reward}")
        return new_state, reward, terminated, truncated, info

    def get_current_state(self):
        """
        Gets the current state from the CoppeliaSim simulation
        :return: state in numbers format
        """
        # define variables for convenience
        clientID = self.drone_sim_model.client_ID
        heli = self.drone_sim_model.heli_handle
        target = self.drone_sim_model.target_handle
        opmode = sim.simx_opmode_blocking

        # Get drone quaternion
        _, drone_quat = sim.simxGetObjectQuaternion(clientID, heli, -1, operationMode=opmode)
        # NOTE: quaternion is list in the format: x,y,z,w and not reversed!
        # print(f"drone_quat = {drone_quat}")

        # Get drone position
        _,  drone_pos = sim.simxGetObjectPosition(clientID, heli, -1, operationMode=opmode)
        # print(f"drone position: {drone_pos}")

        # Get drone orientation - for termination test
        _, drone_ori = sim.simxGetObjectOrientation(clientID, heli, -1, operationMode=opmode)

        # get drone velocities
        _, v_lin, v_ang = sim.simxGetObjectVelocity(clientID, heli, operationMode=opmode)
        # print(f"drone linear velocity: {v_lin}, drone angular velocity: {v_ang}")

        # get Target position
        _, target_pos = sim.simxGetObjectPosition(clientID, target, -1, operationMode=opmode)
        # print(f"target position: {target_pos}")

        # save state dict
        self.current_state_dict = {
            "drone_pos" : drone_pos,
            "drone_ori": drone_ori,
            "drone_v_lin": v_lin,
            "drone_v_ang": v_ang,
            "target_pos": target_pos
        }

        # Construct state
        state_list = drone_quat + drone_pos + v_lin + v_ang + target_pos #4+3+3+3+3 = 16
        print("state_list: ",state_list)
        # print(f"state list: {state_list}")
        state = np.array(state_list)
        print("state: ", state)
        return state

    def calculate_reward(self):
        reward_scaling = 10
        reward_normalizing = 2.0
        state_dict = self.current_state_dict
        # pos = np.array(state_dict["drone_pos"])
        alpha = state_dict["drone_ori"][0]
        beta = state_dict["drone_ori"][1]
        height = state_dict["drone_pos"][2]
        # target_pos = np.array(state_dict["target_pos"])
        target_height = state_dict["target_pos"][2]

        # calculate distance between drone and target

        # calculate reward
        reward = reward_scaling*(max([0,1-abs(height-target_height)])) - abs(alpha) - abs(beta)
        # print(f"step reward: {reward}")
        return reward

    def check_terminal_state(self):
        '''
        checks if the current state is terminal or not
        :return: bool: whether this state is terminal
        '''
        is_terminal = False

        # unpack current state
        state_dict = self.current_state_dict
        pos = np.array(state_dict["drone_pos"])
        alpha = np.array(state_dict["drone_ori"][0])
        beta = np.array(state_dict["drone_ori"][1])
        v_lin = np.array(state_dict["drone_v_lin"])
        v_ang = np.array(state_dict["drone_v_ang"])
        target_pos = np.array(state_dict["target_pos"])

        # check if position is too far
        horizontal_distance = np.sqrt((pos[0]-target_pos[0])**2 + (pos[1]-target_pos[1])**2)
        if horizontal_distance > 5:
            is_terminal = True
            print("Terminated: too far from target")

        # check if colliding with ground
        if pos[2] <= 0.1:
            is_terminal = True
            print("Terminated: reached ground")

        # check if orientation is too extreme
        threshold = 60  # degrees, for example
        if abs(alpha) > np.radians(threshold) or abs(beta) > np.radians(threshold):
            is_terminal = True
            print("Terminated: angle too big")

        #TODO: check velocities

        return is_terminal

    def reset(self, seed=None):
        # Stop the simulation
        sim.simxStopSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_blocking)
        sim.simxGetPingTime(self.drone_sim_model.client_ID)
        time.sleep(0.01)  # ensure the CoppeliaSim is stopped

        # Restart the simulation
        sim.simxStartSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_oneshot)
        sim.simxSynchronousTrigger(self.drone_sim_model.client_ID)

        # restart step count
        self.step_count = 0

        # Send the drone to its initial position:
        initial_pos = (-0.55, 0.6, 0.5)
        initial_ori = (0.0, 0.0, 0.0)
        opmode = sim.simx_opmode_buffer
        sim.simxSetObjectPosition(self.drone_sim_model.client_ID, self.drone_sim_model.heli_handle,-1,  initial_pos, opmode)
        sim.simxSetObjectOrientation(self.drone_sim_model.client_ID, self.drone_sim_model.heli_handle, -1, initial_ori, opmode)


        # Fetch the initial state from CoppeliaSim again
        state = self.get_current_state()
        self.state = state
        # print(f"State fetched after reset: {self.current_state_dict}")
        return state, {}

    def render(self):
        return None

    def close(self):
        sim.simxStopSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_blocking)  # stop the simulation
        sim.simxFinish(-1)  # Close the connection
        print('Close the environment')
        return None


    def angular_deviation(self,q1, q2):
        # Assuming q1 and q2 are normalized quaternions
        dot_product = np.dot(q1, q2)
        # Clamp dot product to ensure acos is valid
        dot_product = np.clip(dot_product, -1.0, 1.0)
        # Calculate the angle between quaternions
        angle = 2 * np.arccos(dot_product)
        return np.degrees(angle)  # Convert to degrees

