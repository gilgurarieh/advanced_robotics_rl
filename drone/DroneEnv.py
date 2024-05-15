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
        self.action_scaling = 1

        # define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        drone_params = 4 + 3 + 6  # orientation quaternion + position + velocities
        target_params = 3 # position only
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(drone_params+target_params,), dtype=np.float32)

        # initialize random seed
        self.seed()

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
            print(f"Initial state fetched at environment creation: {self.current_state_dict}")
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

        # Apply the action to the CoppeliaSim model
        action_list = action.tolist()
        self.drone_sim_model.setPropellerThrust(action_list)

        # Get new state:
        new_state = self.get_current_state()
        print(f"State fetched during step: \n {self.current_state_dict}\n")

        # Check if state is terminated
        terminated = self.check_terminal_state()

        # Get reward
        reward = self.calculate_reward()  # Normal reward calculation based on the latest state
        print(f"reward: {reward}")

        # Determine if the episode is truncated
        truncated = False  # Assuming normal conditions for truncation

        # Compile all information into a dictionary
        info = {}

        # update state - maybe we don't need it!
        if not terminated and not truncated:
            self.state = new_state

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
        # print(f"state list: {state_list}")
        state = np.array(state_list)

        return state

    def calculate_reward(self):
        reward_scaling = 1
        state_dict = self.current_state_dict
        pos = np.array(state_dict["drone_pos"])
        target_pos = np.array(state_dict["target_pos"])

        # calculate distance between drone and target
        dist = np.linalg.norm(pos-target_pos)

        # calculate reward
        reward = -reward_scaling * dist

        return reward

    # def calculate_reward(self, new_state):
    #     drone_dq = new_state['drone']
    #     target_dq = new_state['target']
    #
    #     # Calculate current distance and orientation deviation
    #     current_distance, orientation_deviation = utils.calculate_distance_and_orientation(drone_dq, target_dq)
    #
    #     # Calculate progress
    #     if self.previous_distance is not None:
    #         distance_improvement = self.previous_distance - current_distance
    #     else:
    #         distance_improvement = 0
    #
    #     # Update previous distance
    #     self.previous_distance = current_distance
    #
    #     # Define reward components
    #     position_reward = distance_improvement * 10  # Scale to make meaningful
    #     stability_reward = 1  # Constant reward for staying in flight
    #     orientation_penalty = -orientation_deviation  # Penalty for large orientation deviations
    #
    #     # Combine rewards
    #     reward = position_reward + stability_reward + orientation_penalty
    #
    #     # Check for excessive tilt and apply penalties
    #     tilt_threshold = 30  # degrees
    #     if orientation_deviation > tilt_threshold:
    #         reward -= 100  # Large penalty for losing balance
    #
    #     return reward

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
        dist_to_target = np.linalg.norm(pos-target_pos)
        if dist_to_target > 8:
            is_terminal = True

        # check if orientation is too extreme
        threshold = 30  # degrees, for example
        if alpha > threshold or beta > threshold:
            is_terminal = True

        #TODO: check velocities

        return is_terminal

    def reset(self):
        # Stop the simulation
        sim.simxStopSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_blocking)
        sim.simxGetPingTime(self.drone_sim_model.client_ID)
        time.sleep(0.01)  # ensure the CoppeliaSim is stopped

        # Restart the simulation
        sim.simxStartSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_oneshot)
        sim.simxSynchronousTrigger(self.drone_sim_model.client_ID)

        # Fetch the initial state from CoppeliaSim again
        state = self.get_current_state()
        self.state = state
        print(f"State fetched after reset: {self.current_state_dict}")
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

