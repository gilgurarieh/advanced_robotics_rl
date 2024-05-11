import numpy as np
import gymnasium
from gymnasium.utils import seeding
from gymnasium import spaces
import time
from dqrobotics import DQ
import sys

sys.path.append('VREP_RemoteAPIs')
import sim

from Drone_model import DroneModel
import utils


class DroneEnv(gymnasium.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        # define some internal values
        self.action_scaling = 1

        # define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        drone_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        target_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'drone': drone_observation_space,
            'target': target_observation_space
        })

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
            self.state = self.get_current_state()
            print("Initial state fetched at environment creation:")
            self.print_current_state(self.state)
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
        print("State fetched during step:")
        self.print_current_state(new_state)

        # Calculate the positional distance and angular deviation using the utility function
        positional_distance, rotation_angle = utils.calculate_distance_and_orientation(new_state['drone'],
                                                                                 new_state['target'])

        # Convert rotation angle from radians to degrees
        rotation_angle_degrees = np.degrees(rotation_angle)

        # Check if the rotation exceeds the threshold
        threshold = 30  # degrees, for example
        if rotation_angle_degrees > threshold:
            terminated = True
            reward = -100  # Penalize for losing balance
        else:
            terminated = False
            reward = self.calculate_reward(new_state)  # Normal reward calculation based on the latest state

        # Determine if the episode is truncated
        truncated = False  # Assuming normal conditions for truncation

        # Compile all information into a dictionary
        info = {}
        return new_state, reward, terminated, truncated, info

    def get_current_state(self):
        """
        Gets the current state from the CoppeliaSim simulation
        :return: state in dictionary form
        """
        # Get dual quaternion for drone and target
        drone_dual_quat = utils.get_dual_quaternion(self.client_ID, self.drone_sim_model.heli_handle)
        target_dual_quat = utils.get_dual_quaternion(self.client_ID, self.drone_sim_model.target_handle)

        # Calculate the difference in dual quaternion form
        dq_difference = utils.dq_position_difference(drone_dual_quat, target_dual_quat)

        # Construct state dictionary
        state = {
            'drone': drone_dual_quat,
            'target': target_dual_quat,
            'difference': dq_difference
        }
        return state

    # def calculate_reward(self, new_state):
    #     # TODO - complete this function!
    #     reward = 0.0
    #     return reward

    def calculate_reward(self, new_state):
        drone_dq = new_state['drone']
        target_dq = new_state['target']

        # Calculate current distance and orientation deviation
        current_distance, orientation_deviation = utils.calculate_distance_and_orientation(drone_dq, target_dq)

        # Calculate progress
        if self.previous_distance is not None:
            distance_improvement = self.previous_distance - current_distance
        else:
            distance_improvement = 0

        # Update previous distance
        self.previous_distance = current_distance

        # Define reward components
        position_reward = distance_improvement * 10  # Scale to make meaningful
        stability_reward = 1  # Constant reward for staying in flight
        orientation_penalty = -orientation_deviation  # Penalty for large orientation deviations

        # Combine rewards
        reward = position_reward + stability_reward + orientation_penalty

        # Check for excessive tilt and apply penalties
        tilt_threshold = 30  # degrees
        if orientation_deviation > tilt_threshold:
            reward -= 100  # Large penalty for losing balance

        return reward


    def reset(self):
        # Stop the simulation
        sim.simxStopSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_blocking)
        sim.simxGetPingTime(self.drone_sim_model.client_ID)
        time.sleep(0.01)  # ensure the CoppeliaSim is stopped

        # Restart the simulation
        sim.simxStartSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_oneshot)
        sim.simxSynchronousTrigger(self.drone_sim_model.client_ID)

        # Fetch the initial state from CoppeliaSim again
        self.state = self.get_current_state()
        print("State fetched after reset:")
        self.print_current_state(self.state)
        return self.state, {}

    def render(self):
        return None

    def close(self):
        sim.simxStopSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_blocking)  # stop the simulation
        sim.simxFinish(-1)  # Close the connection
        print('Close the environment')
        return None

    def print_current_state(self, state):
        print("Drone Pose (DQ):", state['drone'])
        print("Target Pose (DQ):", state['target'])
        print("Difference (DQ):", state['difference'])

    def angular_deviation(self,q1, q2):
        # Assuming q1 and q2 are normalized quaternions
        dot_product = np.dot(q1, q2)
        # Clamp dot product to ensure acos is valid
        dot_product = np.clip(dot_product, -1.0, 1.0)
        # Calculate the angle between quaternions
        angle = 2 * np.arccos(dot_product)
        return np.degrees(angle)  # Convert to degrees

