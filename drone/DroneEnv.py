import numpy as np
import gymnasium
from gymnasium.utils import seeding
from gymnasium import spaces, logger
import time

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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape = (4,), dtype=np.float32)

        drone_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        target_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'drone': drone_observation_space,
            'target': target_observation_space
        })

        # initialize random seed
        self.seed()

        # Define initial state
        self.initial_drone_position = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.initial_target_position = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.state = {
            'drone': self.initial_drone_position,
            'target': self.initial_target_position
        }

        # Connect to VREP (CoppeliaSim)
        sim.simxFinish(-1)  # close all opened connections
        while True:
            client_ID = sim.simxStart('127.0.0.1', 19997, True, False, 5000, 5)  # Connect to CoppeliaSim
            if client_ID > -1:  # connected
                print('Connect to remote API server.')
                break
            else:
                print('Failed connecting to remote API server! Try it again ...')

        # Open synchronous mode
        sim.simxSynchronous(client_ID, True)
        sim.simxStartSimulation(client_ID, sim.simx_opmode_oneshot)
        sim.simxSynchronousTrigger(client_ID)

        # initialize drone model
        self.drone_sim_model = DroneModel()
        self.drone_sim_model.initializeSimModel(client_ID)
        sim.simxSynchronousTrigger(client_ID)

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

        # scale the action using action scaling factor
        action = self.action_scaling * action

        # apply the action to the CoppeliaSim model
        action_list = action.tolist()
        self.drone_sim_model.setPropellerThrust(action_list)

        # get new state:
        new_state = self.get_current_state()
        drone_dq = new_state["drone"]  # (8,)  np array
        target_dq = new_state["target"] # (8,) np array

        # calculate reward
        reward = self.calculate_reward(new_state)

        # Others
        terminated = False
        truncated = False
        info = {}
        return new_state, reward, terminated, truncated, info

    def get_current_state(self):
        '''
        Gets the current state from the CoppeliaSim simulation
        :return: state in dictionary form
        '''

        # Get simulation parameters - for aesthetic reasons
        clientID = self.drone_sim_model.client_ID
        heli = self.drone_sim_model.heli_handle
        target = self.drone_sim_model.target_handle
        opmode = sim.simx_opmode_blocking

        # Get drone orientation quaternion
        _, drone_quat = sim.simxGetObjectQuaternion(clientID, heli, -1, opmode)

        # Get drone position
        _, drone_pos = sim.simxGetObjectPosition(clientID, heli, -1, opmode)

        # Get target orientation quaternion
        _, target_quat = sim.simxGetObjectQuaternion(clientID, target, -1, opmode)

        # Get target position
        _, target_pos = sim.simxGetObjectPosition(clientID, target, -1, opmode)

        # Construct dual quaternion for drone
        drone_dual_quat = utils.dual_quaternion_from_trans_and_rot(drone_pos, drone_quat)

        # Construct dual quaternion for target
        target_dual_quat = utils.dual_quaternion_from_trans_and_rot(target_pos, target_quat)

        # Construct state dictionary
        state = {
            'drone': drone_dual_quat,
            'target': target_dual_quat
        }

        return state

    def calculate_reward(self, new_state):
        #TODO - complete this function!
        reward = 0.0
        return reward

    def reset(self, seed=None):
        # reset the state
        self.initial_drone_position = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.initial_target_position = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.state = {
            'drone': self.initial_drone_position,
            'target': self.initial_target_position
        }

        # Stop the simulation
        sim.simxStopSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_blocking)
        sim.simxGetPingTime(self.drone_sim_model.client_ID)
        time.sleep(0.01)  # ensure the coppeliasim is stopped

        # Restart the simulation
        sim.simxStartSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_oneshot)

        # in cart pole they sent a zero action to the simulation here - why?
        sim.simxSynchronousTrigger(self.drone_sim_model.client_ID)
        sim.simxGetPingTime(self.drone_sim_model.client_ID)

        return self.state, {}

    def render(self):
        return None

    def close(self):
        sim.simxStopSimulation(self.drone_sim_model.client_ID, sim.simx_opmode_blocking) # stop the simulation
        sim.simxFinish(-1)  # Close the connection
        print('Close the environment')
        return None

