import sys

sys.path.append('VREP_RemoteAPIs')
import sim as vrep_sim


# drone simulation model for CoppeliaSim
class DroneModel():
    def __init__(self, name='Drone'):
        """
        :param: name: string
            name of objective
        """
        super(self.__class__, self).__init__()
        self.name = name
        self.client_ID = None

        self.base_handle = None
        self.target_handle = None
        self.heli_handle = None

    def initializeSimModel(self, client_ID):
        try:
            print('Connected to remote API server')
            client_ID != -1
        except:
            print('Failed connecting to remote API server')

        self.client_ID = client_ID

        return_code, self.base_handle = vrep_sim.simxGetObjectHandle(client_ID, 'base',
                                                                                vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('get object base ok.')

        return_code, self.target_handle = vrep_sim.simxGetObjectHandle(client_ID, 'target',
                                                                           vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('get object target ok.')

        return_code, self.heli_handle = vrep_sim.simxGetObjectHandle(client_ID, 'Quadcopter',
                                                                    vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('get object heli ok.')

        # Get the base and target position - IS THIS NECESSARY?
        # return_code, q = vrep_sim.simxGetObjectPosition(self.client_ID, self.base_handle,
        #                                                vrep_sim.simx_opmode_streaming)
        # return_code, q = vrep_sim.simxGetObjectPosition(self.client_ID, self.target_handle,
        #                                                vrep_sim.simx_opmode_streaming)

        # Set the initialized position for each joint
        # self.setPropellerThrust([1,1,1,1])

    def setPropellerThrust(self, thrust):
        vrep_sim.simxPauseCommunication(self.client_ID, True)
        vrep_sim.simxSetFloatSignal(self.client_ID, 'thrust_signal1', thrust[0], vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetFloatSignal(self.client_ID, 'thrust_signal2', thrust[1], vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetFloatSignal(self.client_ID, 'thrust_signal3', thrust[2], vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetFloatSignal(self.client_ID, 'thrust_signal4', thrust[3], vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxPauseCommunication(self.client_ID, False)