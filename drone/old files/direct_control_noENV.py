import sys
sys.path.append('VREP_RemoteAPIs')
import sim as vrep_sim
from numpy import random

from Drone_model import DroneModel


print ('Program started')

#%% ------------------------------- Connect to VREP (CoppeliaSim) -------------------------------
vrep_sim.simxFinish(-1) # just in case, close all opened connections
while True:
    client_ID = vrep_sim.simxStart('127.0.0.1', 19997, True, False, 5000, 5) # Connect to CoppeliaSim
    if client_ID > -1: # connected
        print('Connect to remote API server.')
        break
    else:
        print('Failed connecting to remote API server! Try it again ...')

# Open synchronous mode
vrep_sim.simxSynchronous(client_ID, True)
# Start simulation
vrep_sim.simxStartSimulation(client_ID, vrep_sim.simx_opmode_oneshot)
vrep_sim.simxSynchronousTrigger(client_ID)  # trigger one simulation step, takes about 11 ms on Windows 10

drone_model = DroneModel('Drone')
drone_model.initializeSimModel(client_ID)

for i in range(20):
    ## Horizontal control
    # get target relative position to base


    action = [6.4353168663368,6.0979366060253,6.0959686560516,7.5415910104827]
    drone_model.setPropellerThrust(action)

    vrep_sim.simxSynchronousTrigger(client_ID)
    _, ping_time = vrep_sim.simxGetPingTime(client_ID) # make sure the last simulation step is finished

vrep_sim.simxStopSimulation(client_ID, vrep_sim.simx_opmode_blocking) # stop the simulation
vrep_sim.simxFinish(-1)  # Close the connection
print('Program terminated')
