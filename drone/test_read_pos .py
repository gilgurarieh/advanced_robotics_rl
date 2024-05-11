import sys
import time

sys.path.append('path_to_vrep_remoteAPIs')  # Update this path
import sim  # Python Remote API for CoppeliaSim
from dqrobotics import DQ


def velocity_to_dq(linear_velocity, angular_velocity):
    """
    Convert linear and angular velocities to a dual quaternion representation using DQ Robotics.
    :param linear_velocity: array-like, contains the linear velocity components [vx, vy, vz]
    :param angular_velocity: array-like, contains the angular velocity components [wx, wy, wz]
    :return: DQ, representing the velocity in dual quaternion form
    """
    # Angular velocity as a pure quaternion
    angular_part = DQ([0, angular_velocity[0], angular_velocity[1], angular_velocity[2]])

    # Linear velocity as a pure quaternion multiplied by 0.5
    linear_part = DQ([0, linear_velocity[0], linear_velocity[1], linear_velocity[2]]) * 0.5

    # Dual quaternion velocity
    velocity_dq = angular_part + DQ.E * (linear_part * angular_part)

    return velocity_dq

def main():
    print('Program started')
    sim.simxFinish(-1)  # Just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim

    if clientID != -1:
        print('Connected to remote API server')
        # Start the simulation:
        sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)

        # Get the handle for the quadcopter base
        errorCode, quadcopterBaseHandle = sim.simxGetObjectHandle(clientID, 'Quadcopter_base', sim.simx_opmode_blocking)
        if errorCode == sim.simx_return_ok:
            print('Quadcopter base handle acquired: ', quadcopterBaseHandle)

            # Main loop - retrieve data and process it
            while True:
                # Position and orientation
                errorCode, position = sim.simxGetObjectPosition(clientID, quadcopterBaseHandle, -1,
                                                                sim.simx_opmode_blocking)
                errorCode, quaternion = sim.simxGetObjectQuaternion(clientID, quadcopterBaseHandle, -1,
                                                                    sim.simx_opmode_blocking)

                # Velocity
                errorCode, linear_velocity, angular_velocity = sim.simxGetObjectVelocity(clientID, quadcopterBaseHandle,
                                                                                         sim.simx_opmode_blocking)

                if errorCode == sim.simx_return_ok:
                    # Display position and quaternion
                    print(f'Position - X: {position[0]}, Y: {position[1]}, Z: {position[2]}')
                    print(
                        f'Quaternion - x: {quaternion[0]}, y: {quaternion[1]}, z: {quaternion[2]}, w: {quaternion[3]}')

                    # Convert and display dual quaternion
                    dq = DQ([quaternion[3], quaternion[0], quaternion[1], quaternion[2], 0, position[0], position[1],
                             position[2]])
                    print(f'Dual Quaternion from Pose: {dq}')

                    # Convert and display dual quaternion from velocity
                    velocity_dq = velocity_to_dq(linear_velocity, angular_velocity)
                    print(f'Dual Quaternion from Velocity: {velocity_dq}')

                time.sleep(1)  # Sleep for a second before the next read

        else:
            print('Failed to get Quadcopter base handle')

        # Stop simulation and close connection:
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
        sim.simxFinish(clientID)
    else:
        print('Failed connecting to remote API server')


if __name__ == '__main__':
    main()
