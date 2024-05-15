# utility functions
import numpy as np
from dqrobotics import DQ
import sim


def get_dual_quaternion(client_ID, object_handle):
    """
    Retrieves the position and quaternion of a specified object and converts it to a dual quaternion.
    """
    errorCode, position = sim.simxGetObjectPosition(client_ID, object_handle, -1, sim.simx_opmode_blocking)
    errorCode, quaternion = sim.simxGetObjectQuaternion(client_ID, object_handle, -1, sim.simx_opmode_blocking)
    if errorCode == sim.simx_return_ok:
        # Create a dual quaternion (dqrobotics uses [w, x, y, z] order for quaternion)
        dual_quat = DQ([quaternion[3], quaternion[0], quaternion[1], quaternion[2], 0, position[0], position[1], position[2]])
        return dual_quat
    else:
        print("Error retrieving position or orientation.")
        return None

def dq_position_difference(current_dq, target_dq):
    """
    Calculate the difference between two positions represented as dual quaternions.
    :param current_dq: DQ, current position in dual quaternion form
    :param target_dq: DQ, target position in dual quaternion form
    :return: DQ, the dual quaternion representing the difference
    """
    return target_dq - current_dq

def calculate_distance_and_orientation(drone_dq, target_dq):
    """
    Calculate the Euclidean distance and angular difference between two poses represented by dual quaternions.
    Args:
        drone_dq (DQ): The dual quaternion representing the drone's pose.
        target_dq (DQ): The dual quaternion representing the target's pose.
    Returns:
        tuple: A tuple containing the positional distance and the rotation angle in radians.
    """
    # Calculate positional distance
    pos_drone = drone_dq.translation().vec()  # Using .vec() to directly access the vector part of the translation
    pos_target = target_dq.translation().vec()
    positional_distance = np.linalg.norm(pos_drone - pos_target)

    # Calculate orientation difference
    rotation_difference_dq = drone_dq.rotation().conj() * target_dq.rotation()
    rotation_angle = rotation_difference_dq.rotation_angle()  # Extract the rotation angle

    return positional_distance, rotation_angle


def dual_quaternion_from_trans_and_rot(translation, rotation):
    '''
    Constructs a dual quaternion from translation and rotation quaternions
    '''
    translation_quat = np.array([0, translation[0], translation[1], translation[2]])
    dual_translation_quat = quaternion_multiply(rotation, translation_quat) * 0.5

    dual_quat = np.concatenate((rotation, dual_translation_quat))
    return dual_quat


def quaternion_multiply(q1, q2):
    '''
    Multiplies two quaternions
    '''
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

