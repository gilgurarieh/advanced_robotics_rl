# utility functions
import numpy as np


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