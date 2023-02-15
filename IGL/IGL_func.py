import pickle
import os
import numpy as np
import math
import quaternion
import matplotlib.pyplot as plt
import quaternion

import copy

def value_inter(v1, v2, coef):
    return v1*(1-coef) + v2*coef

def traj_interpolation_pos(pos1,pos2,coef):
    traj1_len = len(pos1) - 1
    traj2_len = len(pos2) - 1
    inter_traj_real_len = (traj1_len) * (1 - coef) + (traj2_len) * coef
    inter_traj_len = math.ceil(inter_traj_real_len)

    ratio1 = traj1_len / inter_traj_len
    ratio2 = traj2_len / inter_traj_len

    temp = []
    for i in range(inter_traj_len + 1):
        if i == 0:
            temp.append(value_inter(pos1[0],pos2[0], coef))
        elif i == inter_traj_len:
            temp.append(value_inter(pos1[-1], pos2[-1], coef))
        else:
            idx1 = i * ratio1
            idx2 = i * ratio2
            if (idx1 % 1.0) < 0.0001:
                new_xyz_1 = pos1[math.ceil(idx1)]
            else:
                pre = math.floor(idx1)
                cur = math.ceil(idx1)
                new_xyz_1 = value_inter(pos1[pre], pos1[cur], (idx1 % 1.0))

            if (idx2 % 1.0) < 0.0001:
                new_xyz_2 = pos2[math.ceil(idx2)]
            else:
                pre = math.floor(idx2)
                cur = math.ceil(idx2)
                new_xyz_2 = value_inter(pos2[pre], pos2[cur], (idx2 % 1.0))

            temp.append(value_inter(new_xyz_1, new_xyz_2,coef))
    return np.array(temp)



def traj_interpolation_quat(quat1,quat2,coef):
    traj1_len = len(quat1) - 1
    traj2_len = len(quat2) - 1
    inter_traj_real_len = (traj1_len) * (1 - coef) + (traj2_len) * coef
    inter_traj_len = math.ceil(inter_traj_real_len)

    ratio1 = traj1_len / inter_traj_len
    ratio2 = traj2_len / inter_traj_len

    quat1 = quaternion.as_quat_array(quat1)
    quat2 = quaternion.as_quat_array(quat2)
    temp = []
    for i in range(inter_traj_len + 1):
        if i == 0:
            new_q = quaternion.slerp_evaluate(quat1[0], quat2[0],   coef)
            temp.append(quaternion.as_float_array(new_q))
        elif i == inter_traj_len:
            new_q = quaternion.slerp_evaluate(quat1[-1], quat2[-1], coef)
            temp.append(quaternion.as_float_array(new_q))
        else:
            idx1 = i * ratio1
            idx2 = i * ratio2
            if (idx1 % 1.0) < 0.00001:
                new_q_1 = quat1[math.ceil(idx1)]
            else:
                pre = math.floor(idx1)
                cur = math.ceil(idx1)
                new_q_1 = quaternion.slerp_evaluate(quat1[pre], quat1[cur], (idx1 % 1.0))

            if (idx2 % 1.0) < 0.00001:
                new_q_2 = quat2[math.ceil(idx2)]
            else:
                pre = math.floor(idx2)
                cur = math.ceil(idx2)
                new_q_2 = quaternion.slerp_evaluate(quat2[pre], quat2[cur], (idx2 % 1.0))

            new_q = quaternion.slerp_evaluate(new_q_1, new_q_2, coef)
            # if coef == 0.5:
            #     print("===========")
            #     print(new_q_1)
            #     print(new_q_2)
            #     print(new_q)
            temp.append(quaternion.as_float_array(new_q))
    # print(np.array(temp).shape,tt.shape)
    # for i in range(len(temp)):
    #     print(np.array(temp)[i],tt[i])
    # raise
    return np.array(temp)

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0
def quat2mat(quat):
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, f"Invalid shape quat {quat}"

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))



# def slerp(quat1, quat2, t):
#     """Spherically interpolates between quat1 and quat2 by t.
#     The parameter t is clamped to the range [0, 1]
#     """
#
#     # https://en.wikipedia.org/wiki/Slerp
#
#     v0 = normalise(quat1)
#     v1 = normalise(quat2)
#
#     dot = vector4.dot(v0, v1)
#
#     # TODO: fixlater
#     # If the inputs are too close for comfort,
#     # linearly interpolate and normalize the result.
#     # if abs(dot) > 0.9995:
#     #     pass
#
#     # If the dot product is negative, the quaternions
#     # have opposite handed-ness and slerp won't take
#     # the shorter path. Fix by reversing one quaternion.
#     if dot < 0.0:
#         v1 = -v1
#         dot = -dot
#
#     # clamp
#     dot = np.clamp(dot, -1.0, 1.0)
#     theta = np.acos(dot) * t
#
#     v2 = v1 - v0 * dot
#     res = v0 * np.cos(theta) + v2 * np.sin(theta)
#     return res