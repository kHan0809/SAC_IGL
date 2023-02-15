import pickle
import os
import numpy as np
import math
import quaternion
import matplotlib.pyplot as plt
from IGL_func import traj_interpolation_pos, traj_interpolation_quat
from IGL_utils import get_args, concat_all_data
from IGL_func import quat2mat
args  = get_args()
data_concat = concat_all_data(args.env)

eef_pos, obj_pos, grip_pos, obj_quat, goal_pos, next_eef_pos, next_grip_pos, sg = [], [], [], [], [], [], [], []
coefs       = np.linspace(0,1,3,endpoint=True)
subgoal_num = data_concat[0]["subgoal"][-1][0]

for k in range(subgoal_num+1):
    for i in range(len(data_concat)-1):
        for j in range(i + 1, len(data_concat)):
            i = 2
            j = 8
            idx1 = np.where(data_concat[i]["subgoal"] == k+2)[0]
            idx2 = np.where(data_concat[j]["subgoal"] == k+2)[0]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            z_axis_scale = 100.0
            defal = 0.1

            X, Y, Z = zip(*data_concat[i]["observation"][idx1, :3])
            ax.scatter(X, Y, Z, color='m')

            X, Y, Z = zip(*data_concat[j]["observation"][idx2, :3])
            ax.scatter(X, Y, Z, color='c')

            X, Y, Z = zip(*data_concat[i]["observation"][idx1, -3:])
            ax.scatter(X, Y, Z, color='b')
            X, Y, Z = zip(*data_concat[j]["observation"][idx2, -3:])
            ax.scatter(X, Y, Z, color='b')

            for coef in coefs:
                if coef == 0 or coef ==  1:
                    continue

                new_eef_pos = traj_interpolation_pos(data_concat[i]["observation"][idx1, :3],    data_concat[j]["observation"][idx2, :3],  coef)  # robot pos
                new_obj_pos   = traj_interpolation_pos(data_concat[i]["observation"][idx1, 3:6],   data_concat[j]["observation"][idx2, 3:6], coef)  # obj pos
                new_grip_pos      =  traj_interpolation_pos(data_concat[i]["observation"][idx1, 9:11],   data_concat[j]["observation"][idx2, 9:11], coef)
                new_obj_quat  = traj_interpolation_quat(data_concat[i]["observation"][idx1, 11:15],data_concat[j]["observation"][idx2, 11:15], coef)  # obj quat
                new_goal_pos  = traj_interpolation_pos(data_concat[i]["observation"][idx1,-3:],    data_concat[j]["observation"][idx2, -3:],  coef)  # goal pos
                new_next_eef_pos = traj_interpolation_pos(data_concat[i]["next_obervation"][idx1, :3], data_concat[j]["next_obervation"][idx2, :3], coef)  # robot pos
                new_next_grip_pos = traj_interpolation_pos(data_concat[i]["next_obervation"][idx1, 9:11],   data_concat[j]["next_obervation"][idx2, 9:11], coef)
                new_sugoal         = np.ones(shape=(new_grip_pos.shape[0],1)) * k

                eef_pos.append(new_eef_pos)
                obj_pos.append(new_obj_pos)
                grip_pos.append(new_grip_pos)
                obj_quat.append(new_obj_quat)
                goal_pos.append(new_goal_pos)
                next_eef_pos.append(new_next_eef_pos)
                next_grip_pos.append(new_next_grip_pos)
                sg.append(new_sugoal)

                X, Y, Z = zip(*new_eef_pos)
                ax.scatter(X, Y, Z, color='r')
                X, Y, Z = zip(*new_goal_pos)
                ax.scatter(X, Y, Z, color='b')

            ax.set_xlabel('XX')
            ax.set_ylabel('YY')
            ax.set_zlabel('ZZ')
            plt.show()
            raise

for k in range(subgoal_num+1):
    for i in range(len(data_concat)):
        coef = 1.0
        idx1 = np.where(data_concat[i]["subgoal"])[0]

        new_eef_pos = traj_interpolation_pos(data_concat[i]["observation"][idx1, :3],data_concat[i]["observation"][idx1, :3], coef)  # robot pos
        new_obj_pos = traj_interpolation_pos(data_concat[i]["observation"][idx1, 3:6],data_concat[i]["observation"][idx1, 3:6], coef)  # obj pos
        new_grip_pos = traj_interpolation_pos(data_concat[i]["observation"][idx1, 9:11],data_concat[i]["observation"][idx1, 9:11], coef)
        new_obj_quat = traj_interpolation_quat(data_concat[i]["observation"][idx1, 11:15],data_concat[i]["observation"][idx1, 11:15], coef)  # obj quat
        new_goal_pos = traj_interpolation_pos(data_concat[i]["observation"][idx1, -3:],data_concat[i]["observation"][idx1, -3:], coef)  # goal pos
        new_next_eef_pos = traj_interpolation_pos(data_concat[i]["next_obervation"][idx1, :3],data_concat[i]["next_obervation"][idx1, :3], coef)  # robot pos
        new_next_grip_pos = traj_interpolation_pos(data_concat[i]["next_obervation"][idx1, 9:11],data_concat[i]["next_obervation"][idx1, 9:11], coef)
        new_sugoal         = np.ones(shape=(new_grip_pos.shape[0],1)) * k

        eef_pos.append(new_eef_pos)
        obj_pos.append(new_obj_pos)
        grip_pos.append(new_grip_pos)
        obj_quat.append(new_obj_quat)
        goal_pos.append(new_goal_pos)
        next_eef_pos.append(new_next_eef_pos)
        next_grip_pos.append(new_next_grip_pos)
        sg.append(new_sugoal)





