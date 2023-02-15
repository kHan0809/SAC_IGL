import pickle
import os
import numpy as np
import math
import quaternion
import matplotlib.pyplot as plt
from IGL_func import traj_interpolation_pos, traj_interpolation_quat
from IGL_utils import get_args, concat_all_data

args  = get_args()
data_concat = concat_all_data(args.env)

eef_pos, obj_pos, grip_pos, obj_quat, goal_pos, next_eef_pos, next_grip_pos, sg = [], [], [], [], [], [], [], []
coefs       = np.linspace(0,1,3,endpoint=True)
subgoal_num = data_concat[0]["subgoal"][-1][0]

for k in range(subgoal_num+1):
    for i in range(len(data_concat)-1):
        for j in range(i + 1, len(data_concat)):
            idx1 = np.where(data_concat[i]["subgoal"] == k)[0]
            idx2 = np.where(data_concat[j]["subgoal"] == k)[0]
            for coef in coefs:
                if coef == 0 or coef == 1.0:
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


#====================================================중간중간 한번씩 더 쪼개줘

x_buff = []
y_buff = []
coefs = np.linspace(0,1,3,endpoint=True)
for k in range(subgoal_num+1):
    idxs = [i for i in range(len(sg)) if sg[i][0] == k]
    for i in range(len(idxs) - 1):
        for j in range(i + 1, len(idxs)):
            for coef in coefs:
                if coef == 0 or coef == 1.0:
                    continue
                new_eef_pos   = traj_interpolation_pos(eef_pos[idxs[i]],    eef_pos[idxs[j]],  coef)  # robot pos
                new_obj_pos   = traj_interpolation_pos(obj_pos[idxs[i]],  obj_pos[idxs[j]],  coef)  # obj pos
                new_grip_pos  = traj_interpolation_pos(grip_pos[idxs[i]],  grip_pos[idxs[j]], coef)
                new_obj_quat  = traj_interpolation_pos(obj_quat[idxs[i]],  obj_quat[idxs[j]], coef)
                new_goal_pos     = traj_interpolation_pos(goal_pos[idxs[i]],  goal_pos[idxs[j]],  coef)  # goal pos
                new_sugoal = np.ones(shape=(new_grip_pos.shape[0], 1)) * k
                new_next_eef_pos = traj_interpolation_pos(next_eef_pos[idxs[i]], next_eef_pos[idxs[j]],  coef)  # next_robot_pos
                new_next_grip_pos = traj_interpolation_pos(next_grip_pos[idxs[i]], next_grip_pos[idxs[j]],coef)  # next_grip_pos

                x_buff.append(np.concatenate((new_eef_pos,new_obj_pos,new_grip_pos,new_obj_quat,new_goal_pos,new_sugoal),axis=1))
                y_buff.append(np.concatenate((new_next_eef_pos, new_next_grip_pos), axis=1))

for k in range(subgoal_num+1):
    idxs = [i for i in range(len(sg)) if sg[i][0] == k]
    for i in range(len(idxs)):
        coef = 1.0
        new_eef_pos = traj_interpolation_pos(eef_pos[idxs[i]], eef_pos[idxs[i]], coef)  # robot pos
        new_obj_pos = traj_interpolation_pos(obj_pos[idxs[i]], obj_pos[idxs[i]], coef)  # obj pos
        new_grip_pos = traj_interpolation_pos(grip_pos[idxs[i]], grip_pos[idxs[i]], coef)
        new_obj_quat = traj_interpolation_pos(obj_quat[idxs[i]], obj_quat[idxs[i]], coef)
        new_goal_pos = traj_interpolation_pos(goal_pos[idxs[i]], goal_pos[idxs[i]], coef)  # goal pos
        new_sugoal = np.ones(shape=(new_grip_pos.shape[0], 1)) * k
        new_next_eef_pos = traj_interpolation_pos(next_eef_pos[idxs[i]], next_eef_pos[idxs[i]], coef)  # next_robot_pos
        new_next_grip_pos = traj_interpolation_pos(next_grip_pos[idxs[i]], next_grip_pos[idxs[i]],coef)  # next_grip_pos

        x_buff.append(np.concatenate((new_eef_pos, new_obj_pos, new_grip_pos, new_obj_quat, new_goal_pos, new_sugoal), axis=1))
        y_buff.append(np.concatenate((new_next_eef_pos, new_next_grip_pos), axis=1))


x_buff = np.concatenate(x_buff,axis=0)
y_buff = np.concatenate(y_buff,axis=0)
print(x_buff.shape,y_buff.shape)
np.save(os.getcwd()+'/IGL_data/IGL_x_'+args.env+'.npy', x_buff)
np.save(os.getcwd()+'/IGL_data/IGL_y_'+args.env+'.npy', y_buff)


