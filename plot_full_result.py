import numpy as np
import torch
import gym
import argparse
import os

import matplotlib.pyplot as plt

def move_ave_func(data,screen):
	tmp = []
	for i in range(0,len(data)-screen+1):
		tmp.append(data[i:i+screen].mean())
	return np.array(tmp)
if __name__ == "__main__":

	envs = [
		# "FetchReach-v1",
		# "FetchPush-v1",
		"FetchPickAndPlace"
	]
	p_dir = "./results/"
	ext = ".npy"
	SAC_buffer = []
	SAC_HER_buffer = []
	SAC_HER_OVER_buffer = []
	SAC_HER_IGL_buffer = []
	moving_ave = 50

	for i,file_name in enumerate(os.listdir(p_dir)):
		if envs[0] in file_name:
			# if "SAC_" in file_name:
			# 	data = np.load(p_dir + file_name)
			# 	SAC_buffer.append(data)
			if "SAC+HER_" in file_name:
				data = np.load(p_dir + file_name)
				SAC_HER_buffer.append(data)
			if "SAC+HER+IGL_" in file_name:
				data = np.load(p_dir + file_name)
				SAC_HER_IGL_buffer.append(data)
			if "SAC+HER+OVER" in file_name:
				data = np.load(p_dir + file_name)
				SAC_HER_OVER_buffer.append(data)

	# SAC_buffer = np.array(SAC_buffer)
	SAC_HER_buffer = np.array(SAC_HER_buffer)
	SAC_HER_IGL_buffer = np.array(SAC_HER_IGL_buffer)
	SAC_HER_OVER_buffer = np.array(SAC_HER_OVER_buffer)
	# print(SAC_buffer.shape,SAC_HER_buffer.shape,SAC_HER_IGL_buffer.shape,SAC_HER_OVER_buffer.shape)
	# mean_sac         = move_ave_func(SAC_buffer.mean(axis=0), moving_ave)
	mean_sac_her     = move_ave_func(SAC_HER_buffer.mean(axis=0),moving_ave)
	mean_sac_her_igl = move_ave_func(SAC_HER_IGL_buffer.mean(axis=0), moving_ave)
	mean_sac_her_over = move_ave_func(SAC_HER_OVER_buffer.mean(axis=0), moving_ave)

	std_div = 5
	# std_sac = move_ave_func(SAC_buffer.std(axis=0), moving_ave)
	std_sac_her = move_ave_func(SAC_HER_buffer.std(axis=0),moving_ave)/std_div
	std_sac_her_igl = move_ave_func(SAC_HER_IGL_buffer.std(axis=0), moving_ave)/std_div
	std_sac_her_over = move_ave_func(SAC_HER_OVER_buffer.std(axis=0), moving_ave)/std_div


	# plt.plot(mean_sac,color='g',label='SAC')
	plt.plot(mean_sac_her,color='orange',label='SAC+HER')
	plt.plot(mean_sac_her_igl,color='red',label='SAC+HER+IGL')
	plt.plot(mean_sac_her_over, color='blue', label='SAC+HER+OVER')
	x = np.linspace(0,len(mean_sac_her)-1,len(mean_sac_her))
	# plt.fill_between(x, mean_sac - std_sac, mean_sac + std_sac,alpha=0.2, color='g')
	plt.fill_between(x, mean_sac_her - std_sac_her, mean_sac_her + std_sac_her, alpha=0.2, color='orange')
	plt.fill_between(x, mean_sac_her_igl - std_sac_her_igl, mean_sac_her_igl + std_sac_her_igl,alpha=0.2,color='red')
	plt.fill_between(x, mean_sac_her_over - std_sac_her_over, mean_sac_her_over + std_sac_her_over, alpha=0.2, color='blue')
	plt.title(envs[0])
	plt.xlabel("2000 step")
	plt.ylabel("success rate")
	# plt.legend(loc="upper left")
	plt.legend(loc="lower right")
	plt.grid()
	plt.show()





