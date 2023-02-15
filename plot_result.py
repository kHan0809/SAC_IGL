import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import matplotlib.pyplot as plt
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	args = parser.parse_args()

	envs = [
		# "FetchReach-v1",
		"FetchPush-v1",
	]
	p_dir = "./results/"
	ext = ".npy"

	for idx,env in enumerate(envs):
		for i,file_name in enumerate(os.listdir(p_dir)):
			if env in file_name:
				data = np.load(p_dir + file_name)
				# plt.subplot(1,3,i+1)
				plt.subplot(1,1,1)
				plt.plot(data)
				plt.title(env)
				plt.title(file_name[:-6])
				plt.xlabel("5000 step")
	plt.show()





