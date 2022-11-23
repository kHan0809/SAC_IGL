import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC
import matplotlib.pyplot as plt
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	args = parser.parse_args()

	envs = [
		"halfcheetah-random-v0",
		"hopper-random-v0",
		"walker2d-random-v0",
		"halfcheetah-medium-v0",
		"hopper-medium-v0",
		"walker2d-medium-v0",
		"halfcheetah-expert-v0",
		"hopper-expert-v0",
		"walker2d-expert-v0",
		"halfcheetah-medium-expert-v0",
		"hopper-medium-expert-v0",
		"walker2d-medium-expert-v0",
		"halfcheetah-medium-replay-v0",
		"hopper-medium-replay-v0",
		"walker2d-medium-replay-v0",
	]
	p_dir = "./results/"
	ext = ".npy"

	for idx,env in enumerate(envs):
		try:
			file_name = f"{args.policy}_{env}_{args.seed}"
			data = np.load(p_dir + file_name + ext)
			plt.subplot(3,5,idx+1)
			plt.plot(data)
			plt.title(env)
			plt.xlabel("time steps")
		except:
			pass
	plt.show()





