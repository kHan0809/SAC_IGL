import torch
import gym

from model.BC import BC
import numpy as np
import argparse
import os
import pickle
from Utils.gym_robotics_wrapper import robot_wrapper


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="BC")  # Policy name
	parser.add_argument("--env", default="FetchPickAndPlace-v1")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--hidden_dim", default=[256, 256])  # Target network update rate
	parser.add_argument("--batch_size", default=16, type=int)  # batch size
	parser.add_argument("--epoch", default=500, type=int)  # batch size
	parser.add_argument("--lr", default=5e-5, type=float)  # batch size
	parser.add_argument("--render", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	env = robot_wrapper(gym.make(args.env))
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	# Initialize policy
	agent = BC(state_dim,action_dim,args)



	obs_np,act_np = agent.pick2np(args.env)
	agent.dataset_split(obs_np,act_np)
	agent.train(save=True)
