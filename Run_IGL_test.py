import torch
import gym

from model.IGL import IGL, make_subgoal
import numpy as np
import argparse
import os
from Utils.gym_robotics_wrapper import robot_wrapper


def eval_policy(policy, env_name, seed, seed_offset=200, eval_episodes=20,render=False):
	eval_env = robot_wrapper(gym.make(env_name))
	eval_env.seed(seed + seed_offset)
	max_action = float(eval_env.action_space.high[0])

	avg_reward, success_rate = 0., 0.
	for _ in range(eval_episodes):
		state, done, success_add = eval_env.reset(), False ,0.0
		subgoal_class = make_subgoal(args.env)
		while not done:
			action = policy.select_action(state,subgoal_class.get_subgoal(state),evaluate=True)
			state, reward, done, info = eval_env.step(action*max_action)
			if render:
				eval_env.render()

			avg_reward  += reward
			success_add += info['is_success']
		if success_add >= 5.0:
			success_rate += 1.0

	avg_reward /= eval_episodes
	success_rate /= eval_episodes
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}  success rate: {success_rate:.2f}" )
	print("---------------------------------------")
	return success_rate


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="IGL")  # Policy name
	parser.add_argument("--env", default="FetchReach-v1", help = "FetchReach-v1 FetchPush-v1 FetchPickAndPlace-v1")  # OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--hidden_dim", default=[512, 256, 256])  # Target network update rate
	parser.add_argument("--batch_size", default=256, type=int)  # batch size
	parser.add_argument("--epoch", default=150, type=int)  # batch size
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
	agent = IGL(16,5,args)
	agent.load_model(os.getcwd()+'/IGL/IGL_model/', args.env)
	eval_policy(agent, args.env, args.seed, render=args.render)