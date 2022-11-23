import numpy as np
import torch
import gym
import argparse
import os
from model.SAC import SAC


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, seed_offset=100, eval_episodes=5):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)
	max_action = float(eval_env.action_space.high[0])

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(state,evaluate=True)
			state, reward, done, _ = eval_env.step(action*max_action)
			avg_reward += reward

	avg_reward /= eval_episodes
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="SAC")               # Policy name
	parser.add_argument("--env", default="Hopper-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--action_start_steps", default=5e3, type=int)  # How often (time steps) we evaluate
	parser.add_argument("--update_start_steps", default=2e3, type=int)  # How often (time steps) we evaluate
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps",   default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", default=False)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# SAC
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--alpha", default=0.2)
	parser.add_argument("--hidden_dim", default=[256,256])  # Target network update rate
	parser.add_argument("--automatic_entropy_tuning", default=False)  # Target network update rate


	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Initialize policy
	agent = SAC(state_dim,action_dim,args)


	state, episode_reward, episode_steps, evaluations = env.reset(), 0, 0, []
	for t in range(int(args.max_timesteps)):

		if args.action_start_steps > t:
			action = env.action_space.sample()/max_action  # Sample random action
		else:
			action = agent.select_action(state)  # Sample action from policy

		next_state, reward, done, _ = env.step(action*max_action)  # Step
		episode_steps += 1
		episode_reward += reward


		fix_done = False if episode_steps == env._max_episode_steps else float(done)
		agent.buffer.add(state,action,next_state,reward,fix_done)
		state = next_state

		if agent.buffer.size > args.update_start_steps:
			agent.update_parameters()

		if done:
			state, episode_reward, episode_steps = env.reset(), 0, 0

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			evaluations.append(eval_policy(agent, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
