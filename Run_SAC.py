import numpy as np
import torch
import gym
import argparse
import os
from model.SAC import SAC
from Utils.gym_robotics_wrapper import robot_wrapper

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, eval_episodes=10, render=False):

	max_action = float(eval_env.action_space.high[0])
	avg_reward, success_rate = 0., 0.
	for _ in range(eval_episodes):
		state, done, success_add = eval_env.reset(), False ,0.0
		while not done:
			action = policy.select_action(state,evaluate=True)
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
	# Experiment
	parser.add_argument("--policy", default="SAC")               # Policy name
	parser.add_argument("--env", default="FetchPush-v1")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--action_start_steps", default=5e3, type=int)  # How often (time steps) we evaluate
	parser.add_argument("--update_start_steps", default=2e3, type=int)  # How often (time steps) we evaluate
	parser.add_argument("--eval_freq", default=2e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps",   default=15e5, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", default=False)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--render", default=False)  # Save model and optimizer parameters
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

	env = robot_wrapper(gym.make(args.env))
	env.seed(args.seed)                # Set seeds
	env.action_space.seed(args.seed)

	eval_env = robot_wrapper(gym.make(args.env))
	eval_env.seed(seed=args.seed+100)  # Set seeds

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

		next_state, reward, done, info = env.step(action*max_action)  # Step
		episode_steps += 1
		episode_reward += reward


		fix_done = False if episode_steps == env._max_episode_steps else float(done)
		agent.buffer.add(state,action,next_state,reward,fix_done)
		state = next_state

		if agent.buffer.size > args.update_start_steps:
			agent.update_parameters()


		if done:
			state, episode_reward, episode_steps = env.reset(), 0, 0
			agent._soft_update()

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			evaluations.append(eval_policy(agent, eval_env, render=args.render))
			np.save(f"./results/{file_name}", evaluations)
