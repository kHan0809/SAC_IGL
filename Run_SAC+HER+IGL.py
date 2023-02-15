import numpy as np
import torch
import gym
import argparse
import os
from model.SAC_HER import SAC_HER
from model.IGL import IGL


def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def eval_policy(policy, env_name, seed, seed_offset=100, eval_episodes=5, render=False):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)
    max_action = float(eval_env.action_space.high[0])

    avg_reward, success_rate = 0., 0.
    for _ in range(eval_episodes):
        observation, done, success_add = eval_env.reset(), False, 0.0
        obs, ag, g = observation['observation'], observation['achieved_goal'], observation['desired_goal']
        while not done:
            action = policy.select_action(agent._preproc_inputs(obs, g), evaluate=True)
            observation, reward, done, info = eval_env.step(action * max_action)
            obs, ag, g = observation['observation'], observation['achieved_goal'], observation['desired_goal']
            if render:
                eval_env.render()

            avg_reward += reward
            success_add += info['is_success']
        if success_add >= 5.0:
            success_rate += 1.0

    avg_reward /= eval_episodes
    success_rate /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}  success rate: {success_rate:.2f}")
    print("---------------------------------------")
    return success_rate


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="SAC+HER+IGL")  # Policy name
    parser.add_argument("--env", default="FetchPush-v1")  # OpenAI gym environment name
    parser.add_argument("--seed", default=2, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--action_start_steps", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--update_start_steps", default=2e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", default=False)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--render", default=False)  # Save model and optimizer parameters
    # SAC
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--alpha", default=0.2)
    parser.add_argument("--hidden_dim", default=[256, 256])  # Target network update rate
    parser.add_argument("--automatic_entropy_tuning", default=False)  # Target network update rate
    parser.add_argument("--lr", default=3e-4, type=float)  # batch size
    # Her
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')

    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(args.env)
    env_params = get_env_params(env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    max_action = float(env.action_space.high[0])

    # Initialize policy
    agent = SAC_HER(env_params, env, args)
    igl   = IGL(13, 3, args)
    igl.load_model(os.getcwd() + '/IGL/IGL_model/', args.env)

    observation, evaluations = env.reset(), []
    obs, ag, g = observation['observation'], observation['achieved_goal'], observation['desired_goal']
    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
    for t in range(int(args.max_timesteps)):

        if args.action_start_steps > t:
            action = env.action_space.sample() / max_action  # Sample random action
        else:
            action = agent.select_action(agent._preproc_inputs(obs, g))  # Sample action from policy
        action += igl.noise_action(obs,g)

        observation_new, _, done, info = env.step(action * max_action)  # Step
        obs_new, ag_new = observation_new['observation'], observation_new['achieved_goal']

        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())
        ep_g.append(g.copy())
        ep_actions.append(action.copy())
        # re-assign the observation
        obs = obs_new
        ag = ag_new

        if done:
            ep_obs.append(obs.copy()), ep_ag.append(ag.copy())
            ep_obs, ep_ag, ep_g, ep_actions = np.expand_dims(np.array(ep_obs), axis=0), np.expand_dims(np.array(ep_ag),
                                                                                                       axis=0), \
                                              np.expand_dims(np.array(ep_g), axis=0), np.expand_dims(
                np.array(ep_actions), axis=0)
            agent.buffer.store_episode([ep_obs, ep_ag, ep_g, ep_actions])

            if t > args.update_start_steps:
                for _ in range(env._max_episode_steps):
                    agent.update_parameters(args.batch_size)

            observation = env.reset()
            obs, ag, g = observation['observation'], observation['achieved_goal'], observation['desired_goal']
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            evaluations.append(eval_policy(agent, args.env, args.seed, render=args.render))
            np.save(f"./results/{file_name}", evaluations)


# 결과 좋으면 Overcomming Exploration 구현해보자.
