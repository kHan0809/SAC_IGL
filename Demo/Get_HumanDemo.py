import copy

import metaworld
import random
import numpy as np
import pickle
import argparse
from Utils.gym_robotics_wrapper import robot_wrapper
import gym
import torch
from demo_utils import human_key_control, get_args, reach_control, selection_control, make_subgoal

args = get_args()

def get_epi():
  env = gym.make(args.env)
  # Set seeds
  env.action_space.seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  state, success_count = env.reset(), 0

  epi_obs, epi_act, epi_next_obs, epi_subgoal = [], [], [], []
  subgoal_class = make_subgoal(env_name=args.env)
  subgoal = subgoal_class.get_subgoal(state)
  for i in range(env._max_episode_steps):

    a = selection_control(state, i, args.env,subgoal)

    subgoal = subgoal_class.get_subgoal(state)
    epi_obs.append(np.concatenate((state['observation'], state['desired_goal']), axis=0))
    epi_act.append(a)
    next_state, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    epi_next_obs.append(np.concatenate((next_state['observation'],next_state['desired_goal']),axis=0))
    epi_subgoal.append(copy.deepcopy(subgoal))
    # print(state['observation'],state['observation'].shape, a.shape, subgoal)
    state=next_state

    # env.render()
    if info['is_success']:
      success_count += 1
      if success_count >= 5:
        print("Epi Length : ",i)
        env.close()
        break
  episode = dict(observation=np.array(epi_obs),action=np.array(epi_act),next_obervation=np.array(epi_next_obs),subgoal=np.array(epi_subgoal)) #나중에 subgoal도 포함하는 코드를 짜야될듯
  return episode

if __name__ == "__main__":

  total_epi = []
  sample_num = "0"
  traj_num = 0 # 이거 건들지마
  while True:
    epi = get_epi()
    for i in range(10):

      save_flag = 'y'
      # save_flag = input()
      if save_flag == ("y" or "Y"):
        if len(epi['observation']) >= 12:
          total_epi.append(epi)
          traj_num += 1
        break
      elif save_flag == ("n" or "N"):
        break
      else:
        continue
    if traj_num == 1000:
      break
    print(traj_num)
    with open('./Demo_data/data_'+args.env+"_"+sample_num+'.pickle', 'wb') as f:
      pickle.dump(total_epi, f, pickle.HIGHEST_PROTOCOL)
