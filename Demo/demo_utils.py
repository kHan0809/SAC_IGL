import copy

import numpy as np
import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--env", default="FetchReach-v1", help = "FetchReach-v1 FetchPush-v1 FetchPickAndPlace-v1")  # OpenAI gym environment name
  parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds

  args = parser.parse_args()
  return args

class make_subgoal():
    def __init__(self,env_name):
        self.subgoal  = np.array([0])
        self.env_name = env_name
    def get_subgoal(self,state):
        if self.env_name == "FetchReach-v1":
            return self.subgoal
        if self.env_name == "FetchPush-v1":
            target = np.array([state['observation'][3], state['observation'][4], state['observation'][5] + 0.055])
            if np.linalg.norm(target - state['observation'][:3]) > 0.010 and self.subgoal == 0:
                pass
            elif np.linalg.norm(target - state['observation'][:3]) < 0.010 and self.subgoal == 0:
                self.subgoal += 1
            return self.subgoal
        if self.env_name == "FetchPickAndPlace-v1":
            target = np.array([state['observation'][3], state['observation'][4], state['observation'][5] + 0.055])
            if np.linalg.norm(target - state['observation'][:3]) > 0.010 and self.subgoal == 0:
                pass
            elif np.linalg.norm(target - state['observation'][:3]) < 0.010 and self.subgoal == 0:
                self.subgoal += 1
            elif np.linalg.norm(state['observation'][3:6] - state['observation'][:3]) < 0.010 and self.subgoal == 1:
                self.subgoal += 1
            return self.subgoal



def reach_control(current_pos, target_pos, grip_close=False):
    a = ((target_pos - current_pos) * 5).clip(-1.0, 1.0)
    if grip_close:
        a = np.concatenate((a, np.array([-0.01])), axis=0)
    else:
        a = np.concatenate((a, np.array([1.0])), axis=0)
    return a

def selection_control(state,i,env_name,subgoal):
    global flag
    if env_name == "FetchReach-v1":
        return reach_control(state['observation'][:3],state['desired_goal'])
    if env_name == "FetchPush-v1":
        if i < 2:
            return np.array([0.0, 0.0, 1.0, -1.0])
        if subgoal == 0:
            target = np.array([state['observation'][3], state['observation'][4], state['observation'][5] + 0.055])
            return reach_control(state['observation'][:3],target)
        else:
            try:
                a = human_key_control(input())
            except:
                a = human_key_control(input())
            return  a
    if env_name == "FetchPickAndPlace-v1":
        if subgoal == 0:
            target = np.array([state['observation'][3], state['observation'][4], state['observation'][5] + 0.055])
            return reach_control(state['observation'][:3],target)
        elif subgoal == 1:
            return reach_control(state['observation'][:3],state['observation'][3:6],grip_close=True)
        else:
            return reach_control(state['observation'][:3], state["desired_goal"],grip_close=True)


def human_key_control(key):
    scale  = 0.6
    if "a" in key:
        a = np.array([0.0, -scale, 0.0, -scale])
    if "d" in key:
        a = np.array([0.0, scale, 0.0, -scale])
    if "w" in key:
        a = np.array([-scale, 0.0, 0.0, -scale])
    if "s" in key:
        a = np.array([scale, 0.0, 0.0, -scale])
    if "r" in key:
        a = np.array([0.0,0.0,scale,-scale])
    if "f" in key:
        a = np.array([0.0,0.0,-scale,-scale])

    if "m" in key:
        a = np.array([0.0, 0.0, 0.0, scale])
    if "," in key:
        a = np.array([0.0, 0.0, 0.0, -scale])

    if "." in key:
        a *= np.array([scale,scale,scale,-scale])

    return a