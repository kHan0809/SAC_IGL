import copy

import numpy as np
import argparse
import os
import pickle
def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--env", default="FetchPush-v1", help = "FetchReach-v1 FetchPush-v1 FetchPickAndPlace-v1")  # OpenAI gym environment name
  parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--hidden_dim_igl", default=[1024, 256, 256])  # Target network update rate
  parser.add_argument("--batch_size", default=512, type=int)  # batch size
  parser.add_argument("--epoch", default=801, type=int)  # batch size
  parser.add_argument("--lr", default=3e-4, type=float)  # batch size

  args = parser.parse_args()
  return args

def concat_all_data(env_name):
    data_concat = []
    for pickle_data in os.listdir(os.getcwd() + '/../Demo/Demo_data'):
        if env_name in pickle_data:
            with open(os.getcwd() + '/../Demo/Demo_data/' + pickle_data, 'rb') as f:
                data = pickle.load(f)
                data_concat.extend(data)
    return  data_concat