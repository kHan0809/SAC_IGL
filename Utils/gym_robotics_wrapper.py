import numpy as np
import gym

class robot_wrapper():
    def __init__(self, robo_env):
        self.env = robo_env
        self.state_dim = self.get_state_dim()
        self._max_episode_steps = 50

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.state_dim, np.float64)
        self.action_space = self.env.action_space

    def reset(self):
        dict_state = self.env.reset()
        state = self.dict2num(dict_state)
        return state

    def render(self):
        self.env.render()

    def step(self,action):
        next_dict_state, reward, done, info = self.env.step(action)
        next_state = self.dict2num(next_dict_state)
        return next_state, reward, done, info


    def dict2num(self,dict_state):
        state_lst = []
        for key in dict_state.keys():
            if 'achieved_goal' in key:
                pass
            else:
                state_lst.append(dict_state[key].flatten())
        return np.concatenate(state_lst)

    def get_state_dim(self):
        dict_state = self.env.reset()
        state = self.dict2num(dict_state)
        return state.shape

    def seed(self,seed):
        self.env.seed(seed)

