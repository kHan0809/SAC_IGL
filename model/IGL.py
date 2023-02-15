import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam
import copy
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataSet(Dataset):
    def __init__(self,numpy_x,numpy_y):
        self.x = numpy_x
        self.y = numpy_y
        self.shape = self.x.shape
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

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
            target = np.array([state[3], state[4], state[5] + 0.055])
            # print(target,state[:3],np.linalg.norm(target - state[:3]))
            if np.linalg.norm(target - state[:3]) > 0.010 and self.subgoal == 0:
                pass
            elif np.linalg.norm(target - state[:3]) < 0.010 and self.subgoal == 0:
                self.subgoal += 1
            elif np.linalg.norm(state[3:6] - state[:3]) < 0.010 and self.subgoal == 1:
                self.subgoal += 1
            print(self.subgoal)
            return self.subgoal

def reach_control(current_pos, target_pos, grip_close=False):
    a = ((target_pos - current_pos) * 5).clip(-1.0, 1.0)
    if grip_close:
        a = np.concatenate((a, np.array([-0.01])), axis=0)
    else:
        a = np.concatenate((a, np.array([1.0])), axis=0)
    return a

class IGL(object):
    def __init__(self, state_dim, next_state_dim, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = IGL_mlp(state_dim,next_state_dim,args.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)

    def dataset_split(self,x,y, split=False):
        total_len = x.shape[0]
        idx = np.array(list(range(total_len)),dtype=int)
        split_len = round(total_len*0.95)
        print(split_len)

        choice = np.random.choice(total_len, split_len, replace=False)
        train_idx = idx[choice]
        idx = np.delete(idx,choice)
        test_idx  = idx
        train_x, test_x, train_y, test_y = x[train_idx], x[test_idx], y[train_idx], y[test_idx]

        train_set = CustomDataSet(train_x,train_y)
        test_set  = CustomDataSet(test_x,test_y)

        self.train_loader = DataLoader(train_set, shuffle=True,batch_size=self.args.batch_size)
        self.test_loader  = DataLoader(test_set, shuffle=True,batch_size =self.args.batch_size)



    def select_action(self,obs,subgoal,evaluate=True):
        if subgoal == 0:
            target = copy.deepcopy(obs[3:6])
            target[2] = target[2] + 0.055
            return reach_control(obs[:3],target)
        # if subgoal == 1:
        #     return reach_control(obs[:3],obs[3:6],grip_close=True)

        self.actor.eval()
        state = torch.FloatTensor(np.concatenate((obs[0:6], obs[9:15], obs[-3:], subgoal), axis=0)).to(self.device).unsqueeze(0)
        next_pos = self.actor(state).detach().cpu().numpy().flatten()
        pos_action  = (next_pos[:3] - obs[:3]) * 10
        # print(obs.shape,obs)
        # print(obs[9:11] - next_pos[3:])
        grip_action = np.sum(next_pos[3:] - obs[9:11]).reshape(1)/2
        # grip_action = np.array([-0.01]).reshape(1)
        # print(obs[9:11] - next_pos[3:], grip_action)



        return np.concatenate((pos_action, grip_action))

    def noise_action(self,obs,goal,subgoal):
        self.actor.eval()
        state = torch.FloatTensor(np.concatenate((obs[0:6], obs[9:15], goal, subgoal), axis=0)).to(self.device).unsqueeze(0)
        next_pos = self.actor(state).detach().cpu().numpy().flatten()
        action = (next_pos - obs[:3])*3
        return np.concatenate((action, np.array([0.0])))



    def train(self):
        for step in range(self.args.epoch):
            check_train_loss, check_test_loss = 0, 0
            for batch_idx, (input,label) in enumerate(self.train_loader):
                self.actor_optimizer.zero_grad()
                predict = self.actor(input.type(torch.FloatTensor).to(self.device))

                loss = F.mse_loss(predict,label.type(torch.FloatTensor).to(self.device))
                loss.backward()

                self.actor_optimizer.step()
                check_train_loss += loss.item()

            for test_idx, (input,label) in enumerate(self.test_loader):
                output = self.actor(input.type(torch.FloatTensor).to(self.device))
                test_loss = F.mse_loss(output,label.to(self.device))
                check_test_loss += test_loss.item()
            if step % 10 == 0:
                print("step : ", step ,"train loss :", check_train_loss, "test loss : ", check_test_loss)
        self.save_model(os.getcwd()+'/IGL_model/', self.args.env)
    def test(self):
        self.actor.eval()
        for test_idx, (input, label) in enumerate(self.test_loader):
            output = self.actor(input.type(torch.FloatTensor).to(self.device))
            print("======================")
            print(input[0])
            print(output[0],label[0])


    def save_model(self,dir,task):
        torch.save(self.actor.state_dict(), dir + task + '.pt')
    def load_model(self,dir,task):
        # self.load_model(os.getcwd()+'/IGL_model/', self.args.env)
        self.actor.load_state_dict(torch.load(dir + task + '.pt'))

class IGL_mlp(nn.Module):
    def __init__(self,state_dim,next_state_dim,hidden_dim,activation=nn.ReLU()):
        super(IGL_mlp, self).__init__()
        self.hidden_dim = list(hidden_dim)
        self.net = mlp([state_dim] + self.hidden_dim, activation)
        self.mu_layer = nn.Linear(self.hidden_dim[-1], next_state_dim)

    def forward(self,state):
        output = self.net(state)
        mean   = self.mu_layer(output)
        return mean


def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]),nn.BatchNorm1d(sizes[j+1]), act]
    return nn.Sequential(*layers)