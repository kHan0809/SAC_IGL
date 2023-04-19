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
import pickle

class CustomDataSet(Dataset):
    def __init__(self,numpy_x,numpy_y):
        self.x = numpy_x
        self.y = numpy_y
        self.shape = self.x.shape
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

class BC(object):
    def __init__(self, state_dim, next_state_dim, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = BC_mlp(state_dim,next_state_dim,args.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,)
        self.actor.eval()

    def dataset_split(self,x,y):
        total_len = x.shape[0]
        idx = np.array(list(range(total_len)),dtype=int)
        split_len = round(total_len*0.95)
        self.state_mean, self.state_std   = x.mean(axis = 0),  x.std(axis=0)
        # self.action_mean, self.action_std = y.mean(axis= 0), y.std(axis=0)
        self.eps = 1e-6
        x = (x - self.state_mean)  / (self.state_std + self.eps)
        # y = (y - self.action_mean) / (self.action_std + self.eps)

        choice = np.random.choice(total_len, split_len, replace=False)
        train_idx = idx[choice]
        idx = np.delete(idx,choice)
        test_idx  = idx
        train_x, test_x, train_y, test_y = x[train_idx], x[test_idx], y[train_idx], y[test_idx]

        train_set = CustomDataSet(train_x,train_y)
        test_set  = CustomDataSet(test_x ,test_y)

        self.train_loader = DataLoader(train_set, shuffle=True,batch_size=self.args.batch_size)
        self.test_loader  = DataLoader(test_set, shuffle=True,batch_size =self.args.batch_size)
        self.actor.eval()

    def dataset_all(self,x,y):
        train_set = CustomDataSet(x, y)
        self.train_loader = DataLoader(train_set, shuffle=True,batch_size=self.args.batch_size)

    def pick2np(self,env_name,parent=False):
        dir_n = '/Demo/Demo_data/' if parent else '/../Demo/Demo_data/'
        for pickle_data in os.listdir(os.getcwd() + dir_n):
            if env_name in pickle_data:
                with open(os.getcwd() + dir_n + pickle_data, 'rb') as f:
                    p_data = pickle.load(f)

        obs_tmp,act_tmp = [], []
        for i in range(len(p_data)):
            obs_tmp.append(np.array(p_data[i]['observation']))
            act_tmp.append(np.array(p_data[i]['action']))
        obs_np = np.concatenate(obs_tmp, axis=0)
        act_np = np.concatenate(act_tmp, axis=0)
        return obs_np, act_np

    def select_action(self,obs,evaluate=True):
        obs = (obs - self.state_mean)  / (self.state_std + self.eps)
        state = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().flatten()
        # action = (self.action_std + self.eps) * action + self.action_mean
        return action

    def train(self,save=False):
        for step in range(self.args.epoch):
            check_train_loss, check_test_loss = 0, 0
            self.actor.train()
            for batch_idx, (input,label) in enumerate(self.train_loader):
                self.actor_optimizer.zero_grad()
                predict = self.actor(input.type(torch.FloatTensor).to(self.device))

                loss = F.mse_loss(predict,label.type(torch.FloatTensor).to(self.device))
                loss.backward()

                self.actor_optimizer.step()
                check_train_loss += loss.item()
            self.actor.eval()
            for test_idx, (input,label) in enumerate(self.test_loader):
                output = self.actor(input.type(torch.FloatTensor).to(self.device))
                test_loss = F.mse_loss(output,label.to(self.device))
                check_test_loss += test_loss.item()
            if step % 10 == 0:
                print("step : ", step ,"train loss :", check_train_loss, "test loss : ", check_test_loss)
        if save:
            self.save_model(os.getcwd()+'/BC_model/', self.args.env)
    def test(self):
        self.actor.eval()
        check_train_loss, check_test_loss = 0, 0
        for test_idx, (input, label) in enumerate(self.test_loader):
            output = self.actor(input.type(torch.FloatTensor).to(self.device))
            test_loss = F.mse_loss(output, label.to(self.device))
            check_test_loss += test_loss.item()
            print(output[0],label[0])


    def save_model(self,dir,task):
        torch.save(self.actor.state_dict(), dir + task + '.pt')
    def load_model(self,dir,task):
        # self.load_model(os.getcwd()+'/IGL_model/', self.args.env)
        self.actor.load_state_dict(torch.load(dir + task + '.pt'))

class BC_mlp(nn.Module):
    def __init__(self,state_dim,act_dim,hidden_dim,activation=nn.ReLU()):
        super(BC_mlp, self).__init__()
        self.hidden_dim = list(hidden_dim)
        self.net = mlp([state_dim] + self.hidden_dim, activation)
        self.mu_layer = nn.Linear(self.hidden_dim[-1], act_dim)

    def forward(self,state):
        output = self.net(state)
        mean   = self.mu_layer(output)
        return mean


def mlp(sizes, activation, output_activation=nn.Tanh()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        # layers += [nn.Linear(sizes[j], sizes[j+1]),nn.BatchNorm1d(sizes[j+1]), act]
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)