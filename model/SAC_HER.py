import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam
import copy
import numpy as np
from Utils.Buffer import her_buffer
from Utils.her import her_sampler


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class SAC_HER(object):
    def __init__(self, env_params, env, args):
        self.env = env
        self.env_params = env_params
        self.state_dim  = env_params['obs'] + env_params['goal']
        self.action_dim = env_params['action']
        self.clip_obs   = 200.

        self.buffer_size = args.max_timesteps
        self.batch_size  = args.batch_size

        self.gamma = args.discount
        self.tau = args.tau
        self.alpha = args.alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.automatic_entropy_tuning = args.automatic_entropy_tuning


        self.critic           = Double_Q(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target    = copy.deepcopy(self.critic)

        self.her_module = her_sampler(args.replay_strategy, args.replay_k, self.env.compute_reward)
        self.buffer = her_buffer(env_params, self.buffer_size,self.her_module.sample_her_transitions)

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.Tensor([self.action_dim]).to(self.device).item()
            self.log_alpha = np.log([args.alpha]).astype(np.float32)
            self.log_alpha = torch.tensor(self.log_alpha, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=3e-4)

        self.policy = Squashed_Gaussian_Actor(self.state_dim, self.action_dim,args.hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=3e-4)

    def _preproc_inputs(self,obs,g,train=False):
        if train:
            inputs = np.concatenate([obs, g],axis=1)
        else:
            inputs = np.concatenate([obs, g])
        return inputs

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _ = self.policy(state)
        else:
            action, _ = self.policy(state,Eval=True)
        return action.detach().cpu().numpy().flatten()

    def update_parameters(self, batch_size=256):
        transitions = self.buffer.sample(batch_size)

        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        state_batch      = self._preproc_inputs(transitions['obs'],transitions['g'],train=True)
        next_state_batch = self._preproc_inputs(transitions['obs_next'],transitions['g_next'],train=True)

        state_batch      = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch     = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
        reward_batch     = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)


        self.critic_optimizer.zero_grad()
        critic_loss, Q1, Q2 = self.compute_loss_q(state_batch, action_batch, next_state_batch, reward_batch)
        # Optimize the critic
        critic_loss.backward()
        self.critic_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = False

        self.policy_optim.zero_grad()
        policy_loss = self.compute_loss_pi(state_batch, action_batch, next_state_batch, reward_batch)
        policy_loss.backward()
        self.policy_optim.step()

        for p in self.critic.parameters():
            p.requires_grad = True


        if self.automatic_entropy_tuning:
            self.alpha_optim.zero_grad()
            alpha_loss = self.compute_loss_alpha(state_batch, action_batch, next_state_batch, reward_batch)
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            pass

        with torch.no_grad():
            soft_update(self.critic_target, self.critic, self.tau)

        # return Q1.mean().item(), Q2.mean().item(), critic_loss.item(), policy_loss.item(), alpha_loss.item()

    def compute_loss_q(self,state_batch, action_batch, next_state_batch, reward_batch):

        Q1, Q2 = self.critic(state_batch, action_batch)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action, next_log_pi = self.policy(next_state_batch)
            # Compute critic loss
            target_Q1, target_Q2 = self.critic_target(next_state_batch,next_action)
            minq = torch.min(target_Q1, target_Q2)
            target_y = reward_batch + self.gamma*(minq-self.alpha*next_log_pi.reshape(-1,1))

        critic_loss = F.mse_loss(Q1,target_y) + F.mse_loss(Q2,target_y)

        return critic_loss, Q1, Q2
    def compute_loss_pi(self,state_batch, action_batch, next_state_batch, reward_batch):
        pi, log_pi = self.policy(state_batch)
        q1, q2 = self.critic(state_batch, pi)
        min_q = torch.minimum(q1, q2)

        policy_loss = (self.alpha * log_pi - min_q).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        return policy_loss

    def compute_loss_alpha(self,state_batch, action_batch, next_state_batch, reward_batch):
        pi, log_pi = self.policy(state_batch)
        alpha_loss = (self.log_alpha * -(log_pi + self.target_entropy).detach()).mean()
        return alpha_loss


def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)


class Squashed_Gaussian_Actor(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_dim,activation=nn.ReLU(),log_std_min=-20, log_std_max=2):
        super(Squashed_Gaussian_Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden_dim = list(hidden_dim)
        self.net = mlp([obs_dim] + self.hidden_dim, activation)
        self.mu_layer = nn.Linear(self.hidden_dim[-1], act_dim)
        self.log_std_layer = nn.Linear(self.hidden_dim[-1], act_dim)

    def forward(self,state,Eval=False):
        output = self.net(state)
        mean   = self.mu_layer(output)
        log_std = self.log_std_layer(output)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        pi_distribution = Normal(mean, std)

        if Eval:
            log_pi = pi_distribution.log_prob(mean).sum(axis=-1)
            log_pi -= (2*(np.log(2) - mean - F.softplus(-2*mean))).sum(axis=1)
            tanh_mean = torch.tanh(mean)

            return tanh_mean, log_pi
        else:
            sample_action = pi_distribution.rsample()
            log_pi = pi_distribution.log_prob(sample_action).sum(axis=-1)
            log_pi -= (2*(np.log(2) - sample_action - F.softplus(-2*sample_action))).sum(axis=1)
            tanh_sample = torch.tanh(sample_action)

            return tanh_sample, log_pi


class Double_Q(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, activation=nn.ReLU()):
        super().__init__()
        self.hidden_dim = list(hidden_dim)
        self.q1 = mlp([obs_dim + act_dim] + self.hidden_dim + [1], activation)
        self.q2 = mlp([obs_dim + act_dim] + self.hidden_dim + [1], activation)

    def forward(self, obs, act):
        q1 = self.q1(torch.cat([obs, act], dim=-1))
        q2 = self.q2(torch.cat([obs, act], dim=-1))
        return q1, q2

    def Q1(self,obs, act):
        q1 = self.q1(torch.cat([obs, act], dim=-1))
        return q1