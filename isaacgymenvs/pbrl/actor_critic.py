import torch
import torch.nn as nn
import numpy as np
from isaacgymenvs.utils.pbrl.pytorch import SquashedNormal
from torch.distributions import Independent, Normal


class Actor(nn.Module):
    def __init__(self, n_obs, n_actions, layers_array, activation):
        super(Actor, self).__init__()

        activations = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}

        # Outputs mean value of action
        self.actor = create_net(n_obs, n_actions, layers_array, activations[activation])

    def forward(self, x):
        return self.actor(x)

    def get_actions(self, x, variance, is_deterministic=False):
        mean = self.actor(x)
        # logstd = nn.Parameter(torch.full((mean.shape[-1],), variance))
        # logstd = logstd.expand_as(mean)
        # std = torch.exp(logstd)
        # dist = Independent(Normal(loc=mean, scale=std), 1)
        # actions = dist.rsample()
        covariance_matrix = torch.diag(torch.full((mean.shape[-1],), variance))
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix)
        if is_deterministic:
            actions = mean
        else:
            actions = dist.rsample()
        return actions, dist


class Critic(nn.Module):
    def __init__(self, n_obs, layers_array, activation):
        super(Critic, self).__init__()

        activations = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}

        # Outputs value function
        self.critic = create_net(n_obs, 1, layers_array, activations[activation])

    def forward(self, x):
        return self.critic(x)


class SACPolicy(Actor):
    def __init__(self, n_obs, n_actions, layers_array, activation):
        super().__init__(n_obs, n_actions, layers_array, activation)

        self.log_std_min = -5
        self.log_std_max = 5

    def forward(self, x):
        return self.get_actions(x)

    def get_actions(self, state, is_deterministic=False):
        mu, log_std = self.actor(state).chunk(2, dim=-1)
        std = log_std.clamp(self.log_std_min, self.log_std_max).exp()
        dist = SquashedNormal(mu, std)
        if is_deterministic:
            actions = mu
        else:
            actions = dist.rsample()
        return actions, dist


class DoubleQCritic(nn.Module):
    def __init__(self, state_dim, act_dim, layers_array, activation):
        super().__init__()
        activations = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}

        self.net_q1 = create_net(state_dim + act_dim, 1, layers_array, activations[activation])
        self.net_q2 = create_net(state_dim + act_dim, 1, layers_array, activations[activation])

    def get_q_min(self, state, action):
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state, action):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x), self.net_q2(input_x)  # two Q values

    def get_q1(self, state, action):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x)


def create_net(input_size, output_size, layers_array, activation):
    layers = []
    layers.append(nn.Linear(input_size, layers_array[0]))
    layers.append(activation)
    for i in range(len(layers_array) - 1):
        layers.append(nn.Linear(layers_array[i], layers_array[i + 1]))
        layers.append(activation)
    layers.append(nn.Linear(layers_array[-1], output_size))
    return nn.Sequential(*layers)
