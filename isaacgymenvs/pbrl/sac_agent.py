import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.nn.utils import clip_grad_norm_
from isaacgymenvs.utils.pbrl.pytorch import optimizer_update, soft_update_target_network, optimizer_cuda
from isaacgymenvs.pbrl.normalizer import Normalizer
from isaacgymenvs.pbrl.buffer import NStepReplay, ReplayBuffer
from copy import deepcopy


class SACAgent():
    def __init__(self, config, actor, critic, obs_dim, ac_dim, init_params):
        self.cfg = config
        self.actor = actor
        self.critic = critic
        self.critic_target = deepcopy(self.critic)
        self.actor_target = self.actor
        if self.cfg.is_train:
            self.actor_lr = init_params['actor_lr']
            self.critic_lr = init_params['critic_lr']
            self.target_entropy = init_params['target_entropy']
            self.actor_optim = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optim = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)
            self.log_alpha = nn.Parameter(torch.zeros(1, device=self.cfg.device))
            self.alpha_optim = optim.AdamW([self.log_alpha], lr=0.005)
            self.n_step_buffer = NStepReplay(
                obs_dim, ac_dim, self.cfg.envs_per_agent, self.cfg.nstep, self.cfg.device
            )
        
        if self.cfg.ob_norm:
            self.obs_norm = Normalizer(obs_dim, device=self.cfg.device)

        self.memory = ReplayBuffer(self.cfg.memory_size, obs_dim, ac_dim, self.cfg.device)

    def act(self, obs, is_deterministic=False):
        if self.cfg.ob_norm:
            self.obs_norm.update(obs)
            obs = self.obs_norm.normalize(obs)

        actions, _ = self.actor.get_actions(obs, is_deterministic)
        return actions

    def store_episode(self, rollout, i):
        obs = rollout[('obs', i)].swapaxes(0, 1)
        actions = rollout[('action', i)].swapaxes(0, 1)
        rewards = rollout[('reward', i)].swapaxes(0, 1).unsqueeze(2)
        next_obs = rollout[('next_obs', i)].swapaxes(0, 1)
        resets = rollout[('reset', i)][1:].swapaxes(0, 1).unsqueeze(2)
        trajectory = self.n_step_buffer.add_to_buffer(obs, actions, rewards, next_obs, resets)
        self.memory.add_to_buffer(trajectory)

    def state_dict(self):
        state = {
            # state_dict contains info about model params (learnable tensors: weights&biases)
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            # state_dict contains info about optim state and hyperparams
            'optim_state_dict': self.actor_optim.state_dict()
        }
        state['obs_norm_state_dict'] = self.obs_norm.state_dict() if self.cfg.ob_norm else None
        return state
    
    def load_state_dict(self, ckpt):
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        # required when loading optim state from checkpoint
        # optimizer_cuda(self.actor_optim, self.cfg.device)
        
        if self.cfg.ob_norm:
            self.obs_norm.load_state_dict( ckpt['obs_norm_state_dict']) 

    def train(self, i):
        critic_loss_list = list()
        actor_loss_list = list()
        for k in range(self.cfg.num_epochs):
            obs, action, reward, next_obs, done = self.memory.sample_batch(self.cfg.batch_size)
            if self.cfg.ob_norm:
                obs = self.obs_norm.normalize(obs)
                next_obs = self.obs_norm.normalize(next_obs)
            critic_loss = self.update_critic(obs, action, reward, next_obs, done)
            critic_loss_list.append(critic_loss)

            actor_loss = self.update_actor(obs)
            actor_loss_list.append(actor_loss)

            soft_update_target_network(self.critic_target, self.critic, 0.05)

        log_info = {
            ('Loss/actor', i): np.mean(actor_loss_list),
            ('Loss/critic', i): np.mean(critic_loss_list),
        }

        return log_info

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_actions, dist = self.actor.get_actions(next_obs)
            log_prob = dist.log_prob(next_actions).sum(-1, keepdim=True)
            alpha = self.log_alpha.exp().detach()
            target_Q = self.critic_target.get_q_min(next_obs, next_actions) - alpha * log_prob
            target_Q = reward + (1 - done) * (self.cfg.gamma ** self.cfg.nstep) * target_Q
        current_Q1, current_Q2 = self.critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        grad_norm = optimizer_update(self.critic_optim, critic_loss)

        return critic_loss.item()

    def update_actor(self, obs):
        self.critic.requires_grad_(False)
        actions, dist = self.actor.get_actions(obs)
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        Q = self.critic.get_q_min(obs, actions)
        actor_loss = (self.log_alpha.exp().detach() * log_prob - Q).mean()
        grad_norm = optimizer_update(self.actor_optim, actor_loss)
        self.critic.requires_grad_(True)

        alpha_loss = (self.log_alpha.exp() * (-log_prob - self.target_entropy).detach()).mean()
        optimizer_update(self.alpha_optim, alpha_loss)

        return actor_loss.item()
