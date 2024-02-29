import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from isaacgymenvs.utils.pbrl.pytorch import optimizer_update, soft_update_target_network, optimizer_cuda
from isaacgymenvs.utils.pbrl.pytorch import add_mixed_normal_noise, add_normal_noise
from isaacgymenvs.pbrl.normalizer import Normalizer
from isaacgymenvs.pbrl.buffer import NStepReplay, ReplayBuffer
from copy import deepcopy


class DDPGAgent():
    def __init__(self, config, actor, critic, obs_dim, ac_dim, init_params):
        self.cfg = config
        self.actor = actor
        self.critic = critic
        self.critic_target = deepcopy(self.critic)
        self.actor_target = self.actor
        if self.cfg.is_train:
            self.actor_lr = init_params['actor_lr']
            self.critic_lr = init_params['critic_lr']
            self.std_min = init_params['std_min']
            self.std_max = init_params['std_max']
            self.actor_optim = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optim = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)
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

        # deterministic actions
        actions = self.actor(obs).tanh()
        # add noise to the actions
        if not is_deterministic:
            actions = add_mixed_normal_noise(actions,
                                            std_min=self.std_min,
                                            std_max=self.std_max,
                                            out_bounds=[-1., 1.])
        return actions

    def store_episode(self, rollout, i):
        state = rollout[('obs', i)].swapaxes(0, 1)
        actions = rollout[('action', i)].swapaxes(0, 1)
        rewards = rollout[('reward', i)].swapaxes(0, 1).unsqueeze(2)
        next_state = rollout[('next_obs', i)].swapaxes(0, 1)
        resets = rollout[('reset', i)][1:].swapaxes(0, 1).unsqueeze(2)
        trajectory = self.n_step_buffer.add_to_buffer(state, actions, rewards, next_state, resets)
        self.memory.add_to_buffer(trajectory)

    @torch.no_grad()
    def get_tgt_policy_actions(self, obs):
        actions = self.actor_target(obs).tanh()
        actions = add_normal_noise(actions,
                                   std=0.8,
                                   noise_bounds=[-0.2, 0.2],
                                   out_bounds=[-1., 1.])
        return actions

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
        # self.actor_optim.load_state_dict(ckpt['optim_state_dict'])
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
            next_actions = self.get_tgt_policy_actions(next_obs)
            target_Q = self.critic_target.get_q_min(next_obs, next_actions)
            target_Q = reward + (1 - done) * (self.cfg.gamma ** self.cfg.nstep) * target_Q

        current_Q1, current_Q2 = self.critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        grad_norm = optimizer_update(self.critic_optim, critic_loss)

        return critic_loss.item()

    def update_actor(self, obs):
        self.critic.requires_grad_(False)
        action = self.actor(obs).tanh()
        Q = self.critic.get_q_min(obs, action)
        actor_loss = -Q.mean()
        grad_norm = optimizer_update(self.actor_optim, actor_loss)
        self.critic.requires_grad_(True)

        return actor_loss.item()
