import numpy as np
import torch
import torch.optim as optim
from isaacgymenvs.utils.pbrl.mpi import mpi_average
from isaacgymenvs.utils.pbrl.pytorch import optimizer_update, optimizer_cuda
from isaacgymenvs.pbrl.normalizer import Normalizer


class PPOAgent():
    def __init__(self, config, actor, critic, obs_dim, state_dim, init_params):

        self.cfg = config
        self.envs_per_agent = config.envs_per_agent
        self.horizon = config.horizon
        self.actor = actor
        self.critic = critic
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=config.lr)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=config.lr)
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        if self.cfg.is_train:
            self.kl_threshold = init_params['kl_threshold']
            self.entropy_loss_coeff = init_params['entropy_loss_coeff']
            self.actor_variance = init_params['actor_variance']
        if self.cfg.ob_norm:
            self.obs_norm = Normalizer(obs_dim, device=self.cfg.device)
        if self.cfg.asymmetric_ac:
            self.state_norm = Normalizer(state_dim, device=self.cfg.device)
        if self.cfg.value_norm:
            self.value_norm = Normalizer(1, device=self.cfg.device)
        self.max_steps = 8192
        self.epoch_num = 0

    def act(self, obs, state=None, is_deterministic=False):
        if self.cfg.ob_norm:
            self.obs_norm.update(obs)
            obs = self.obs_norm.normalize(obs)
            if self.cfg.asymmetric_ac:
                self.state_norm.update(state)
                state = self.state_norm.normalize(state)

        value = self.critic(obs).squeeze() if not self.cfg.asymmetric_ac else self.critic(state).squeeze() 
        action, prob_dist = self.actor.get_actions(obs, self.actor_variance, is_deterministic)
        return action, prob_dist, value

    def compute_gae(self, rollout, i):
        # Calculate generalized advantage estimate, looping backwards
        gae = torch.zeros(
            (self.horizon, self.envs_per_agent), device=self.cfg.device, dtype=torch.float
        )

        gae_next = 0
        for step in reversed(range(self.horizon)):
            delta = (rollout[('reward', i)][step] + self.cfg.gamma * rollout[('value', i)][step + 1]
                     * (1 - rollout[('reset', i)][step + 1]) - rollout[('value', i)][step])

            gae[step] = (delta + self.cfg.gamma * self.cfg.gae_lambda * gae_next *
                         (1 - rollout[('reset', i)][step + 1]))
            gae_next = gae[step]

        # Value buf is one element longer than gae_buf, we have bootstrapped next_value
        returns = gae + rollout[('value', i)][0: self.horizon]

        # Make a buffer to update nets
        if self.cfg.asymmetric_ac:
            b_states = rollout[('state', i)].reshape(-1, rollout[('state', i)].shape[-1]) 
        else:
            b_states = rollout[('state', i)]

        b_obs = rollout[('obs', i)].reshape(-1, rollout[('obs', i)].shape[-1]) # n_samples * n_obs
        b_actions = rollout[('action', i)].reshape(-1, rollout[('action', i)].shape[-1])
        b_logprobs = rollout[('log_prob', i)].reshape(-1)
        b_values = rollout[('value', i)][0: self.horizon].reshape(-1)
        b_gae = gae.reshape(-1)
        b_returns = returns.reshape(-1)

        if self.cfg.adv_norm:
            b_gae = (b_gae - b_gae.mean()) / (b_gae.std() + 1e-8)
        if self.cfg.value_norm:
            self.value_norm.update(b_values)
            b_values = self.value_norm.normalize(b_values)
            self.value_norm.update(b_returns)
            b_returns = self.value_norm.normalize(b_returns)

        return (b_obs, b_states, b_actions, b_logprobs, b_gae, b_values, b_returns)
    
    def state_dict(self):
        state = {
            # state_dict contains info about model params (learnable tensors: weights&biases)
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            # state_dict contains info about optim state and hyperparams
            'optim_state_dict': self.actor_optim.state_dict()
        }
        state['obs_norm_state_dict'] = self.obs_norm.state_dict() if self.cfg.ob_norm else None
        state['state_norm_state_dict'] = self.state_norm.state_dict() if self.cfg.asymmetric_ac else None
        state['value_norm_state_dict'] = self.value_norm.state_dict() if self.cfg.value_norm else None
        state['actor_variance'] = self.actor_variance
        return state

    def load_state_dict(self, ckpt):
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.actor_optim.load_state_dict(ckpt['optim_state_dict'])
        # required when loading optim state from checkpoint
        optimizer_cuda(self.actor_optim, self.cfg.device)
        
        if self.cfg.ob_norm:
            self.obs_norm.load_state_dict( ckpt['obs_norm_state_dict']) 
        if self.cfg.asymmetric_ac:
            self.state_norm.load_state_dict( ckpt['state_norm_state_dict']) 
        if self.cfg.value_norm:
            self.value_norm.load_state_dict(ckpt['value_norm_state_dict'])
        self.actor_variance =  ckpt['actor_variance']

    def train(self, buffer, i):
        b_obs, b_states, b_actions, b_logprobs, b_gae, b_values, b_returns = buffer
        buffer_size = b_obs.size()[0]
        assert buffer_size >= self.cfg.batch_size
        self.epoch_num += 1

        batch_ids = np.arange(buffer_size)
        for k in range(self.cfg.num_epochs):
            kl_divergences = []
            np.random.shuffle(batch_ids)
            for start in range(0, buffer_size, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                mb_ids = batch_ids[start: end]
                if self.cfg.ob_norm:
                    obs = self.obs_norm.normalize(b_obs[mb_ids])
                    if self.cfg.asymmetric_ac:
                        state = self.state_norm.normalize(b_states[mb_ids])
                else:
                    obs = b_obs[mb_ids]
                    if self.cfg.asymmetric_ac:
                        state = b_states[mb_ids]

                mu, prob_dists = self.actor.get_actions(obs, self.actor_variance)
                log_probs_new = prob_dists.log_prob(b_actions[mb_ids])
                entropy = prob_dists.entropy()
                values = self.critic(obs).squeeze() if not self.cfg.asymmetric_ac else self.critic(state).squeeze() 
                log_prob = b_logprobs[mb_ids]
                values_old = b_values[mb_ids]
                returns = b_returns[mb_ids].detach().clone()
                advantage = b_gae[mb_ids].detach().clone()
                # compute aproximate KL divergence for lr update
                with torch.no_grad():
                    ratio = log_probs_new - log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)
                train_info = self.update_net(log_prob, log_probs_new, values_old, values, returns, advantage, entropy, mu, i)
            
            if not self.cfg.lr_schedule == "fixed":
                self.actor_optim.param_groups[0]["lr"] = self.update_lr(self.actor_optim.param_groups[0]["lr"], kl_divergences)
                self.critic_optim.param_groups[0]["lr"] = self.update_lr(self.critic_optim.param_groups[0]["lr"], kl_divergences)
        return train_info

    def update_net(self, log_prob, log_prob_new, values_old, values, returns, advantage, entropy, mu, i):
        info = {}
        # Calculate loss function for actor network
        r = torch.exp(log_prob_new - log_prob).squeeze()
        # Loss negated as we want SGD
        l1 = advantage * r
        l2 = torch.clamp(r, 1 - self.cfg.epsilon, 1 + self.cfg.epsilon) * advantage
        actor_loss = -torch.min(l1, l2).mean()

        # Critic loss
        critic_loss = self.cfg.value_loss_coeff * 0.5 * torch.pow(values - returns, 2).mean()
        
        # Critic Loss Clipped rl_games
        # value_pred_clipped = values_old + (values - values_old).clamp(-self.cfg.epsilon, self.cfg.epsilon)
        # value_losses = (values - returns)**2
        # value_losses_clipped = (value_pred_clipped - returns)**2
        # c_loss = torch.max(value_losses, value_losses_clipped)
        # critic_loss = self.cfg.value_loss_coeff * 0.5 * c_loss.mean()

        # b_loss = self.bound_loss(mu)
        # b_loss = b_loss.mean() * 0.0001
        
        entropy_loss = self.entropy_loss_coeff * entropy.mean()
        actor_loss -= entropy_loss

        tot_loss = actor_loss + critic_loss 

        grad_norm = optimizer_update(self.actor_optim, actor_loss)
        grad_norm = optimizer_update(self.critic_optim, critic_loss)

        info[('Loss/total', i)] = tot_loss.cpu().item()
        info[('Loss/Actor', i)] = actor_loss.cpu().item()
        info[('Loss/Critic', i)] = critic_loss.cpu().item()

        return mpi_average(info)

    def update_lr(self, current_lr, kl_divergences):
        lr = current_lr
        if self.cfg.lr_schedule == "linear":
            mul = max(0, self.max_steps - self.epoch_num)/self.max_steps 
            lr = self.min_lr + (0.003 - self.min_lr) * mul
        else:   
            kl = sum(kl_divergences) / len(kl_divergences)
            if kl > (2.0 * self.kl_threshold):
                lr = max(current_lr / 1.5, self.min_lr)
            if kl < (0.5 * self.kl_threshold):
                lr = min(current_lr * 1.5, self.max_lr)
        return lr

    # def bound_loss(self, mu):
    #     soft_bound = 1.1
    #     mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
    #     mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
    #     b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
    #     return b_loss   
