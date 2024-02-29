import numpy as np
import torch


class RolloutRunner():
    """
    Run rollout given environments and agents.
    """

    def __init__(self, config, envs, agents):
        """
        Args:
            TBD
        """
        self.cfg = config
        self.num_agents = config.num_agents
        self.envs_per_agent = config.envs_per_agent
        self.rollout_steps = config.horizon
        self.device = config.device
        self.num_actions = envs.action_space.shape[0]
        self.num_obs = envs.observation_space.shape[0]
        self.num_states = envs.state_space.shape[0]
        self.envs = envs
        self.agents = agents

        # Buffer needed for keeping track of episodic rewards
        self.reward_sum = torch.zeros(
            (self.num_agents, self.envs_per_agent), device=self.device, dtype=torch.float
        )
        self.next_reset = torch.zeros(
            (self.num_agents, self.envs_per_agent), device=self.device
        )

    def init_buffer(self, rollout_len):
        obs = torch.zeros(
            (rollout_len, self.envs_per_agent, self.num_obs), device=self.device, dtype=torch.float,
        )
        states = torch.zeros(
            (rollout_len, self.envs_per_agent, self.num_states), device=self.device, dtype=torch.float,
        )
        actions = torch.zeros(
            (rollout_len, self.envs_per_agent, self.num_actions), device=self.device, dtype=torch.float,
        )
        rewards = torch.zeros(
            (rollout_len, self.envs_per_agent), device=self.device, dtype=torch.float
        )
        next_obs = torch.zeros(
            (rollout_len, self.envs_per_agent, self.num_obs), device=self.device, dtype=torch.float
        )
        next_states = torch.zeros(
            (rollout_len, self.envs_per_agent, self.num_states), device=self.device, dtype=torch.float,
        )
        resets = torch.ones(
            (rollout_len + 1, self.envs_per_agent), device=self.device, dtype=torch.long,
        )
        # We need rollout_step+1 values for advantage estimation
        values = torch.zeros(
            (rollout_len + 1, self.envs_per_agent), device=self.device, dtype=torch.float,
        )
        log_probs = torch.zeros(
            (rollout_len, self.envs_per_agent), device=self.device, dtype=torch.float
        )

        self.traj = {}

        # All of the buffers for all the workers
        for i in range(self.num_agents):
            self.traj.update({
                ('obs', i): obs,
                ('state', i): states, # dummy zero tensors for symmetric input
                ('action', i): actions,
                ('reward', i): rewards,
                ('next_obs', i): next_obs,
                ('next_state', i): next_states,
                ('reset', i): resets,
                ('value', i): values,
                ('log_prob', i): log_probs,
            })

    def reset_envs(self):
        # Get first observation and divide per agent
        obs_dict = self.envs.reset()
        self.obs = obs_dict["obs"].reshape(self.num_agents, self.envs_per_agent, -1)
        if self.num_states > 0:
            self.state = obs_dict["states"].reshape(self.num_agents, self.envs_per_agent, -1)

    def run_episode(self, rollout_len, is_deterministic=False, random=False):
        """
        Runs one episode and returns the rollout.
        """

        # Initialize buffer dict for a trajectory
        self.init_buffer(rollout_len)
        state = torch.zeros(
            (self.envs_per_agent, self.num_states), device=self.device, dtype=torch.float
        )

        obs = self.obs
        if self.num_states > 0:
            state = self.state
        next_reset = self.next_reset

        episode_reward_sum = torch.zeros(
            (self.num_agents), device=self.device, dtype=torch.float
        )
        finished_episodes = torch.zeros(
            (self.num_agents), device=self.device, dtype=torch.int
        )
        actions = torch.zeros(
            (self.num_agents, self.envs_per_agent, self.num_actions), device=self.device, dtype=torch.float
        )

        # Collect rollout
        for step in range(rollout_len):

            for i in range(self.num_agents):
                self.traj[('obs', i)][step] = obs[i]
                self.traj[('state', i)][step] = state[i]
                self.traj[('reset', i)][step] = next_reset[i]

                # Only inference, no grad needed
                with torch.no_grad():
                    if self.cfg.algo == 'ppo':
                        actions[i], prob_dist, value = self.agents[i].act(obs[i], state[i], is_deterministic)
                        self.traj[('value', i)][step] = value
                        self.traj[('log_prob', i)][step] = prob_dist.log_prob(actions[i])
                    else:
                        if random:
                            actions[i] = torch.rand((self.cfg.envs_per_agent, self.num_actions)) * 2.0 - 1.0
                        else:
                            actions[i] = self.agents[i].act(obs[i], is_deterministic)

            # Take environment step
            next_obs_dict, reward, next_reset, extras = self.envs.step(actions.view(-1, self.num_actions))
            next_obs = next_obs_dict["obs"].reshape(self.num_agents, self.envs_per_agent, -1)
            reward = reward.reshape(self.num_agents, self.envs_per_agent)
            next_reset = next_reset.reshape(self.num_agents, self.envs_per_agent)
            extras['time_outs'] = extras['time_outs'].reshape(self.num_agents, self.envs_per_agent)
            if "successes" in extras.keys(): # only required for factory tasks
                extras['successes'] = extras['successes'].reshape(self.num_agents, self.envs_per_agent)
            obs = next_obs
            if self.num_states > 0:
                next_state = next_obs_dict["states"].reshape(self.num_agents, self.envs_per_agent, -1)
                state = next_state

            for i in range(self.num_agents):
                self.traj[('action', i)][step] = actions[i]
                self.traj[('next_obs', i)][step] = next_obs[i]
                if self.num_states > 0:
                    self.traj[('next_state', i)][step] = next_state[i]
                if self.cfg.value_bootstrap:
                    self.traj[('reward', i)][step] = reward[i] + \
                        self.cfg.gamma * self.traj[('value', i)][step] * extras['time_outs'][i].float()
                else:
                    self.traj[('reward', i)][step] = reward[i] 

                # Calculate mean episode reward
                self.reward_sum[i] += reward[i]
                r_finished = torch.masked_select(self.reward_sum[i], next_reset[i].bool())

                finished_episodes[i] += len(r_finished)

                if len(r_finished) > 0:
                    episode_reward_sum[i] += r_finished.sum()
                self.reward_sum[i] *= torch.logical_not(next_reset[i])

        # last steps for the next episode
        self.obs = obs
        if self.num_states > 0:
            self.state = state 
        self.next_reset = next_reset
        ep_info = {}
        ep_rew = np.empty([self.num_agents])

        for i in range(self.num_agents):
            # Only required for last step
            self.traj[('reset', i)][-1] = next_reset[i]
            if self.cfg.algo == 'ppo': 
                with torch.no_grad():
                    _, _, last_value = self.agents[i].act(obs[i], state[i], is_deterministic)
                    self.traj[('value', i)][-1] = last_value
            
            # Calculate reward
            if finished_episodes[i] != 0:
                mean_ep_reward = episode_reward_sum[i] / finished_episodes[i]
                ep_rew[i] = mean_ep_reward.detach().cpu()
            else:
                ep_rew[i] = 0
            ep_info.update({('ep_rew', (i)): ep_rew[i]})

            # Log params for PBRL during training
            if self.cfg.algo == 'ppo' and self.cfg.is_train:
                ep_info.update({('learning_rate', (i)): self.agents[i].actor_optim.param_groups[0]["lr"]})
                ep_info.update({('kl_threshold', (i)): self.agents[i].kl_threshold})
                ep_info.update({('entropy_loss_coefficient', (i)): self.agents[i].entropy_loss_coeff})
                ep_info.update({('actor_variance', (i)): self.agents[i].actor_variance})

            if self.cfg.algo == 'ddpg' and self.cfg.is_train:
                ep_info.update({('actor_lr', (i)): self.agents[i].actor_optim.param_groups[0]["lr"]})
                ep_info.update({('critic_lr', (i)): self.agents[i].critic_optim.param_groups[0]["lr"]})
                ep_info.update({('std_min', (i)): self.agents[i].std_min})
                ep_info.update({('std_max', (i)): self.agents[i].std_max})

            if self.cfg.algo == 'sac' and self.cfg.is_train:
                ep_info.update({('actor_lr', (i)): self.agents[i].actor_optim.param_groups[0]["lr"]})
                ep_info.update({('critic_lr', (i)): self.agents[i].critic_optim.param_groups[0]["lr"]})
                ep_info.update({('target_entropy', (i)): self.agents[i].target_entropy})
                
            # required for factory or industreal tasks
            if "sdf_reward" in extras.keys():
                ep_info.update({('insertion_successes', (i)): 0})
                ep_info.update({('curr_max_disp', (i)): 0})
                ep_info.update({('sdf_reward', (i)): extras["sdf_reward"].cpu().numpy()})
                ep_info.update({('sapu_adjusted_reward', (i)): extras["sapu_adjusted_reward"].cpu().numpy()})
            if "insertion_successes" in extras.keys():
                ep_info.update({('insertion_successes', (i)): extras["insertion_successes"].cpu().numpy()})
                ep_info.update({('curr_max_disp', (i)): extras["curr_max_disp"]})
            if "successes" in extras.keys():
                ep_info.update({('successes', (i)): extras["successes"][i].mean().cpu().numpy()})
        return self.traj, ep_info, ep_rew
