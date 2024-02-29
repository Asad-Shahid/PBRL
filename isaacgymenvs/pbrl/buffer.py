import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device='cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.next_p = 0  # next pointer
        self.if_full = False
        self.cur_capacity = 0  # current capacity
        self.capacity = int(capacity)

        ret = create_buffer(capacity=self.capacity, obs_dim=obs_dim, action_dim=action_dim, device=device)
        self.buf_obs, self.buf_action, self.buf_next_obs, self.buf_reward, self.buf_done = ret

    @torch.no_grad()
    def add_to_buffer(self, trajectory):
        obs, actions, rewards, next_obs, dones = trajectory
        obs = obs.reshape(-1, self.obs_dim)
        actions = actions.reshape(-1, self.action_dim)
        rewards = rewards.reshape(-1, 1)
        next_obs = next_obs.reshape(-1, self.obs_dim)
        dones = dones.reshape(-1, 1).bool()
        p = self.next_p + rewards.shape[0]

        if p > self.capacity:
            self.if_full = True

            self.buf_obs[self.next_p:self.capacity] = obs[:self.capacity - self.next_p]
            self.buf_action[self.next_p:self.capacity] = actions[:self.capacity - self.next_p]
            self.buf_reward[self.next_p:self.capacity] = rewards[:self.capacity - self.next_p]
            self.buf_next_obs[self.next_p:self.capacity] = next_obs[:self.capacity - self.next_p]
            self.buf_done[self.next_p:self.capacity] = dones[:self.capacity - self.next_p]

            p = p - self.capacity
            self.buf_obs[0:p] = obs[-p:]
            self.buf_action[0:p] = actions[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            self.buf_next_obs[0:p] = next_obs[-p:]
            self.buf_done[0:p] = dones[-p:]
        else:
            self.buf_obs[self.next_p:p] = obs
            self.buf_action[self.next_p:p] = actions
            self.buf_reward[self.next_p:p] = rewards
            self.buf_next_obs[self.next_p:p] = next_obs
            self.buf_done[self.next_p:p] = dones

        self.next_p = p  # update pointer
        self.cur_capacity = self.capacity if self.if_full else self.next_p

    @torch.no_grad()
    def sample_batch(self, batch_size, device='cuda'):
        indices = torch.randint(self.cur_capacity, size=(batch_size,), device=device)
        return (
            self.buf_obs[indices].to(device),
            self.buf_action[indices].to(device),
            self.buf_reward[indices].to(device),
            self.buf_next_obs[indices].to(device),
            self.buf_done[indices].to(device).float()
        )


class NStepReplay:
    def __init__(self, obs_dim, action_dim, num_envs, nstep=3, device='cuda', gamma=0.99):
        self.num_envs = num_envs
        self.nstep = nstep
        buf = create_buffer((self.num_envs, self.nstep), obs_dim, action_dim)
        self.nstep_buf_obs, self.nstep_buf_action, self.nstep_buf_next_obs, \
            self.nstep_buf_reward, self.nstep_buf_done = buf
        self.nstep_count = 0
        self.gamma = gamma
        self.gamma_array = torch.tensor([self.gamma ** i for i in range(self.nstep)]).to(device).view(-1, 1)

    @torch.no_grad()
    def add_to_buffer(self, obs, actions, rewards, next_obs, dones):
        if self.nstep > 1:
            obs_list, action_list, reward_list, next_obs_list, done_list = list(), list(), list(), list(), list()
            for i in range(obs.shape[1]):
                self.nstep_buf_obs = self.fifo_shift(self.nstep_buf_obs, obs[:, i])
                self.nstep_buf_next_obs = self.fifo_shift(self.nstep_buf_next_obs, next_obs[:, i])
                self.nstep_buf_done = self.fifo_shift(self.nstep_buf_done, dones[:, i])
                self.nstep_buf_action = self.fifo_shift(self.nstep_buf_action, actions[:, i])
                self.nstep_buf_reward = self.fifo_shift(self.nstep_buf_reward, rewards[:, i])
                self.nstep_count += 1
                if self.nstep_count < self.nstep:
                    continue

                obs_list.append(self.nstep_buf_obs[:, 0])
                action_list.append(self.nstep_buf_action[:, 0])
                reward, next_ob, done = compute_nstep_return(nstep_buf_next_obs=self.nstep_buf_next_obs,
                                                             nstep_buf_done=self.nstep_buf_done,
                                                             nstep_buf_reward=self.nstep_buf_reward,
                                                             gamma_array=self.gamma_array)
                reward_list.append(reward)
                next_obs_list.append(next_ob)
                done_list.append(done)

            return torch.cat(obs_list), torch.cat(action_list), torch.cat(reward_list), torch.cat(next_obs_list), torch.cat(done_list)
        else:
            return obs, actions, rewards, next_obs, dones

    def fifo_shift(self, queue, new_tensor):
        queue = torch.cat((queue[:, 1:], new_tensor.unsqueeze(1)), dim=1)
        return queue


def create_buffer(capacity, obs_dim, action_dim, device='cuda'):
    if isinstance(capacity, int):
        capacity = (capacity,)
    buf_obs = torch.empty(
        (*capacity, obs_dim), dtype=torch.float32, device=device
    )
    buf_action = torch.empty(
        (*capacity, action_dim), dtype=torch.float32, device=device
    )
    buf_reward = torch.empty(
        (*capacity, 1), dtype=torch.float32, device=device
    )
    buf_next_obs = torch.empty(
        (*capacity, obs_dim), dtype=torch.float32, device=device
    )
    buf_done = torch.empty(
        (*capacity, 1), dtype=torch.bool, device=device
    )
    return buf_obs, buf_action, buf_next_obs, buf_reward, buf_done


@torch.jit.script
def compute_nstep_return(nstep_buf_next_obs, nstep_buf_done, nstep_buf_reward, gamma_array):
    buf_done = nstep_buf_done.squeeze(-1)
    buf_done_ids = torch.where(buf_done)
    buf_done_envs = torch.unique_consecutive(buf_done_ids[0])
    buf_done_steps = buf_done.argmax(dim=1)

    done = nstep_buf_done[:, -1].clone()
    done[buf_done_envs] = True

    next_obs = nstep_buf_next_obs[:, -1].clone()
    next_obs[buf_done_envs] = nstep_buf_next_obs[buf_done_envs, buf_done_steps[buf_done_envs]].clone()

    mask = torch.ones(buf_done.shape, device=buf_done.device, dtype=torch.bool)
    mask[buf_done_envs] = torch.arange(mask.shape[1],
                                       device=buf_done.device) <= buf_done_steps[buf_done_envs][:, None]
    discounted_rewards = nstep_buf_reward * gamma_array
    discounted_rewards = (discounted_rewards * mask.unsqueeze(-1)).sum(1)
    return discounted_rewards, next_obs, done
