import os
import torch
import wandb
import math
import numpy as np
from tqdm import tqdm, trange
from time import time
from copy import deepcopy
from collections import defaultdict, OrderedDict
from isaacgymenvs.pbrl.actor_critic import Actor, Critic, SACPolicy, DoubleQCritic
from isaacgymenvs.pbrl.ppo_agent import PPOAgent
from isaacgymenvs.pbrl.sac_agent import SACAgent
from isaacgymenvs.pbrl.ddpg_agent import DDPGAgent
from isaacgymenvs.pbrl.rollout import RolloutRunner
from isaacgymenvs.utils.pbrl.logger import logger
from isaacgymenvs.utils.pbrl.pytorch import get_ckpt_path, soft_update_target_network
from isaacgymenvs.utils.pbrl.mpi import mpi_sum

class Trainer():
    """
    Trainer class for PBRL in PyTorch.
    """

    def __init__(self, config, envs, params_to_tune):
        self.cfg = config
        num_agents = config.num_agents
        obs_dim = envs.observation_space.shape[0]
        state_dim = envs.state_space.shape[0]
        ac_dim = envs.action_space.shape[0]
        hid_size = config.hid_size
        activation = config.activation
        self.cfg.off_policy = config.algo != 'ppo'
        self.cfg.asymmetric_ac = state_dim > 0

        self.agents = np.array([])
        init_params = {}

        for i in range(num_agents):
            for param, value in params_to_tune.items():
                assert len(value) > 1
                if len(value) == 2:
                    val = np.random.uniform(value[0], value[1])
                else:
                    idx = np.random.randint(0, len(value))
                    val = value[idx]
                init_params[param] = val

            if self.cfg.off_policy:
                critic = DoubleQCritic(obs_dim, ac_dim, hid_size, activation).to(config.device)
                if config.algo == 'sac':
                    actor = SACPolicy(obs_dim, ac_dim*2, hid_size, activation).to(config.device)
                    self.agents = np.append(self.agents,
                                            [SACAgent(config, actor, critic, obs_dim, ac_dim, init_params)])
                else:
                    actor = Actor(obs_dim, ac_dim, hid_size, activation).to(config.device)
                    self.agents = np.append(self.agents,
                                            [DDPGAgent(config, actor, critic, obs_dim, ac_dim, init_params)])
            else:
                actor = Actor(obs_dim, ac_dim, hid_size, activation).to(config.device)
                critic = Critic(obs_dim, hid_size, activation).to(config.device)
                if self.cfg.asymmetric_ac:
                    critic = Critic(state_dim, hid_size, activation).to(config.device)
                self.agents = np.append(self.agents,
                                        [PPOAgent(config, actor, critic, obs_dim, state_dim, init_params)])

        self.runner = RolloutRunner(config, envs, self.agents)

        if self.cfg.is_train:
            exclude = ['device']
            if not self.cfg.wandb:
                os.environ['WANDB_MODE'] = 'dryrun'
                mode = "disabled"
            else:
                mode = "online"

        # Weights and Biases (wandb) is used for logging, set the account details below or dry run above
            # user or team name
            entity = self.cfg.wandb_entity
            # project name
            project = self.cfg.wandb_project

            wandb.init(
                mode=mode,
                group=self.cfg.algo,
                resume=self.cfg.run_name,
                project=project,
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=self.cfg.log_dir,
                entity=entity,
                notes=self.cfg.notes
            )

    def train(self, params_to_tune, mutation_coeffs):
        """ Trains one agent or the population of agents. """
        num_agents = self.cfg.num_agents

        # load checkpoint
        step, update_iter = self._load_ckpt()

        logger.info("Start training at step=%d", step)

        pbar = tqdm(initial=step, total=self.cfg.max_global_step, desc=self.cfg.run_name)
        ep_info = defaultdict(list)

        st_time = time()

        st_step = step
        log_step = 0
        returns_aux = []
        evo_iter = self.cfg.evo_interval // (self.cfg.envs_per_agent * self.cfg.horizon) + 1
        self.runner.reset_envs()

        # warm start
        if self.cfg.off_policy:
            rollout, info, ep_rew = self.runner.run_episode(rollout_len=32, random=True)
            for i in range(num_agents):
                self.agents[i].store_episode(rollout, i)

        while step < self.cfg.max_global_step:
            # collect rollouts
            run_step = 0

            rollout, info, ep_rew = self.runner.run_episode(self.cfg.horizon)
            returns_aux.append(ep_rew)
            run_step += self.cfg.horizon * self.cfg.envs_per_agent
            log_step += self.cfg.horizon * self.cfg.envs_per_agent

            for k, v in info.items():
                if isinstance(v, list):
                    ep_info[k].extend(v)
                else:
                    # if v != 0: # for off-policy methods: horizon is 1
                    ep_info[k].append(v)

            self._log_ep(log_step, ep_info)
            ep_info = defaultdict(list)
            if self.cfg.verbose:
                logger.info("ROLLOUT:")
                for k, v in info.items():
                    logger.info('%s', {k: v})

            step_per_batch = mpi_sum(run_step)

            # train agents
            if self.cfg.verbose:
                logger.info('Update networks %d', update_iter)

            train_info_global = []
            for i in range(num_agents):
                if self.cfg.off_policy:
                    self.agents[i].store_episode(rollout, i)
                    train_info = self.agents[i].train(i)
                else:
                    batch = self.agents[i].compute_gae(rollout, i)
                    train_info = self.agents[i].train(batch, i)
                train_info_global.append(train_info)

            if self.cfg.verbose:
                logger.info('Update networks done')

            step += step_per_batch
            update_iter += 1

            # log training and episode information
            pbar.update(step_per_batch)
            if update_iter % self.cfg.log_interval == 0:
                train_info_global.append({
                    'sec': (time() - st_time) / self.cfg.log_interval,
                    'steps_per_sec': (step - st_step) / (time() - st_time),
                    'update_iter': update_iter
                })
                st_time = time()
                st_step = step
                self._log_train(step, train_info_global, num_agents)

            if update_iter % self.cfg.ckpt_interval == 0:
                for i in range(num_agents):
                    self._save_ckpt(step, update_iter, i)

            # Evolution
            if self.cfg.pbrl:
                if step >= self.cfg.start_evo and update_iter % evo_iter == 0:
                    # Sort returns over last rollouts
                    returns = {}
                    for i in range(num_agents):
                        returns.update({('agent', i): []})
                        for j in range(len(returns_aux)):
                            returns[('agent', i)] = np.append(returns[('agent', i)], returns_aux[j][i])
                        # sort returns
                        returns[('agent', i)] = sorted(returns[('agent', i)])
                        # select returns from only best N rollouts
                        # returns[('agent', i)] = returns[('agent', i)][-20:]

                    fitness = {}
                    returns_aux = []

                    for i in range(num_agents):
                        fitness.update({(str(i)): np.mean(returns[('agent', i)])})
                    ranking = OrderedDict(sorted(fitness.items(), key=lambda t: t[1]))
                    agents_sorted = [int(w) for w in ranking.keys()]

                    num_selected = int(math.ceil(num_agents / 4.0))  # 25 percent of the population
                    top_agents = agents_sorted[-num_selected:]
                    bottom_agents = agents_sorted[:num_selected]
                    # mid_agents = agents_sorted[num_selected:-num_selected]

                    logger.warning('Evolution of the agents')
                    logger.info('Sorted Returns of the Workers:')
                    for k, v in ranking.items():
                        logger.info('%s', {k: v})

                    logger.info('Top 25% Workers: {}'.format(top_agents))
                    logger.info('Bottom 25% Workers: {}'.format(bottom_agents))

                    for w in bottom_agents:
                        # randomly choose the agents that will replace
                        rand_idx = np.random.randint(0, len(top_agents))
                        successor = top_agents[rand_idx]

                        logger.info('Replacing the bottom 25% workers by random top 25%')
                        logger.info('{} ------> {}' .format(successor, w))

                        soft_update_target_network(self.agents[w].actor, self.agents[successor].actor, 1.0)
                        soft_update_target_network(self.agents[w].critic, self.agents[successor].critic, 1.0)

                        # replace the normalizers
                        if self.cfg.ob_norm:
                            self.agents[w].obs_norm = deepcopy(self.agents[successor].obs_norm)
                        if self.cfg.asymmetric_ac:
                            self.agents[w].state_norm = deepcopy(self.agents[successor].state_norm)
                        if self.cfg.off_policy:
                            soft_update_target_network(self.agents[w].critic_target,
                                                       self.agents[successor].critic_target, 1.0)
                            soft_update_target_network(self.agents[w].actor_target,
                                                       self.agents[successor].actor_target, 1.0)

                            # replace the buffers
                            del self.agents[w].n_step_buffer
                            self.agents[w].n_step_buffer = deepcopy(self.agents[successor].n_step_buffer)
                            del self.agents[w].memory
                            self.agents[w].memory = deepcopy(self.agents[successor].memory)

                        logger.warning('Mutation of the parameters')

                        for k, v in params_to_tune.items():
                            # Mutation: Perturbation
                            if self.cfg.mut_scheme == "perturb":
                                rand_idx = np.random.randint(0, len(mutation_coeffs))
                                mutation_rand = mutation_coeffs[rand_idx]  # random mutation factor
                                mut_param = getattr(self.agents[successor], k) * mutation_rand
                                if k == "actor_lr":
                                    mut_param = self.agents[successor].actor_optim.param_groups[0]["lr"] * mutation_rand
                                    self.agents[w].actor_optim.param_groups[0]["lr"] = mut_param
                                elif k == "critic_lr":
                                    mut_param = self.agents[successor].critic_optim.param_groups[0]["lr"] * mutation_rand
                                    self.agents[w].critic_optim.param_groups[0]["lr"] = mut_param
                                else:
                                    # mut_param = np.clip(mut_param, v[0], v[1])  # whether to enforce param bounds
                                    setattr(self.agents[w], k, mut_param)
                            
                            # Mutation: Sampling from a uniform distribution
                            elif self.cfg.mut_scheme == "sample":
                                mut_param = np.random.uniform(v[0], v[1])
                                if k == "actor_lr":
                                    self.agents[w].actor_optim.param_groups[0]["lr"] = mut_param
                                elif k == "critic_lr":
                                    self.agents[w].critic_optim.param_groups[0]["lr"] = mut_param
                                else:
                                    setattr(self.agents[w], k, mut_param)
                            
                            # Mutation: DEX-PBT scheme (refer to the paper)                            
                            elif self.cfg.mut_scheme == "dex-pbt":
                                perturb_amount = np.random.uniform(mutation_coeffs[0], mutation_coeffs[1])
                                if np.random.rand() < 0.5:
                                    if k == "actor_lr":
                                        mut_param = self.agents[successor].actor_optim.param_groups[0]["lr"] / perturb_amount
                                        self.agents[w].actor_optim.param_groups[0]["lr"] = mut_param
                                    elif k == "critic_lr":
                                        mut_param = self.agents[successor].critic_optim.param_groups[0]["lr"] / perturb_amount
                                        self.agents[w].critic_optim.param_groups[0]["lr"] = mut_param
                                    else:
                                        mut_param = getattr(self.agents[successor], k) / perturb_amount
                                        setattr(self.agents[w], k, mut_param)
                                else:
                                    if k == "actor_lr":
                                        mut_param = self.agents[successor].actor_optim.param_groups[0]["lr"] * perturb_amount
                                        self.agents[w].actor_optim.param_groups[0]["lr"] = mut_param
                                    elif k == "critic_lr":
                                        mut_param = self.agents[successor].critic_optim.param_groups[0]["lr"] * perturb_amount
                                        self.agents[w].critic_optim.param_groups[0]["lr"] = mut_param
                                    else:
                                        mut_param = getattr(self.agents[successor], k) * perturb_amount
                                        setattr(self.agents[w], k, mut_param)
                        logger.info('Mutation of the parameters done')

        logger.info('Reached %s steps. Training stopped.', step)

    def evaluate(self, num_agents):
        """ Evaluates an agent stored in chekpoint with @self.cfg.ckpt_num. """

        for i in range(num_agents):
            step, update_iter = self._load_ckpt_eval(i, ckpt_num=self.cfg.ckpt_num)

        logger.info('Run %d evaluations at step=%d, update_iter=%d', self.cfg.num_eval, step, update_iter)

        for i in trange(self.cfg.num_eval):
            logger.warning("Evalute run %d", i+1)
            rollout, info = self._evaluate()

    # Helper Functions
    def _save_ckpt(self, ckpt_num, update_iter, i):
        """
        Save checkpoint to log directory.

        Args:
            ckpt_num: number appended to checkpoint name. The number of environment step is used in this code.
            update_iter: number of policy update.
        """
        ckpt_path = os.path.join(self.cfg.log_dir + '/agent_' + str(i), 'ckpt_%08d.pt' % (ckpt_num))
        state_dict = {'step': ckpt_num, 'update_iter': update_iter}
        state_dict['agent'] = self.agents[i].state_dict()
        torch.save(state_dict, ckpt_path)
        logger.warning('Save checkpoint: %s', ckpt_path)

    def _load_ckpt(self, ckpt_num=None):
        """
        Loads checkpoint with index number @ckpt_num. If @ckpt_num is None,
        it loads and returns the checkpoint with the largest index number.
        """
        ckpt_path, ckpt_num = get_ckpt_path(self.cfg.log_dir, ckpt_num)

        if ckpt_path is None:
            logger.warning('Randomly initialize models')
            return 0, 0

    def _load_ckpt_eval(self, i, ckpt_num=None):
        """
        Loads checkpoint for evaluating the policy with index number.
        """
        ckpt_path, ckpt_num = get_ckpt_path(self.cfg.log_dir + '/agent_' + str(i), ckpt_num)

        if ckpt_path is not None:
            logger.warning('Load checkpoint %s', ckpt_path)
            ckpt = torch.load(ckpt_path)
            self.agents[i].load_state_dict(ckpt['agent'])

            return ckpt['step'], ckpt['update_iter']

    def _log_ep(self, step, ep_info):
        """
        Logs episode information to wandb.
        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
        """
        for k, v in ep_info.items():
            if k == 'len':
                wandb.log({'train_ep/%s' % k: np.mean(v)}, step=step)

            elif isinstance(k, tuple):
                wandb.log({'train_ep/%s_%d' % k: np.mean(v)}, step=step)
                wandb.log({'train_ep_max/%s_%d' % k: np.max(v)}, step=step)

    def _log_train(self, step, train_info_global, num_agents):
        """
        Logs training information to wandb.
        Args:
            step: the number of environment steps.
            train_info: training information to log, such as loss, gradient.
            num_agents: number of agents.
        """
        for i in range(num_agents + 1):
            for k, v in train_info_global[i].items():

                if isinstance(k, tuple):
                    wandb.log({'train_rl/%s_%d' % k: v}, step=step)
                else:
                    wandb.log({'train_rl/%s' % k: v}, step=step)

    def _evaluate(self):
        """
        Runs a rollout.
        """
        for i in range(self.cfg.num_record_samples):
            self.runner.reset_envs()
            rollout, info, _ = self.runner.run_episode(self.cfg.horizon)

        logger.info("\nROLLOUT")
        for k, v in info.items():
            logger.info('%s', {k: v})
        return rollout, info
