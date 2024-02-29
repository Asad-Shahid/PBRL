from isaacgymenvs.pbrl.env import create_environments
from isaacgymenvs.pbrl.actor_critic import Actor, Critic, SACPolicy, DoubleQCritic
from isaacgymenvs.pbrl.ppo_agent import PPOAgent
from isaacgymenvs.pbrl.sac_agent import SACAgent
from isaacgymenvs.pbrl.ddpg_agent import DDPGAgent
from isaacgymenvs.pbrl.rollout import RolloutRunner
from isaacgymenvs.utils.pbrl.pytorch import get_ckpt_path
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.pbrl.logger import logger
from isaacgym import torch_utils
import hydra
import torch
import numpy as np
import os
import tf.transformations as tf

class PolicyEvaluator():
    """
    Policy Evaluator class to test single agent in sim or deploy on real robot.
    """

    def __init__(self, config, obs_dim=None, ac_dim=None):
        """
        Initializes class with the task configuration and creates an Agent.
        """
        self.cfg = config
        self.cfg.envs_per_agent = int(config.num_envs / config.num_agents) # required to evaluate in sim
        state_dim = 0
        hid_size = self.cfg.hid_size
        activation = self.cfg.activation
        if self.cfg.gpu is not None:
            self.cfg.device = torch.device("cuda")
            torch.set_default_device(self.cfg.device)
        self.cfg.asymmetric_ac = state_dim > 0
        self.cfg.off_policy = self.cfg.algo != 'ppo'
        
        cfg_dict = self.get_cfg_dict(self.cfg.num_envs, self.cfg.task, self.cfg.sub_task)

        obs_dim = obs_dim if obs_dim is not None else cfg_dict['env']['numObservations']
        ac_dim = ac_dim if ac_dim is not None else cfg_dict['env']['numActions']

        if self.cfg.off_policy:
            critic = DoubleQCritic(obs_dim, ac_dim, hid_size, activation).to(self.cfg.device) # same critic for both sac and ddpg
            if config.algo == 'sac':
                actor = SACPolicy(obs_dim, ac_dim*2, hid_size, activation).to(self.cfg.device)
                self.agent = SACAgent(config, actor, critic, obs_dim, ac_dim, init_params=None)
            else:
                actor = Actor(obs_dim, ac_dim, hid_size, activation).to(self.cfg.device)
                self.agent = DDPGAgent(config, actor, critic, obs_dim, ac_dim, init_params=None)
        else:
            actor = Actor(obs_dim, ac_dim, hid_size, activation).to(self.cfg.device)
            critic = Critic(obs_dim, hid_size, activation).to(self.cfg.device)
            self.agent = PPOAgent(self.cfg, actor, critic, obs_dim, state_dim, init_params=None)

        if self.cfg.sim:
            envs = create_environments(self.cfg.task, cfg_dict, self.cfg.headless)
            self.agents = np.array([self.agent])
            self.runner = RolloutRunner(self.cfg, envs, self.agents)

    
    def restore(self, ckpt_path=None):
        # laod checkpoint
        if ckpt_path is None:
            run_name  = '{}.{}'.format(self.cfg.prefix, self.cfg.suffix)
            log_dir = os.path.join(self.cfg.log_root_dir, run_name)
            ckpt_path, ckpt_num = get_ckpt_path(log_dir + '/agent_' + str(0), self.cfg.ckpt_num)
        logger.info(f'Loading checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        
        logger.info(f'Restoring the RL policy from checkpoint: {ckpt_path}')
        self.agent.load_state_dict(ckpt['agent'])
        if self.cfg.sim:
            self.agents[0].load_state_dict(ckpt['agent']) # required for evaluating in simulation

    def evaluate(self):   
        # run rollouts for evaluation
        for i in range(self.cfg.num_eval):
            logger.info(f"Evalute run {i+1}")
            self.runner.reset_envs()
            rollout, info, _ = self.runner.run_episode(self.cfg.horizon, is_deterministic=True)
            logger.info("ROLLOUT:")
            for k, v in info.items():
                logger.info('%s', {k: v})

    def get_action(self, obs, is_deterministic=True):   
        # run rollouts for evaluation
        if self.cfg.off_policy:
            action = self.agent.act(obs, is_deterministic)
        else:
            action, _, _ = self.agent.act(obs, is_deterministic)
        return action
    
    def get_cfg_dict(self, num_envs, task, sub_task):
        hydra.initialize(config_path="../cfg")
        cfg = hydra.compose(config_name="config", overrides=[f"task={task}"])
        cfg_dict = omegaconf_to_dict(cfg.task)
        cfg_dict['env']['numEnvs'] = num_envs
        cfg_dict['env']['subtask'] = sub_task
        return cfg_dict
        
    def pose_world_to_robot_base(self, pos, quat):
        """Convert pose from fixed frame to robot base frame."""

        # Factory task pick
        robot_base_quat = torch.tensor([0.0, 0.0, 1.0, 0.0])
        robot_base_pos = torch.tensor([0.5, 0.0, 0.4])

        robot_base_transform_inv = torch_utils.tf_inverse(robot_base_quat, robot_base_pos)
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(
            robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos)

        return pos_in_robot_base, quat_in_robot_base
