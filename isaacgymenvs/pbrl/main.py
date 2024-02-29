import sys
import signal
import json
import os
import random
import hydra
import numpy as np
from isaacgymenvs.cfg.pbrl import argparser
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.pbrl.env import create_environments
from isaacgymenvs.pbrl.trainer import Trainer
from isaacgymenvs.utils.pbrl.logger import logger
import torch


def run():

    config, unparsed = argparser()
    config.envs_per_agent = int(config.num_envs / config.num_agents)

    if not config.num_envs % config.num_agents == 0:
        raise ValueError('Number of workers must fully divide the number of environments')

    logger.warn('Running a population of workers')
    make_log_files(config)

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # set global seed
    random.seed(config.seed)
    np.random.seed(config.seed)

    # set the display no. configured with gpu
    os.environ["DISPLAY"] = ":1"

    # use gpu or cpu
    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)
        assert torch.cuda.is_available()
        config.device = torch.device("cuda")
        torch.set_default_device(config.device)
        torch.manual_seed(config.seed)
    else:
        config.device = torch.device("cpu")
        torch.manual_seed(config.seed)

    # load pbrl params file
    pbrl_param_file = os.path.join('cfg/pbrl/' + config.pbrl_params + '.json')
    with open(pbrl_param_file, 'rt') as r:
        pbrl_params = json.load(r)

    params_to_tune = {}
    for param, value in pbrl_params.items():
        if not param.startswith('mutation'):
            params_to_tune[str(param)] = [float(val) for val in value.split(',')]
        else:
            mutation_coeffs = [float(val) for val in value.split(',')]
    
    cfg_dict = get_cfg_dict(config.num_envs, config.task, config.sub_task)
    envs = create_environments(config.task, cfg_dict, config.headless)
    trainer = Trainer(config, envs, params_to_tune)

    if config.is_train:
        trainer.train(params_to_tune, mutation_coeffs)
        logger.info("Finished training")

    else:
        trainer.evaluate(config.num_agents)
        logger.info("Finished evaluating")


def make_log_files(config):
    """
    Sets up log directories.
    """
    config.run_name = '{}.{}'.format(config.prefix, config.suffix)
    config.log_dir = os.path.join(config.log_root_dir, config.run_name)
    logger.info('Create log directory: %s', config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)
    for i in range(config.num_agents):
        os.makedirs((config.log_dir + '/agent_' + str(i)), exist_ok=True)

def get_cfg_dict(num_envs, task, sub_task):
    hydra.initialize(config_path="../cfg")
    cfg = hydra.compose(config_name="config", overrides=[f"task={task}"])
    cfg_dict = omegaconf_to_dict(cfg.task)
    cfg_dict['env']['numEnvs'] = num_envs
    cfg_dict['env']['subtask'] = sub_task
    return cfg_dict

if __name__ == '__main__':
    run()
