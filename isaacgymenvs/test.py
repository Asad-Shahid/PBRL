from isaacgymenvs.cfg.pbrl import argparser
from isaacgymenvs.pbrl.evaluator import PolicyEvaluator
import os


config, unparsed = argparser()

policy = PolicyEvaluator(config)
path = os.path.join('/home/asad/git/industreallib/src/industreallib/rl/checkpoints/franka_pick/nn', 'franka_pick.pt')
policy.restore(path)
policy.evaluate()
