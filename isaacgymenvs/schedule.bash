#!/bin/bash

# With this script you can schedule successive runs

python -m pbrl.main \
	--num_envs 4096 \
	--pbrl False \
	--num_agents 1 \
	--algo ppo \
	--prefix VP-PPO \
	--suffix 971108 \
	--seed 971108 \
	--notes seed-971108; \
python -m pbrl.main \
	--num_envs 4096 \
	--pbrl False \
	--num_agents 1 \
	--algo ppo \
	--prefix VP-PPO \
	--suffix 666 \
	--seed 666 \
	--notes seed-666; \
python -m pbrl.main \
	--num_envs 4096 \
	--pbrl True \
	--num_agents 4 \
	--algo ppo \
	--prefix VP-PBRL-PPO \
	--suffix 4-agents-1919 \
	--seed 1919 \
	--notes seed-1919;
