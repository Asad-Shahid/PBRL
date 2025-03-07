#!/bin/bash

# Define an array of configurations
CONFIGS=(
    "VP-SH-SAC 101-3 101 seed-101"
    "VP-SH-SAC 102-3 102 seed-102"
	"VP-SH-SAC 103-3 103 seed-103"
	"VP-SH-SAC 104-3 104 seed-104"
	"VP-SH-SAC 105-3 105 seed-105"
	"VP-SH-SAC 106-3 106 seed-106"
	"VP-SH-SAC 107-3 107 seed-107"
	"VP-SH-SAC 108-3 108 seed-108"
	"VP-SH-SAC 109-3 109 seed-109"
	"VP-SH-SAC 110-3 110 seed-110"
)

# Common parameters
NUM_ENVS=2048
PBRL=False
NUM_AGENTS=1
PBRL_PARAMS=sac
ALGO=sac
NUM_EPOCHS=4
BATCH_SIZE=4096
HORIZON=1
TASK=ShadowHand
HEADLESS=True
WANDB=True
WANDB_ENTITY=panda
WANDB_PROJECT=shadow-hand

# Loop through each configuration and execute the script
for CONFIG in "${CONFIGS[@]}"; do
    read PREFIX SUFFIX SEED NOTES <<< "$CONFIG"
    
    python -m pbrl.main \
        --num_envs $NUM_ENVS \
        --pbrl $PBRL \
        --num_agents $NUM_AGENTS \
        --pbrl_params $PBRL_PARAMS \
        --algo $ALGO \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --horizon $HORIZON \
        --task $TASK \
        --headless $HEADLESS \
        --wandb $WANDB \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project $WANDB_PROJECT \
        --prefix $PREFIX \
        --suffix $SUFFIX \
        --seed $SEED \
        --notes $NOTES;
    
    # Cleanup after each run
    cd log/
    python3 delete_checkpoint.py
    cd -
done
