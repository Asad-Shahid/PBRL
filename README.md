# PBRL - Population-Based Reinforcement Learning
[Website](https://sites.google.com/view/pbrl) | [Paper](https://sites.google.com/view/pbrl) | [Videos](https://drive.google.com/file/d/1keHq58eQPqFtObyXEti-oR8SIek8KgLV/view)

## About this repository

This repository contains the code to train RL agents in Population-Based manner.

![](pbrl-policy.gif)


## Installation

1. Create a new conda environment with:

    ```sh
    conda create -n pbrl python=3.8
    conda activate pbrl
    ```

2. Install IsaacGym (tested with `Preview 4 Release`). Follow the [instructions](https://developer.nvidia.com/isaac-gym) to download the package.

    ```sh
    tar -xvf IsaacGym_Preview_4_Package.tar.gz
    cd isaacgym/python
    pip install -e .
    # Test IsaacGym installation
    cd examples
    python joint_monkey.py
    ```

3. Install this repo:

    ```sh
    git clone https://github.com/Asad-Shahid/PBRL.git
    cd PBRL 
    pip install -e .
    ```

## Getting Started

Navigate to the `isaacgymenvs` directory and run:

```python
python -m pbrl.main --task ShadowHand --num_envs 4096 --num_agents 4
```

Some key arguments are:

- `--task` selects a task from isaacgymenvs. All tasks released in `isaacgymenvs==1.5.1` are supported.
- `--num_envs` selects the number of environments to run.
- `--num_agents` chooses the number of agents to train in parallel for pbrl. Note: `num_envs` must divide `num_agents`.
- `--pbrl` whether to use PBRL.\
 Note: when `True`, `num_agents` must be a multiple of `4` (Top 25% of the agents are selected to replace/get replaced).
- `--algo` which RL algorithm to use for training. Options are: `ppo, sac, ddpg`.
- `--pbrl_params` name of `.json` file with hyperparameters to tune in [`cfg/pbrl`](./isaacgymenvs/cfg/pbrl). Only hyperparameters listed in corresponding `.json` files are currently supported. Initial values are sampled uniformly from the specified range. When training a single RL agent, specify inital values in the file.
- `--mut_scheme` which mutation scheme to use for mutating hyperparameters.

All other arguments can be found in [`cfg/pbrl/_init__.py`](./isaacgymenvs/cfg/pbrl/__init__.py)


## Citing

Please cite this work as:

```bibtex
@article{Shahid2024pbrl,
  author = {Asad Ali Shahid and Yashraj Narang and Vincenzo Petrone and Enrico Ferrentino and Dieter Fox and Marco Pavone and Loris Roveda},
  title = {Scaling Population-Based Reinforcement Learning with GPU Accelerated Simulation},
  journal = {arXiv preprint},
  year = {2024},
  doi = {10.48550/arXiv.TBD.TBD},
  url = {https://doi.org/10.48550/arXiv.TBD.TBD},
}
```

If you reuse our code, you can site this repo as:

```bibtex
@software{Shahid2024pbrlrepo,
author = {Shahid, Asad Ali and Narang, Yashraj and Petrone, Vincenzo and Ferrentino, Enrico and Handa, Ankur and Fox, Dieter and Pavone, Marco and Roveda, Loris},
doi = {10.5281/zenodo.TBD},
month = mar,
title = {{Scaling Population-Based Reinforcement Learning with GPU Accelerated Simulation}},
url = {https://github.com/Asad-Shahid/PBRL},
version = {1.0.0},
year = {2024}
}
```
