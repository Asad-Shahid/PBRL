import isaacgym
import isaacgymenvs
import hydra
from isaacgymenvs.tasks import isaacgym_task_map
from hydra.core.hydra_config import HydraConfig
import torch


def create_environments(task, cfg_dict, headless):
    # reset current hydra config if already parsed (but not passed in here)
    if HydraConfig.initialized():
        task = HydraConfig.get().runtime.choices['task']
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Create an env
    env = isaacgym_task_map[task](
        cfg=cfg_dict,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=headless,
        virtual_screen_capture=False,
        force_render=True,
    )
    return env
