# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for nut-bolt pick task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskReach
"""

import hydra
import omegaconf
import os
import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgymenvs.utils import torch_jit_utils as torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_env_reach import FactoryEnvReach
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgymenvs.utils import torch_jit_utils

import matplotlib.pyplot as plt


class FactoryTaskReach(FactoryEnvReach, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()  # defined in superclass

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()  # defined in superclass

        if self.viewer is not None:
            self._set_viewer_params()

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_nut_bolt.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

        ppo_path = 'train/FactoryTaskReachPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(
            self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_gripper = torch.zeros(
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3), dtype=torch.float32, device=self.device)
        self.keypoints_goal = torch.zeros_like(self.keypoints_gripper, device=self.device)

        # Identity quaternion
        self.identity_quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pos of keypoints on gripper and nut in base frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_gripper[:, idx] = torch_jit_utils.tf_combine(self.fingertip_midpoint_quat,
                self.fingertip_midpoint_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1))[1]
            self.keypoints_goal[:, idx] = torch_jit_utils.tf_combine(self.goal_quat,
                self.goal_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1))[1]

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(actions=self.actions,
            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max, do_scale=True)

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()  # Defined in superclass
        self.refresh_env_tensors()  # Defined in superclass
        self._refresh_task_tensors()
        self.compute_observations()

        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        # Convert poses to robot base frame
        fingertip_midpoint_pos, fingertip_midpoint_quat = self.pose_world_to_robot_base(
            self.fingertip_midpoint_pos, self.fingertip_midpoint_quat)
        goal_pos, goal_quat = self.pose_world_to_robot_base(self.goal_pos, self.goal_quat)

        # Compute observation tensor
        obs_tensors = [
            self.arm_dof_pos,  # 7
            fingertip_midpoint_pos,  # 3
            fingertip_midpoint_quat,  # 4
            goal_pos,  # 3
            goal_quat,  # 4
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        return self.obs_buf

    def compute_reward(self):
        """Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf), self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        # keypoint_reward = -self._get_keypoint_dist()
        distance_reward = -self._get_distance()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale

        self.rew_buf[:] = distance_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)

        self._reset_buffers(env_ids)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        self.dof_pos[env_ids] = torch.cat(
            (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
            dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

    def _reset_object(self, env_ids):
        """Reset root states of goal."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Get number of environments to reset
        num_envs = len(env_ids)

        # Get randomize configuration
        goal_pos = torch.tensor(self.cfg_task.randomize.goal_pos, device=self.device)
        goal_rot = torch.tensor(self.cfg_task.randomize.goal_rot, device=self.device)

        # Randomize goal state
        random_goal_pos = torch.distributions.Uniform(goal_pos[:, 0], goal_pos[:, 1]).sample((num_envs,))
        random_goal_rot = torch.distributions.Uniform(goal_rot[:, 0], goal_rot[:, 1]).sample((num_envs,))

        # Convert goal rot to quaternion
        random_goal_quat = torch_utils.quat_from_euler_xyz(
            random_goal_rot[:, 0], random_goal_rot[:, 1], random_goal_rot[:, 2])

        # Transform target pose from robot base frame to world frame
        goal_pose = self.pose_robot_base_to_world(random_goal_pos, random_goal_quat)
        self.root_pos[env_ids, self.goal_actor_id_env, :] = goal_pose[0]
        self.root_quat[env_ids, self.goal_actor_id_env, :] = goal_pose[1]

        # Set goal velocity to 0
        self.root_linvel[env_ids, self.goal_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.goal_actor_id_env] = 0.0

        # Call gym API to set the goal state
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_state), gymtorch.unwrap_tensor(self.goal_actor_ids_sim[env_ids]), num_envs)

    def _reset_buffers(self, env_ids):
        pass
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat, torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets
    
    def _get_distance(self):
        """Get distance from robot end-effector to nut"""

        pos_error = torch.sum(torch.norm(self.goal_pos - self.finger_midpoint_pos, p=2, dim=-1), dim=-1)
        ori_error = torch.sum(torch.norm(torch_utils.quat_mul(
            self.goal_quat, torch_utils.quat_conjugate(self.fingertip_midpoint_quat)), dim=-1), dim=-1)
        return pos_error + ori_error * 0

    def _get_keypoint_dist(self):
        """Get keypoint distance."""

        return torch.sum(torch.norm(self.keypoints_goal - self.keypoints_gripper, p=2, dim=-1), dim=-1)

    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # Get randomize configuration
        fingertip_midpoint_pos = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos, device=self.device)
        fingertip_midpoint_rot = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot, device=self.device)
        
        # Set target pos in robot base frame
        ctrl_target_pos = torch.distributions.Uniform(
            fingertip_midpoint_pos[:, 0], fingertip_midpoint_pos[:, 1]).sample((self.num_envs,))

        # Set target rot in robot base frame
        ctrl_target_rot = torch.distributions.Uniform(
            fingertip_midpoint_rot[:, 0], fingertip_midpoint_rot[:, 1]).sample((self.num_envs,))

        # Convert target rot to quaternion
        ctrl_target_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_rot[:, 0], ctrl_target_rot[:, 1], ctrl_target_rot[:, 2])

        # Transform target pose from robot base frame to world frame
        ctrl_target_pose = self.pose_robot_base_to_world(ctrl_target_pos, ctrl_target_quat)
        self.ctrl_target_fingertip_midpoint_pos = ctrl_target_pose[0]
        self.ctrl_target_fingertip_midpoint_quat = ctrl_target_pose[1]

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()  # Defined in superclass
            self.refresh_env_tensors()  # Defined in superclass
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'], rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=actions,
                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max, do_scale=False)

            self.gym.simulate(self.sim)
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
