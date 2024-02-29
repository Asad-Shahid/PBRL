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

"""Factory: class for reach.

Inherits base class and abstract environment class. Inherited by reach task classes. Not directly executed.

Configuration defined in FactoryEnvReach.yaml.
"""

import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi, gymutil
from isaacgymenvs.tasks.factory.factory_base import FactoryBase
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from isaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv


class FactoryEnvReach(FactoryBase, FactoryABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_env_yaml_params()

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        config_path = 'task/FactoryEnvReach.yaml'  # relative to Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        goal_asset = self._import_env_assets()
        self._create_actors(lower, upper, num_per_row, franka_asset, table_asset, goal_asset)

    def _import_env_assets(self):
        """Set goal asset options. Create assets."""

        goal_options = gymapi.AssetOptions()
        goal_options.angular_damping = 0.0  # default = 0.5
        goal_options.armature = 0.0  # default = 0.0
        goal_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        goal_options.density = self.cfg_env.env.goal_density
        goal_options.disable_gravity = True
        goal_options.enable_gyroscopic_forces = True
        goal_options.fix_base_link = False
        goal_options.flip_visual_attachments = False
        goal_options.linear_damping = 0.0  # default = 0.0
        goal_options.max_angular_velocity = 64.0  # default = 64.0
        goal_options.max_linear_velocity = 1000.0  # default = 1000.0
        goal_options.thickness = 0.0  # default = 0.02
        goal_options.use_mesh_materials = False
        goal_options.use_physx_armature = True

        if self.cfg_base.mode.export_scene:
            goal_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        return self.gym.create_sphere(self.sim, self.cfg_env.env.goal_radius, goal_options)

    def _create_actors(self, lower, upper, num_per_row, franka_asset, table_asset, goal_asset):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        franka_pose = gymapi.Transform()
        franka_pose.p.x = self.cfg_env.env.franka_depth
        franka_pose.p.y = 0.0
        franka_pose.p.z = self.cfg_base.env.table_height
        franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.franka_handles = []
        self.table_handles = []
        self.goal_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        self.goal_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        for i in range(self.num_envs):

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i + self.num_envs,
                                                      0, 0)
            else:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i, 0, 0)
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, 'table', i, 0, 0)
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            goal_pose = gymapi.Transform()
            goal_pose.p.x = 0.0
            goal_pose.p.y = self.cfg_env.env.goal_lateral_offset
            goal_pose.p.z = self.cfg_base.env.table_height + self.cfg_env.env.goal_height_offset
            goal_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_pose, 'goal', i, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.9, .25, .15))
            self.goal_actor_ids_sim.append(actor_count)
            actor_count += 1

            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_link7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand', gymapi.DOMAIN_ACTOR)
            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                  gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_rightfinger',
                                                                   gymapi.DOMAIN_ACTOR)
            self.shape_ids = [link7_id, hand_id, left_finger_id, right_finger_id]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            goal_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, goal_handle)
            goal_shape_props[0].friction = self.cfg_env.env.goal_friction
            goal_shape_props[0].rolling_friction = 0.0  # default = 0.0
            goal_shape_props[0].torsion_friction = 0.0  # default = 0.0
            goal_shape_props[0].restitution = 0.0  # default = 0.0
            goal_shape_props[0].compliance = 0.0  # default = 0.0
            goal_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, goal_handle, goal_shape_props)

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.table_handles.append(table_handle)
            self.goal_handles.append(goal_handle)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.goal_actor_ids_sim = torch.tensor(self.goal_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.goal_actor_id_env = self.gym.find_actor_index(env_ptr, 'goal', gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.robot_base_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_link0", gymapi.DOMAIN_ENV)        
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, 'panda_hand', gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, 'panda_leftfinger', gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, 'panda_rightfinger', gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, 'panda_fingertip_centered', gymapi.DOMAIN_ENV)
        self.goal_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, goal_handle, 'goal', gymapi.DOMAIN_ENV) 

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        self.goal_pos = self.root_pos[:, self.goal_actor_id_env, 0:3]
        self.goal_quat = self.root_quat[:, self.goal_actor_id_env, 0:4]
        self.goal_linvel = self.root_linvel[:, self.goal_actor_id_env, 0:3]
        self.goal_angvel = self.root_angvel[:, self.goal_actor_id_env, 0:3]
        self.goal_force = self.contact_force[:, self.goal_body_id_env, 0:3]

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.
        pass
