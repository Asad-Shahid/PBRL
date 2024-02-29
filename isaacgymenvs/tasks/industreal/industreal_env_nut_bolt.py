# Copyright (c) 2023, NVIDIA Corporation
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

"""IndustReal: class for pegs environment.

Inherits IndustReal base class and Factory abstract environment class. 
Inherited by IndustReal nut-bolt engagement task class.
Not directly executed.

Configuration defined in IndustRealEnvNutBolt.yaml.
Asset info defined in factory_asset_info_nut_bolt.yaml.
"""


import hydra
import math
import numpy as np
import os
import torch

from isaacgym import gymapi
from isaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from isaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from isaacgymenvs.tasks.industreal.industreal_base import IndustRealBase


class IndustRealEnvNutBolt(IndustRealBase, FactoryABCEnv):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_env_yaml_params()

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = "task/IndustRealEnvNutBolt.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../../assets/factory/yaml/factory_asset_info_nut_bolt.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt[""][""][""][""][""][""]["assets"]["factory"]["yaml"]  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        # Set environment options
        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        # Import assets
        self.print_sdf_warning()
        franka_asset, table_asset = self._import_franka_assets()
        plug_assets, socket_assets = self._import_env_assets()
        
        # Create actors
        self._create_actors(lower, upper, num_per_row, franka_asset, plug_assets, socket_assets, table_asset)

    def _import_franka_assets(self):
        """Set Franka and table asset options. Import assets."""

        # Set Franka URDF path
        urdf_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "idsia", "urdf")
        franka_file = "idsia_franka.urdf"

        # Set Franka asset options
        franka_options = gymapi.AssetOptions()
        franka_options.armature = 0.01  # default = 0.0
        franka_options.collapse_fixed_joints = False
        franka_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        franka_options.density = 1000.0  # default = 1000.0
        franka_options.disable_gravity = True
        franka_options.enable_gyroscopic_forces = True
        franka_options.fix_base_link = True
        franka_options.flip_visual_attachments = True
        franka_options.max_angular_velocity = 2 * math.pi  # default = 64.0; reduced to prevent CUDA errors
        franka_options.max_linear_velocity = 1.0  # default = 1000.0; reduced to prevent CUDA errors
        franka_options.thickness = 0.0  # default = 0.02
        franka_options.use_mesh_materials = True
        franka_options.use_physx_armature = True
        if self.cfg_base.sim.add_damping:
            franka_options.angular_damping = 5.0  # default = 0.5; increased to improve stability
            franka_options.linear_damping = 1.0  # default = 0.0; increased to improve stability
        else:
            franka_options.angular_damping = 0.5  # default = 0.5
            franka_options.linear_damping = 0.0  # default = 0.0
        if self.cfg_base.mode.export_scene:
            franka_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        # Set table asset options
        table_options = gymapi.AssetOptions()
        table_options.angular_damping = 0.0  # default = 0.5
        table_options.armature = 0.0  # default = 0.0
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_options.density = 1000.0  # default = 1000.0
        table_options.disable_gravity = False
        table_options.enable_gyroscopic_forces = True
        table_options.fix_base_link = True
        table_options.flip_visual_attachments = False  # default = False
        table_options.linear_damping = 0.0  # default = 0.0
        table_options.max_angular_velocity = 64.0  # default = 64.0
        table_options.max_linear_velocity = 1000.0  # default = 1000.0
        table_options.thickness = 0.0  # default = 0.02
        table_options.use_mesh_materials = False
        table_options.use_physx_armature = True
        if self.cfg_base.mode.export_scene:
            table_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        # Import Franka asset
        franka_asset = self.gym.load_asset(self.sim, urdf_root, franka_file, franka_options)

        # Import table asset
        table_asset = self.gym.create_box(
            self.sim,
            self.asset_info_franka_table.table_depth,
            self.asset_info_franka_table.table_width,
            self.cfg_base.env.table_height,
            table_options,
        )

        return franka_asset, table_asset        

    def _import_env_assets(self):
        """Set nut and bolt asset options. Import assets."""

        # Set assets paths
        self.nut_files, self.bolt_files = [], []
        urdf_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "factory", "urdf")

        # Set nut asset options
        nut_options = gymapi.AssetOptions()
        nut_options.angular_damping = 0.5  # default = 0.5
        nut_options.armature = 0.0  # default = 0.0
        nut_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        nut_options.density = self.cfg_env.env.nut_bolt_density
        nut_options.disable_gravity = False
        nut_options.enable_gyroscopic_forces = True
        nut_options.fix_base_link = False
        nut_options.flip_visual_attachments = False
        nut_options.linear_damping = 0.5  # default = 0.0
        nut_options.max_angular_velocity = 64.0  # default = 64.0
        nut_options.max_linear_velocity = 1000.0  # default = 1000.0
        nut_options.thickness = 0.0  # default = 0.02
        nut_options.use_mesh_materials = False
        nut_options.use_physx_armature = True
        if self.cfg_base.mode.export_scene:
            nut_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        # Set bolt asset options
        bolt_options = gymapi.AssetOptions()
        bolt_options.angular_damping = 0.0  # default = 0.5
        bolt_options.armature = 0.0  # default = 0.0
        bolt_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        bolt_options.density = self.cfg_env.env.nut_bolt_density
        bolt_options.disable_gravity = False
        bolt_options.enable_gyroscopic_forces = True
        bolt_options.fix_base_link = True
        bolt_options.flip_visual_attachments = False
        bolt_options.linear_damping = 0.0  # default = 0.0
        bolt_options.max_angular_velocity = 2 * math.pi  # default = 64.0
        bolt_options.max_linear_velocity = 1.0  # default = 1000.0
        bolt_options.thickness = 0.0  # default = 0.02
        bolt_options.use_mesh_materials = False
        bolt_options.use_physx_armature = True
        if self.cfg_base.mode.export_scene:
            bolt_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        # Import nut and bolt assets
        nut_assets = []
        bolt_assets = []
        for subassembly in self.cfg_env.env.desired_subassemblies:
            # Get nut and bolt component names
            components = list(self.asset_info_nut_bolt[subassembly])
            
            # Get nut and bolt file names
            nut_file = self.asset_info_nut_bolt[subassembly][components[0]]["urdf_path"] + ".urdf"
            bolt_file = self.asset_info_nut_bolt[subassembly][components[1]]["urdf_path"] + ".urdf"
            
            # Load nut and bolt assets
            nut_asset = self.gym.load_asset(self.sim, urdf_root, nut_file, nut_options)
            bolt_asset = self.gym.load_asset(self.sim, urdf_root, bolt_file, bolt_options)
            nut_assets.append(nut_asset)
            bolt_assets.append(bolt_asset)

            # Save URDF file paths (for loading appropriate meshes during SAPU and SDF-Based Reward calculations)
            self.nut_files.append(os.path.join(urdf_root, nut_file))
            self.bolt_files.append(os.path.join(urdf_root, bolt_file))

        return nut_assets, bolt_assets

    def _create_actors(
        self,
        lower,
        upper,
        num_per_row,
        franka_asset,
        nut_assets,
        bolt_assets,
        table_asset,
    ):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""
        # NOTE: Closely adapted from FactoryEnvInsertion; however, plug grasp offsets, plug widths, socket heights,
        # and asset indices are now stored for possible use during policy learning."""

        # Set initial Franka pose
        franka_pose = gymapi.Transform()
        franka_pose.p.x = -self.cfg_base.env.franka_depth
        franka_pose.p.y = 0.0
        franka_pose.p.z = self.cfg_base.env.table_height
        franka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Set initial table pose
        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Actor handles
        self.env_ptrs = []
        self.franka_handles = []
        self.nut_handles = []
        self.bolt_handles = []
        self.table_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.nut_actor_ids_sim = []  # within-sim indices
        self.bolt_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        # Asset geometrical properties
        self.nut_widths_max = []
        self.nut_heights = []
        self.bolt_widths = []
        self.bolt_head_heights = []
        self.bolt_shank_lengths = []        
        self.asset_indices = []

        # Create actors
        for i in range(self.num_envs):
            # Create environment
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create Franka actor
            franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, "franka", i, 0, 0)
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            # Get random subassembly
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_nut_bolt[subassembly])

            # Set initial nut pose
            nut_pose = gymapi.Transform()
            nut_pose.p.x = 0.0
            nut_pose.p.y = self.cfg_env.env.nut_lateral_offset
            nut_pose.p.z = self.cfg_base.env.table_height
            nut_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            nut_handle = self.gym.create_actor(env_ptr, nut_assets[j], nut_pose, "nut", i, 0, 0)
            self.nut_actor_ids_sim.append(actor_count)
            actor_count += 1

            # Set initial bolt pose
            bolt_pose = gymapi.Transform()
            bolt_pose.p.x = 0.0
            bolt_pose.p.y = 0.0
            bolt_pose.p.z = self.cfg_base.env.table_height
            bolt_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            socket_handle = self.gym.create_actor(env_ptr, bolt_assets[j], bolt_pose, "bolt", i, 0, 0)
            self.bolt_actor_ids_sim.append(actor_count)
            actor_count += 1

            # Create table actor
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            # Get Franka shape indexes to set physical properties
            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, "panda_link7", gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_ACTOR)
            left_finger_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_leftfinger", gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_rightfinger", gymapi.DOMAIN_ACTOR)
            self.shape_ids = [link7_id, hand_id, left_finger_id, right_finger_id]

            # Set Franka physical properties
            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)

            # Set nut physical properties
            nut_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, nut_handle)
            nut_shape_props[0].compliance = 0.0  # default = 0.0
            nut_shape_props[0].friction = self.cfg_env.env.nut_bolt_friction
            nut_shape_props[0].restitution = 0.0  # default = 0.0
            nut_shape_props[0].rolling_friction = 0.0  # default = 0.0
            nut_shape_props[0].thickness = 0.0  # default = 0.0
            nut_shape_props[0].torsion_friction = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, nut_handle, nut_shape_props)

            # Set bolt physical properties
            bolt_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, socket_handle)
            bolt_shape_props[0].compliance = 0.0  # default = 0.0
            bolt_shape_props[0].friction = self.cfg_env.env.nut_bolt_friction
            bolt_shape_props[0].restitution = 0.0  # default = 0.0
            bolt_shape_props[0].rolling_friction = 0.0  # default = 0.0
            bolt_shape_props[0].thickness = 0.0  # default = 0.0
            bolt_shape_props[0].torsion_friction = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, socket_handle, bolt_shape_props)

            # Set table physical properties
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            # Get number of Franka DOFs
            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            # Enable force sensors at Franka DOFs
            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            # Get nut geometrical properties
            nut_width_max = self.asset_info_nut_bolt[subassembly][components[0]]["width_max"]
            nut_height = self.asset_info_nut_bolt[subassembly][components[0]]["height"]

            # Get bolt geometrical properties
            bolt_width = self.asset_info_nut_bolt[subassembly][components[1]]["width"]
            bolt_head_height = self.asset_info_nut_bolt[subassembly][components[1]]["head_height"]
            bolt_shank_length = self.asset_info_nut_bolt[subassembly][components[1]]["shank_length"]

            # Save actor handles
            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.nut_handles.append(nut_handle)
            self.bolt_handles.append(socket_handle)
            self.table_handles.append(table_handle)

            # Save asset geometrical properties
            self.nut_widths_max.append(nut_width_max)
            self.nut_heights.append(nut_height)
            self.bolt_widths.append(bolt_width)
            self.bolt_head_heights.append(bolt_head_height)
            self.bolt_shank_lengths.append(bolt_shank_length)
            self.asset_indices.append(j)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.nut_actor_ids_sim = torch.tensor(self.nut_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.bolt_actor_ids_sim = torch.tensor(self.bolt_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.nut_actor_id_env = self.gym.find_actor_index(env_ptr, "nut", gymapi.DOMAIN_ENV)
        self.bolt_actor_id_env = self.gym.find_actor_index(env_ptr, "bolt", gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.robot_base_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_link0", gymapi.DOMAIN_ENV)
        self.nut_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, nut_handle, "nut", gymapi.DOMAIN_ENV)
        self.bolt_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, socket_handle, "bolt", gymapi.DOMAIN_ENV)
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_leftfinger", gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_rightfinger", gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_fingertip_centered", gymapi.DOMAIN_ENV)

        self.hand_body_id_env_actor = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_ACTOR)
        self.left_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_leftfinger", gymapi.DOMAIN_ACTOR)
        self.right_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_rightfinger", gymapi.DOMAIN_ACTOR)
        self.fingertip_centered_body_id_env_actor = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_fingertip_centered", gymapi.DOMAIN_ACTOR)

        # For computing body COM pos
        self.nut_widths_max = torch.tensor(self.nut_widths_max, device=self.device)
        self.nut_heights = torch.tensor(self.nut_heights, device=self.device)
        self.bolt_widths = torch.tensor(self.bolt_widths, device=self.device)
        self.bolt_head_heights = torch.tensor(self.bolt_head_heights, device=self.device)
        self.bolt_shank_lengths = torch.tensor(self.bolt_shank_lengths, device=self.device)           

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        # Acquire nut tensors
        self.nut_pos = self.root_pos[:, self.nut_actor_id_env, 0:3]
        self.nut_quat = self.root_quat[:, self.nut_actor_id_env, 0:4]
        self.nut_linvel = self.root_linvel[:, self.nut_actor_id_env, 0:3]
        self.nut_angvel = self.root_angvel[:, self.nut_actor_id_env, 0:3]

        # Acquire bolt tensors
        self.bolt_pos = self.root_pos[:, self.bolt_actor_id_env, 0:3]
        self.bolt_quat = self.root_quat[:, self.bolt_actor_id_env, 0:4]
        self.bolt_linvel = self.root_linvel[:, self.bolt_actor_id_env, 0:3]
        self.bolt_angvel = self.root_angvel[:, self.bolt_actor_id_env, 0:3]

        # Get nut COM tensors
        # TODO: Define socket height and plug height params in asset info YAML.
        # self.plug_com_pos = self.translate_along_local_z(pos=self.plug_pos,
        #                                                  quat=self.plug_quat,
        #                                                  offset=self.socket_heights + self.plug_heights * 0.5,
        #                                                  device=self.device)
        self.nut_com_quat = self.nut_quat  # always equal
        # self.plug_com_linvel = self.plug_linvel + torch.cross(self.plug_angvel,
        #                                                       (self.plug_com_pos - self.plug_pos),
        #                                                       dim=1)
        self.nut_com_angvel = self.nut_angvel  # always equal

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        pass
