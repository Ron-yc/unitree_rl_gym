
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)


class G1OneMeterRobot(G1Robot):
    def _init_buffers(self):
        super()._init_buffers()

        self.start_pos = self.root_states[:, :2].clone()

        self.forward_distance = torch.zeros(
            self.num_envs,
            dtype = torch.float,
            device = self.device,
        )

        self.remaining_distance = torch.zeros_like(
            self.forward_distance
        )

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)

        self.root_states[env_ids,7:13] = 0.0

        env_ids_int32 = env_ids.to(
            dtype = torch.int32
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim;
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self.env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        if len(env_ids) == 0:
            return 

        self.strat_pos[env_ids] = self.root_states[env_ids, :2]
        self.forward_distance[env_ids] = 0.0
        self.remaining_distance[env_ids] = self.cfg.commands.target_distance
        self.commands[env_ids] = 0.0

    def _resample_commands(self, env_ids):
        self.commands[env_ids] = 0.0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        self.forward_distance = self.root_states[:, 0] - self.start_pos[:, 0]

        self.remaining_distance = self.cfg.commands.target_distance - self.forward_distance

        elapsed_time = self.episode_length_buf.float() * self.dt

        ramp_up = torch.clamp(
            elapsed_time / self.cfg.commands.ramp_up_time,
            min = 0.0,
            max = 1.0,
        )

        ramp_down = torch.clamp(
            self.remaining_distance / self.cfg.commands.slow_down_distance,
            min = 0.0,
            max = 1.0
        )

        target_speed = self.cfg.commands.max_speed * torch.minimum(ramp_down,ramp_up)

        target_speed = torch.where(
            self.remaining_distance > 0.0,
            target_speed,
            torch.zeros_like(target_speed),
        )

        self.commands[:, 0] = target_speed
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = 0.0
        self.commands[:, 3] = 0.0

    def _reward_target_position(self):
        position_error = self.remaining_distance

        return torch.exp(
            -torch.square(position_error) / 0.02
        )

    def _reward_lateral_position(self):
        lateral_distance = self.root_states[:, 1] - self.start_pos[:, 1]

        return torch.square(lateral_distance)

    def _reward_stop_velocity(self):
        near_target = torch.abs(self.remaining_distance) < 0.15

        linear_velocity = torch.sum(
            torch.square(self.root_states[:, 7:10]),
            dim = 1,
        )

        angular_velocity = torch.sum(
            torch.square(self.root_states[:, 10:13]), 
            dim=1,  
        )

        return (linear_velocity + 0.25 * angular_velocity) * near_target.float()

    def _reward_overshoot(self):
        overshoot = torch.clamp(
            self.forward_distance - self.cfg.commands.target_distance,
            min = 0.0,
        )

        return torch.square(overshoot)

    def _reward_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0

        moving = self.commands[:, 0] > 0.05

        gait_score = torch.zeros(
            self.num_envs,
            dtype = torch.float,
            device = self.device,
        )

        for foot_index in range(self.feet_num):
            is_stance = self.leg_phase[:, foot_index] < 0.55    #小于支撑大于摆动

            gait_score += ~(
                contact[:, foot_index] ^ is_stance 
            )
            standing_score = torch.sum(
                contact.float(),
                dim = 1,
            )

            return torch.where(
                moving,
                gait_score,
                standing_score,
            )

    def _reward_feet_swing_height(self):
        contact = (
            torch.norm(
                self.contact_forces[:, self.feet_indices, :3],
                dim = 2
            ) > 1.0
        )

        position_error = (
            torch.square(
                self.feet_pos[:, :, 2] - 0.08
            )
            * ~contact
        )

        moving = self.commands[:, 0] > 0.05

        return torch.sum(position_error, dim=1) * moving.float()