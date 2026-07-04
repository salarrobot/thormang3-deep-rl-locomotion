from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSETS_DIR / "thormang3_scene.xml"

LEG_JOINTS = [
    "l_leg_hip_y", "l_leg_hip_r", "l_leg_hip_p",
    "l_leg_kn_p", "l_leg_an_p", "l_leg_an_r",
    "r_leg_hip_y", "r_leg_hip_r", "r_leg_hip_p",
    "r_leg_kn_p", "r_leg_an_p", "r_leg_an_r",
]

RC_KEYS = ("rc_alive", "rc_forward", "rc_upright", "rc_height",
           "rc_single", "rc_air", "rc_ctrl", "rc_action_rate", "rc_torque",
           "rc_slide", "rc_lateral", "rc_yaw_rate", "rc_joint_vel",
           "ep_speed", "ep_dist", "ep_pushes")


@dataclass
class WalkConfig:
    model_path: str = str(SCENE_XML)
    frame_skip: int = 8
    action_scale: float = 0.4
    action_filter: float = 0.6
    target_speed: float = 0.8
    w_alive: float = 1.0
    w_forward: float = 2.0
    w_upright: float = 0.4
    w_height: float = 0.4
    w_single: float = 0.15
    w_air: float = 1.0
    w_ctrl: float = 0.02
    w_action_rate: float = 0.05
    w_torque: float = 1e-5
    w_slide: float = 0.1
    w_lateral: float = 0.25
    w_yaw_rate: float = 0.15
    w_joint_vel: float = 2e-4
    push_enable: bool = True
    push_interval: tuple = (3.0, 6.0)
    push_duration: float = 0.15
    push_force: tuple = (20.0, 70.0)
    rand_friction: tuple = (0.7, 1.3)
    rand_mass: tuple = (0.9, 1.1)
    obs_noise: float = 1.0
    min_height: float = 0.55
    max_height: float = 1.05
    min_upright: float = 0.5
    max_steps: int = 1000
    noise_qpos: float = 0.03
    noise_qvel: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)


class ThormangWalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: WalkConfig | None = None,
                 render_mode: str | None = None,
                 width: int = 1280, height: int = 720,
                 camera: str = "track"):
        self.cfg = cfg or WalkConfig()
        self.render_mode = render_mode
        self._render_size = (height, width)
        self._camera = camera
        self._renderer: mujoco.Renderer | None = None

        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)
        assert self.model.nu == len(LEG_JOINTS)

        self._qadr = np.array([self.model.joint(j).qposadr[0]
                               for j in LEG_JOINTS])
        self._vadr = np.array([self.model.joint(j).dofadr[0]
                               for j in LEG_JOINTS])
        self._act_ids = np.array([self.model.actuator(j).id
                                  for j in LEG_JOINTS])
        self._pelvis_id = self.model.body("pelvis_link").id
        self._floor_geom = self.model.geom("floor").id
        self._foot_geoms = np.array([self.model.geom("l_foot_col").id,
                                     self.model.geom("r_foot_col").id])
        self._foot_bodies = np.array([self.model.body("l_leg_foot_link").id,
                                      self.model.body("r_leg_foot_link").id])

        self._nominal_mass = self.model.body_mass.copy()
        self._nominal_inertia = self.model.body_inertia.copy()
        self._nominal_friction = self.model.geom_friction[
            self._floor_geom].copy()

        key = self.model.keyframe("stand")
        self._qpos0 = key.qpos.copy()
        self._stand_q = self._qpos0[self._qadr].copy()
        self._stand_z = float(self._qpos0[2])
        self._ctrl_low = self.model.actuator_ctrlrange[self._act_ids, 0]
        self._ctrl_high = self.model.actuator_ctrlrange[self._act_ids, 1]

        self.dt = self.model.opt.timestep * self.cfg.frame_skip
        n_act = len(LEG_JOINTS)
        n_obs = 1 + 3 + 3 + 3 + n_act * 3 + 2
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,),
                                                np.float32)
        self._noise_std = np.concatenate([
            [0.005],
            np.full(3, 0.02),
            np.full(3, 0.10),
            np.full(3, 0.15),
            np.full(n_act, 0.01),
            np.full(n_act, 0.30),
            np.zeros(n_act),
            np.zeros(2),
        ]).astype(np.float32)

        self._prev_action = np.zeros(n_act, np.float32)
        self._filt_target = self._stand_q.copy()
        self._air_time = np.zeros(2)
        self._contact_prev = np.ones(2, bool)
        self._steps = 0
        self._push_timer = 0
        self._push_left = 0
        self._push_vec = np.zeros(3)
        self._ep_sums = {k: 0.0 for k in RC_KEYS}

    def _base_rotations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        quat = self.data.qpos[3:7]
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, quat)
        gravity = np.zeros(3)
        mujoco.mju_rotVecQuat(gravity, np.array([0, 0, -1.0]), neg)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data,
                                 mujoco.mjtObj.mjOBJ_BODY, self._pelvis_id,
                                 vel, 1)
        return gravity, vel[3:6], vel[0:3]

    def _foot_contacts(self) -> np.ndarray:
        touching = np.zeros(2, np.float32)
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact.geom[i]
            for f in range(2):
                if ((g1 == self._foot_geoms[f] and g2 == self._floor_geom) or
                        (g2 == self._foot_geoms[f] and g1 == self._floor_geom)):
                    touching[f] = 1.0
        return touching

    def _foot_xy_speed(self, foot: int) -> float:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data,
                                 mujoco.mjtObj.mjOBJ_BODY,
                                 self._foot_bodies[foot], vel, 0)
        return float(np.linalg.norm(vel[3:5]))

    def _upright(self) -> float:
        up = np.zeros(3)
        mujoco.mju_rotVecQuat(up, np.array([0, 0, 1.0]), self.data.qpos[3:7])
        return float(up[2])

    def _get_obs(self) -> np.ndarray:
        gravity, linvel, angvel = self._base_rotations()
        obs = np.concatenate([
            [self.data.qpos[2]],
            gravity,
            linvel,
            angvel,
            self.data.qpos[self._qadr] - self._stand_q,
            self.data.qvel[self._vadr],
            self._prev_action,
            self._foot_contacts(),
        ]).astype(np.float32)
        if self.cfg.obs_noise > 0:
            obs += (self.np_random.standard_normal(obs.shape)
                    * self._noise_std * self.cfg.obs_noise).astype(np.float32)
        return obs

    def _sample_push_timer(self) -> int:
        lo, hi = self.cfg.push_interval
        return int(self.np_random.uniform(lo, hi) / self.dt)

    def _randomize_dynamics(self) -> None:
        c = self.cfg
        s = self.np_random.uniform(*c.rand_mass)
        self.model.body_mass[:] = self._nominal_mass * s
        self.model.body_inertia[:] = self._nominal_inertia * s
        f = self.np_random.uniform(*c.rand_friction)
        self.model.geom_friction[self._floor_geom] = self._nominal_friction
        self.model.geom_friction[self._floor_geom, 0] *= f

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_dynamics()
        c = self.cfg
        qpos = self._qpos0.copy()
        qpos[self._qadr] += self.np_random.uniform(
            -c.noise_qpos, c.noise_qpos, len(self._qadr))
        qpos[2] += self.np_random.uniform(0.0, 0.005)
        qvel = self.np_random.uniform(-c.noise_qvel, c.noise_qvel,
                                      self.model.nv)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[self._act_ids] = qpos[self._qadr]
        mujoco.mj_forward(self.model, self.data)

        self._prev_action[:] = 0.0
        self._filt_target = qpos[self._qadr].copy()
        self._air_time[:] = 0.0
        self._contact_prev = self._foot_contacts().astype(bool)
        self._steps = 0
        self._push_timer = self._sample_push_timer()
        self._push_left = 0
        self._push_vec[:] = 0.0
        self._ep_sums = {k: 0.0 for k in RC_KEYS}
        return self._get_obs(), {}

    def _update_push(self) -> None:
        c = self.cfg
        self.data.xfrc_applied[self._pelvis_id, :] = 0.0
        if not c.push_enable:
            return
        if self._push_left > 0:
            self._push_left -= 1
            self.data.xfrc_applied[self._pelvis_id, 0:3] = self._push_vec
            if self._push_left == 0:
                self._push_timer = self._sample_push_timer()
        else:
            self._push_timer -= 1
            if self._push_timer <= 0:
                angle = self.np_random.uniform(0, 2 * np.pi)
                force = self.np_random.uniform(*c.push_force)
                self._push_vec = np.array([np.cos(angle), np.sin(angle), 0.0
                                           ]) * force
                self._push_left = max(1, int(c.push_duration / self.dt))
                self._ep_sums["ep_pushes"] += 1

    def step(self, action: np.ndarray):
        c = self.cfg
        action = np.clip(np.asarray(action, np.float32).flatten(), -1.0, 1.0)
        raw_target = np.clip(self._stand_q + action * c.action_scale,
                             self._ctrl_low, self._ctrl_high)
        self._filt_target = (c.action_filter * raw_target
                             + (1 - c.action_filter) * self._filt_target)
        self.data.ctrl[self._act_ids] = self._filt_target
        self._update_push()
        mujoco.mj_step(self.model, self.data, nstep=c.frame_skip)
        self._steps += 1

        z = float(self.data.qpos[2])
        upright = self._upright()
        gravity, linvel, angvel = self._base_rotations()

        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        heading = np.array([R[0], R[3]])
        norm = np.linalg.norm(heading)
        heading = heading / norm if norm > 1e-6 else np.array([1.0, 0.0])
        v_fwd = float(self.data.qvel[0:2] @ heading)

        contacts = self._foot_contacts()
        touchdowns: list[tuple[int, float, float]] = []
        r_air, slide = 0.0, 0.0
        for f in range(2):
            if contacts[f]:
                if not self._contact_prev[f]:
                    r_air += float(np.clip(self._air_time[f], 0.0, 0.5)
                                   - 0.25)
                    xy = self.data.geom_xpos[self._foot_geoms[f]][:2]
                    touchdowns.append((f, float(xy[0]), float(xy[1])))
                self._air_time[f] = 0.0
                slide += self._foot_xy_speed(f)
            else:
                self._air_time[f] += self.dt
        self._contact_prev = contacts.astype(bool)
        single_support = contacts.sum() == 1 and v_fwd > 0.2

        joint_vel = self.data.qvel[self._vadr]
        torque = self.data.actuator_force[self._act_ids]
        comp = {
            "rc_alive": c.w_alive,
            "rc_forward": c.w_forward
            * min(v_fwd, 1.25 * c.target_speed) / c.target_speed,
            "rc_upright": c.w_upright
            * float(np.clip((upright - 0.7) / 0.3, 0.0, 1.0)),
            "rc_height": c.w_height
            * float(np.clip((z - c.min_height)
                            / (self._stand_z - c.min_height), 0.0, 1.0)),
            "rc_single": c.w_single * float(single_support),
            "rc_air": c.w_air * r_air,
            "rc_ctrl": -c.w_ctrl * float(action @ action),
            "rc_action_rate": -c.w_action_rate
            * float(np.sum((action - self._prev_action) ** 2)),
            "rc_torque": -c.w_torque * float(torque @ torque),
            "rc_slide": -c.w_slide * slide,
            "rc_lateral": -c.w_lateral * float(linvel[1] ** 2),
            "rc_yaw_rate": -c.w_yaw_rate * float(angvel[2] ** 2),
            "rc_joint_vel": -c.w_joint_vel * float(joint_vel @ joint_vel),
        }
        reward = float(sum(comp.values()))

        fallen = (z < c.min_height or z > c.max_height
                  or upright < c.min_upright)
        terminated = bool(fallen)
        truncated = self._steps >= c.max_steps and not terminated

        self._prev_action = action
        for k, v in comp.items():
            self._ep_sums[k] += v
        self._ep_sums["ep_dist"] = float(
            np.linalg.norm(self.data.qpos[0:2]))
        self._ep_sums["ep_speed"] = self._ep_sums["ep_dist"] / (
            self._steps * self.dt)

        info = {
            "reward_components": comp,
            "v_forward": v_fwd,
            "height": z,
            "upright": upright,
            "push": (tuple(self._push_vec[:2])
                     if self._push_left > 0 else None),
            "touchdowns": touchdowns,
        }
        if terminated or truncated:
            info.update(self._ep_sums)
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        for attempt in range(2):
            try:
                if self._renderer is None:
                    self._renderer = mujoco.Renderer(self.model,
                                                     *self._render_size)
                self._renderer.update_scene(self.data, camera=self._camera)
                return self._renderer.render()
            except Exception:
                if attempt:
                    raise
                try:
                    self._renderer.close()
                except Exception:
                    pass
                self._renderer = None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id="Thormang3Walk-v1", entry_point=ThormangWalkEnv)
