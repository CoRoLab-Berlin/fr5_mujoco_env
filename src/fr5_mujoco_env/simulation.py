# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from .constants import KD, KP, RECORD_HEIGHT, RECORD_WIDTH
from .controller import DiffIKController
from .mujoco_env import randomize_environment_objects
from .teleop import AutoTeleop, XboxTeleop
from .utils import DataRecorder


class InteractiveSimulation:
    def __init__(
        self,
        scene_path: str,
        instruction: str,
        record_control: str = 'auto',
        recorder_save_dir: str = '.',
        compress_images: bool = True,
        deadzone: float = 0.1,
        movement_speed: float = 0.8,
        orbit_speed: float = 0.1,
        right_stick_sens: float = 0.5,
        dpad_sens: float = 0.5,
    ) -> None:
        self.scene_path = scene_path
        self.orbit_speed = orbit_speed
        self.instruction = instruction

        print(f'Loading {self.scene_path}...')
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        self.robot_home_key_id = self.model.key('robot_home').id
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.robot_home_key_id)
        mujoco.mj_forward(self.model, self.data)

        # --- Sub-Modules ---
        if record_control == 'manual':
            self.teleop = XboxTeleop(deadzone, movement_speed, right_stick_sens, dpad_sens)
        else:
            self.teleop = AutoTeleop()

        self.recorder = DataRecorder(
            instruction=self.instruction,
            save_dir=recorder_save_dir,
            compress_images=compress_images,
        )

        self.robot_joint_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow',
            'wrist_1',
            'wrist_2',
            'wrist_3',
        ]

        self.diffik = DiffIKController(
            model=self.model,
            data=self.data,
            site_name='tool_center_point',
            target_name='target',
            joint_names=self.robot_joint_names,
        )

        # --- Object & Actuator IDs ---
        self.geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'target_geom')
        self.target_body_id = self.model.body('target').id

        self.motor_x_id = self.model.actuator('motor_x').id
        self.motor_y_id = self.model.actuator('motor_y').id
        self.motor_z_id = self.model.actuator('motor_z').id

        self.finger1_id = self.model.actuator('act_finger1').id
        self.finger2_id = self.model.actuator('act_finger2').id

        self.actuator_ids = np.array(
            [self.model.actuator(name).id for name in self.robot_joint_names]
        )
        self.qvel_indices = np.array(
            [self.model.jnt_dofadr[self.model.joint(n).id] for n in self.robot_joint_names]
        )
        self.qpos_indices = np.array(
            [self.model.jnt_qposadr[self.model.joint(n).id] for n in self.robot_joint_names]
        )

        self.gripper_qpos_idx = self.model.jnt_qposadr[self.model.actuator('act_finger1').trnid[0]]
        self.gripper_qvel_idx = self.model.jnt_dofadr[self.model.actuator('act_finger1').trnid[0]]

        # Enable Gravity Compensation
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name and ('link' in name or 'finger' in name):
                self.model.body_gravcomp[i] = 1.0

        # --- Offscreen Renderer Setup ---
        self.renderer = mujoco.Renderer(self.model, height=RECORD_HEIGHT, width=RECORD_WIDTH)
        self.vopt = mujoco.MjvOption()
        self.vopt.geomgroup[0] = 1
        self.vopt.geomgroup[1] = 0
        self.vopt.geomgroup[2] = 1
        self.vopt.geomgroup[3] = 0
        self.vopt.geomgroup[4] = 0
        self.vopt.geomgroup[5] = 0
        self.vopt.sitegroup[0] = 0
        self.vopt.sitegroup[1] = 0
        self.vopt.sitegroup[2] = 0
        self.vopt.sitegroup[3] = 0
        self.vopt.sitegroup[4] = 0

    def _reset_environment_to_home(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.robot_home_key_id)
        randomize_environment_objects(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _capture_and_record_state(self, q_des: np.ndarray, gripper_cmd: float) -> None:
        self.renderer.update_scene(self.data, camera='left', scene_option=self.vopt)
        left_img = self.renderer.render()

        self.renderer.update_scene(self.data, camera='right', scene_option=self.vopt)
        right_img = self.renderer.render()

        qpos_6 = self.data.qpos[self.qpos_indices]
        qvel_6 = self.data.qvel[self.qvel_indices]
        g_pos = self.data.qpos[self.gripper_qpos_idx]
        g_vel = self.data.qvel[self.gripper_qvel_idx]

        qpos_7 = np.concatenate([qpos_6, [g_pos]])
        qvel_7 = np.concatenate([qvel_6, [g_vel]])

        action_10 = np.zeros(10)
        action_10[:6] = q_des
        action_10[6] = gripper_cmd

        self.recorder.add_step(left_img.copy(), right_img.copy(), qpos_7, qvel_7, action_10)

    def run(self) -> None:  # noqa: C901
        self.teleop.print_controls()
        print('Close the viewer window to exit.')

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.target_body_id
            viewer.cam.distance = 1.87
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 180
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            recordings_count = 0

            while viewer.is_running():
                step_start = time.time()
                gripper_cmd = 0.0

                # --- 1. Get Controller Input ---
                cmd = self.teleop.get_command(self.data)

                if cmd:
                    if cmd.record_start and not self.recorder.is_recording:
                        self._reset_environment_to_home()
                        self.recorder.start_recording(
                            instruction=self.instruction,
                            start_time=self.data.time,
                        )
                        q_des_start = self.diffik.compute_q_des(self.data)
                        self._capture_and_record_state(q_des_start, gripper_cmd=0.0)
                        viewer.sync()
                        time.sleep(0.2)
                    elif cmd.record_stop:
                        self.recorder.stop_and_save()
                        recordings_count += 1
                        print(f'[INFO] Session recordings count: {recordings_count}')
                        time.sleep(0.2)

                    az_rad = math.radians(viewer.cam.azimuth)
                    forward_x, forward_y = math.cos(az_rad), math.sin(az_rad)
                    right_x, right_y = math.sin(az_rad), -math.cos(az_rad)

                    # Move Target Position
                    if getattr(cmd, 'use_world_frame', False):
                        self.data.ctrl[self.motor_x_id] = cmd.x_raw
                        self.data.ctrl[self.motor_y_id] = cmd.y_raw
                    else:
                        self.data.ctrl[self.motor_x_id] = (cmd.x_raw * right_x) + (
                            cmd.y_raw * forward_x
                        )
                        self.data.ctrl[self.motor_y_id] = (cmd.x_raw * right_y) + (
                            cmd.y_raw * forward_y
                        )

                    self.data.ctrl[self.motor_z_id] = cmd.z_cmd

                    # Move Target Rotation
                    self.data.xfrc_applied[self.target_body_id, 3] = (
                        -cmd.tilt_pitch * forward_y
                    ) - (cmd.tilt_roll * right_y)
                    self.data.xfrc_applied[self.target_body_id, 4] = (
                        cmd.tilt_pitch * forward_x
                    ) + (cmd.tilt_roll * right_x)
                    self.data.xfrc_applied[self.target_body_id, 5] = cmd.twist_yaw

                    if cmd.orbit_left:
                        viewer.cam.azimuth -= self.orbit_speed
                    if cmd.orbit_right:
                        viewer.cam.azimuth += self.orbit_speed
                    if self.geom_id != -1:
                        self.model.geom_rgba[self.geom_id] = [
                            1.0 - cmd.gripper_rt,
                            cmd.gripper_rt,
                            0.0,
                            0.2,
                        ]

                    gripper_cmd = cmd.gripper_rt * 0.04
                else:
                    self.data.ctrl[self.motor_x_id] = 0.0
                    self.data.ctrl[self.motor_y_id] = 0.0
                    self.data.ctrl[self.motor_z_id] = 0.0
                    self.data.xfrc_applied[self.target_body_id, 3:6] = 0.0

                self.data.ctrl[self.finger1_id] = gripper_cmd
                self.data.ctrl[self.finger2_id] = gripper_cmd

                # --- 2. Solve DiffIK ---
                q_des = self.diffik.compute_q_des(self.data)

                # --- 3. PD Control ---
                q_curr = self.data.qpos[self.qpos_indices]
                qvel_curr = self.data.qvel[self.qvel_indices]
                tau = KP * (q_des - q_curr) - KD * qvel_curr
                self.data.ctrl[self.actuator_ids] = tau

                # --- 4. Physics Step & Sync ---
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # --- 5. Record Data ---
                if self.recorder.should_record(self.data.time):
                    self._capture_and_record_state(q_des, gripper_cmd)

                # Real-time loop maintenance
                while (time.time() - step_start) < self.model.opt.timestep:
                    pass

        self.teleop.close()
        self.recorder.stop_and_save()
        print('Exited successfully.')
