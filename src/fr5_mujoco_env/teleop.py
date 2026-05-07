# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .utils import xbox


@dataclass
class TeleopCommand:
    x_raw: float = 0.0
    y_raw: float = 0.0
    z_cmd: float = 0.0
    tilt_pitch: float = 0.0
    tilt_roll: float = 0.0
    twist_yaw: float = 0.0
    orbit_left: bool = False
    orbit_right: bool = False
    gripper_rt: float = 0.0
    record_start: bool = False
    record_stop: bool = False


class AutoTeleop:
    def __init__(self) -> None:
        self.timeout = 3.0
        self.recording = False
        self.init_time = time.time()
        self.start_time = time.time()
        self.stage = 0
        self.dist_threshold = 0.015

        # New state variables for time-delays and static waypoints
        self.stage_start_time = 0.0
        self.saved_target_pos = np.zeros(3)

        print('AutoTeleop initialized. Using direct coordinate assignment.')

    def get_command(self, data: mujoco.MjData) -> TeleopCommand | None:  # noqa: C901
        # INITIALIZATION PERIOD
        if time.time() - self.init_time < self.timeout and not self.recording:
            return None

        # NOT RECORDING -> START RECORDING
        if not self.recording:
            print('[AutoTeleop] Starting recording...')
            self.recording = True
            self.start_time = time.time()
            self.stage = 1
            return TeleopCommand(record_start=True)

        cmd = TeleopCommand()

        # Base position of the target marker (needed for local offsets)
        target_body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
        target_base_pos = data.model.body_pos[target_body_id]

        # Get the current global position of the Robot's End Effector
        tcp_pos = data.site('tool_center_point').xpos

        # ----------------------------------------------------
        # STATE MACHINE
        # ----------------------------------------------------
        if self.stage == 1:
            cube_pos = data.body('pickup_cube').xpos
            desired_global_pos = cube_pos + np.array([0.0, 0.0, 0.1])

            local_qpos = desired_global_pos - target_base_pos
            data.joint('slide_x').qpos[0] = local_qpos[0]
            data.joint('slide_y').qpos[0] = local_qpos[1]
            data.joint('slide_z').qpos[0] = local_qpos[2]

            cube_quat = data.joint('pickup_cube_joint').qpos[3:7]
            data.joint('cube_ball').qpos[:] = cube_quat

            distance = np.linalg.norm(tcp_pos - desired_global_pos)

            if distance < self.dist_threshold:
                print('[AutoTeleop] Arrived above cube --> Stage 2 (Inside cube)')
                self.stage = 2

        elif self.stage == 2:
            cube_pos = data.body('pickup_cube').xpos
            desired_global_pos = cube_pos + np.array([0.0, 0.0, -0.01])

            local_qpos = desired_global_pos - target_base_pos
            data.joint('slide_x').qpos[0] = local_qpos[0]
            data.joint('slide_y').qpos[0] = local_qpos[1]
            data.joint('slide_z').qpos[0] = local_qpos[2]

            cube_quat = data.joint('pickup_cube_joint').qpos[3:7]
            data.joint('cube_ball').qpos[:] = cube_quat

            distance = np.linalg.norm(tcp_pos - desired_global_pos)

            if distance < self.dist_threshold:
                print('[AutoTeleop] Arrived inside cube --> Stage 3 (Gripping)')
                self.stage = 3
                self.stage_start_time = time.time()

        elif self.stage == 3:
            cmd.gripper_rt = 1.0

            if time.time() - self.stage_start_time > 0.5:
                print('[AutoTeleop] Gripping cube --> Stage 4 (Lifting)')
                self.stage = 4
                self.saved_target_pos = tcp_pos.copy() + np.array([0.0, 0.0, 0.1])

        elif self.stage == 4:
            cmd.gripper_rt = 1.0

            desired_global_pos = self.saved_target_pos
            local_qpos = desired_global_pos - target_base_pos
            data.joint('slide_x').qpos[0] = local_qpos[0]
            data.joint('slide_y').qpos[0] = local_qpos[1]
            data.joint('slide_z').qpos[0] = local_qpos[2]

            distance = np.linalg.norm(tcp_pos - desired_global_pos)

            if distance < self.dist_threshold:
                print('[AutoTeleop] Lifted cube --> Stage 5 (Above plate)')
                self.stage = 5

        elif self.stage == 5:
            cmd.gripper_rt = 1.0

            plate_pos = data.body('plate').xpos
            desired_global_pos = plate_pos + np.array([0.0, 0.0, 0.1])

            local_qpos = desired_global_pos - target_base_pos
            data.joint('slide_x').qpos[0] = local_qpos[0]
            data.joint('slide_y').qpos[0] = local_qpos[1]
            data.joint('slide_z').qpos[0] = local_qpos[2]

            data.joint('cube_ball').qpos[:] = np.array([1.0, 0.0, 0.0, 0.0])

            distance = np.linalg.norm(tcp_pos - desired_global_pos)

            if distance < self.dist_threshold:
                print(
                    '[AutoTeleop] Arrived above plate --> Stage 6 (Lowering)'
                )
                self.stage = 6

        elif self.stage == 6:
            cmd.gripper_rt = 1.0

            plate_pos = data.body('plate').xpos
            desired_global_pos = plate_pos + np.array([0.0, 0.0, 0.05])

            local_qpos = desired_global_pos - target_base_pos
            data.joint('slide_x').qpos[0] = local_qpos[0]
            data.joint('slide_y').qpos[0] = local_qpos[1]
            data.joint('slide_z').qpos[0] = local_qpos[2]

            data.joint('cube_ball').qpos[:] = np.array([1.0, 0.0, 0.0, 0.0])

            distance = np.linalg.norm(tcp_pos - desired_global_pos)

            if distance < self.dist_threshold:
                print('[AutoTeleop] Lowered cube onto plate --> Stage 7 (Release)')
                self.stage = 7
                self.stage_start_time = time.time()

        elif self.stage == 7:
            cmd.gripper_rt = 0.0

            plate_pos = data.body('plate').xpos
            desired_global_pos = plate_pos + np.array([0.0, 0.0, 0.05])

            local_qpos = desired_global_pos - target_base_pos
            data.joint('slide_x').qpos[0] = local_qpos[0]
            data.joint('slide_y').qpos[0] = local_qpos[1]
            data.joint('slide_z').qpos[0] = local_qpos[2]

            data.joint('cube_ball').qpos[:] = np.array([1.0, 0.0, 0.0, 0.0])
            if time.time() - self.stage_start_time > 0.25:
                print(
                    '[AutoTeleop] Gripper released --> Stage 8 (Retreat)'
                )
                self.stage = 8
                self.stage_start_time = time.time()

        elif self.stage == 8:
            cmd.gripper_rt = 0.0

            plate_pos = data.body('plate').xpos
            desired_global_pos = plate_pos + np.array([0.0, 0.0, 0.15])

            local_qpos = desired_global_pos - target_base_pos
            data.joint('slide_x').qpos[0] = local_qpos[0]
            data.joint('slide_y').qpos[0] = local_qpos[1]
            data.joint('slide_z').qpos[0] = local_qpos[2]

            data.joint('cube_ball').qpos[:] = np.array([1.0, 0.0, 0.0, 0.0])

            if time.time() - self.stage_start_time > 1.0:
                print(
                    '[AutoTeleop] Retreat finished --> Stage 9 (Return Home)'
                )
                self.stage = 9

        elif self.stage == 9:
            cmd.gripper_rt = 0.0

            data.joint('slide_x').qpos[0] = 0.0
            data.joint('slide_y').qpos[0] = 0.0
            data.joint('slide_z').qpos[0] = 0.0

            data.joint('cube_ball').qpos[:] = np.array([1.0, 0.0, 0.0, 0.0])

            distance = np.linalg.norm(tcp_pos - target_base_pos)

            if distance < self.dist_threshold:
                print('[AutoTeleop] Returned to home position.')
                self.stage = 10

        elif self.stage == 10:
            if self.recording:
                print('[AutoTeleop] Sequence finished successfully. Stopping recording...')
                self.recording = False
                return TeleopCommand(record_stop=True)

        return cmd

    def print_controls(self) -> None:
        print('[AutoTeleop] No manual controls. Simulation will run in Auto mode.')

    def close(self) -> None:
        pass


class XboxTeleop:
    def __init__(
        self,
        deadzone: float = 0.1,
        movement_speed: float = 0.8,
        right_stick_sens: float = 0.5,
        dpad_sens: float = 0.5,
    ) -> None:
        self.deadzone = deadzone
        self.movement_speed = movement_speed
        self.right_stick_sens = right_stick_sens
        self.dpad_sens = dpad_sens

        print('Connecting to Xbox controller...')
        try:
            self.joy = xbox.Joystick()
            self.is_connected = True
            print('Controller connected!')
        except OSError as e:
            print(f'Failed to connect controller: {e}. Simulation will run in View-Only mode.')
            self.is_connected = False

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) <= self.deadzone:
            return 0.0
        if self.deadzone >= 1.0:
            return 0.0
        return (1.0 if value > 0 else -1.0) * (
            (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        )

    def print_controls(self) -> None:
        """Print the control mapping if the controller is connected."""
        if not self.is_connected:
            return

        print('- Left Stick: Move X / Y (Camera Relative)')
        print('- Right Stick: Move Z (Up/Down) / Twist Z (Left/Right)')
        print('- D-Pad: Rotate Target Pitch/Roll')
        print('- LB / RB: Orbit Camera')
        print('- Right Trigger (RT): Close Gripper & Change Color')
        print('- A Button: Reset to Home & Start Recording')
        print('- B Button: Stop & Save Recording')

    def get_command(self, data: mujoco.MjData) -> TeleopCommand | None:
        """Return the parsed controller state, or None if disconnected."""
        if not self.is_connected:
            return None

        lx, ly = self.joy.leftStick()
        rx, ry = self.joy.rightStick()
        lb, rb = self.joy.leftBumper(), self.joy.rightBumper()
        rt = self.joy.rightTrigger()
        a, b = self.joy.A(), self.joy.B()

        try:
            dpad_up, dpad_down = self.joy.dpadUp(), self.joy.dpadDown()
            dpad_left, dpad_right = self.joy.dpadLeft(), self.joy.dpadRight()
        except AttributeError:
            dx, dy = self.joy.dpad()
            dpad_up, dpad_down = (1 if dy > 0 else 0), (1 if dy < 0 else 0)
            dpad_right, dpad_left = (1 if dx > 0 else 0), (1 if dx < 0 else 0)

        # Process deadzones and sensitivities
        x_raw = self._apply_deadzone(lx) * self.movement_speed
        y_raw = self._apply_deadzone(ly) * self.movement_speed
        z_cmd = self._apply_deadzone(ry) * self.right_stick_sens
        rz_raw = self._apply_deadzone(rx) * 15 * self.right_stick_sens

        tilt_pitch = (dpad_up - dpad_down) * 1.5 * self.dpad_sens
        tilt_roll = (dpad_right - dpad_left) * 1.5 * self.dpad_sens
        twist_yaw = rz_raw * 1.5

        return TeleopCommand(
            x_raw=x_raw,
            y_raw=y_raw,
            z_cmd=z_cmd,
            tilt_pitch=tilt_pitch,
            tilt_roll=tilt_roll,
            twist_yaw=twist_yaw,
            orbit_left=bool(lb),
            orbit_right=bool(rb),
            gripper_rt=rt,
            record_start=bool(a),
            record_stop=bool(b),
        )

    def close(self) -> None:
        if self.is_connected:
            self.joy.close()
