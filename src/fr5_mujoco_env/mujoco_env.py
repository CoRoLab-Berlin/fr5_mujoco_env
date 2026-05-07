# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from .constants import OBJECT_RND_POS_RANGE, OBJECT_RND_ROT_MAX, RECORD_HEIGHT, RECORD_WIDTH

ROBOT_JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow',
    'wrist_1',
    'wrist_2',
    'wrist_3',
]


@dataclass
class RobotConfig:
    """Configuration for robot joints and actuators extracted from MuJoCo model."""

    actuator_ids: np.ndarray
    qvel_indices: np.ndarray
    qpos_indices: np.ndarray
    joint_ids: np.ndarray
    q_des_min: np.ndarray
    q_des_max: np.ndarray
    finger1_id: int
    finger2_id: int
    gripper_qpos_idx: int
    target_body_id: int
    ee_site_id: int


def load_robot_config(model: mujoco.MjModel, ee_site_name: str) -> RobotConfig:
    """Extract robot configuration from MuJoCo model."""
    actuator_ids = np.array(
        [model.actuator(name).id for name in ROBOT_JOINT_NAMES], dtype=np.int32
    )
    qvel_indices = np.array(
        [model.jnt_dofadr[model.joint(n).id] for n in ROBOT_JOINT_NAMES], dtype=np.int32
    )
    qpos_indices = np.array(
        [model.jnt_qposadr[model.joint(n).id] for n in ROBOT_JOINT_NAMES], dtype=np.int32
    )
    joint_ids = np.array([model.joint(n).id for n in ROBOT_JOINT_NAMES], dtype=np.int32)

    gripper_joint_id = model.actuator('act_finger1').trnid[0]

    return RobotConfig(
        actuator_ids=actuator_ids,
        qvel_indices=qvel_indices,
        qpos_indices=qpos_indices,
        joint_ids=joint_ids,
        q_des_min=model.jnt_range[joint_ids, 0],
        q_des_max=model.jnt_range[joint_ids, 1],
        finger1_id=model.actuator('act_finger1').id,
        finger2_id=model.actuator('act_finger2').id,
        gripper_qpos_idx=model.jnt_qposadr[gripper_joint_id],
        target_body_id=model.body('target').id,
        ee_site_id=model.site(ee_site_name).id,
    )


def create_renderer(model: mujoco.MjModel) -> tuple[mujoco.Renderer, mujoco.MjvOption]:
    """Create MuJoCo renderer with default visualization options."""
    renderer = mujoco.Renderer(model, height=RECORD_HEIGHT, width=RECORD_WIDTH)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[0] = 1
    vopt.geomgroup[1] = 0
    vopt.geomgroup[2] = 1
    vopt.geomgroup[3] = 0
    vopt.geomgroup[4] = 0
    vopt.geomgroup[5] = 0
    return renderer, vopt


def reset_environment(model: mujoco.MjModel, data: mujoco.MjData, robot_home_key_id: int) -> None:
    """Reset MuJoCo environment with randomized objects."""
    mujoco.mj_resetDataKeyframe(model, data, robot_home_key_id)
    randomize_environment_objects(model, data)
    mujoco.mj_forward(model, data)


def launch_viewer(
    model: mujoco.MjModel, data: mujoco.MjData, target_body_id: int, show_ui: bool
) -> mujoco.viewer.Handle:
    """Launch MuJoCo passive viewer with tracking camera."""
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=show_ui, show_right_ui=show_ui)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = target_body_id
    viewer.cam.distance = 1.87
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 180
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    viewer.opt.geomgroup[1] = 0

    return viewer


def randomize_environment_objects(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Randomize X/Y positions of pickup_cube and plate, and Z-rotation of pickup_cube."""
    rng = np.random.default_rng()
    for obj_name in ['pickup_cube', 'plate']:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{obj_name}_joint')

        if jnt_id != -1:
            qpos_adr = model.jnt_qposadr[jnt_id]

            # Record old position
            old_x = data.qpos[qpos_adr]
            old_y = data.qpos[qpos_adr + 1]

            # 1. Randomize X and Y positions
            data.qpos[qpos_adr] += rng.uniform(-OBJECT_RND_POS_RANGE, OBJECT_RND_POS_RANGE)
            data.qpos[qpos_adr + 1] += rng.uniform(-OBJECT_RND_POS_RANGE, OBJECT_RND_POS_RANGE)

            # Record new position
            new_x = data.qpos[qpos_adr]
            new_y = data.qpos[qpos_adr + 1]

            print(
                f'[{obj_name}] Position: ({old_x:.4f}, {old_y:.4f}) -> ({new_x:.4f}, {new_y:.4f})'
            )

            # 2. Randomize Z-axis rotation for the cube only
            if 'cube' in obj_name:
                orig_quat = data.qpos[qpos_adr + 3 : qpos_adr + 7].copy()

                # Convert old quaternion to Euler Z (Yaw) in degrees
                w, x, y, z = orig_quat
                old_yaw = np.degrees(
                    np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                )

                # Apply random rotation (0 to 90 degrees)
                theta = rng.uniform(0, OBJECT_RND_ROT_MAX)
                rot_quat = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
                res_quat = np.zeros(4)

                mujoco.mju_mulQuat(res_quat, rot_quat, orig_quat)
                data.qpos[qpos_adr + 3 : qpos_adr + 7] = res_quat

                # Convert new quaternion to Euler Z (Yaw) in degrees
                w_new, x_new, y_new, z_new = res_quat
                new_yaw = np.degrees(
                    np.arctan2(
                        2.0 * (w_new * z_new + x_new * y_new),
                        1.0 - 2.0 * (y_new * y_new + z_new * z_new),
                    )
                )

                print(f'[{obj_name}] Z-Rotation: {old_yaw:.2f}° -> {new_yaw:.2f}°')
