# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

import mujoco
import numpy as np

from fr5_mujoco_env.constants import KD, KP, MAX_POS_ERROR

from .constants import SMOOTHING_ALPHA


def compute_pd_torque(
    target_positions: np.ndarray,
    current_positions: np.ndarray,
    current_velocities: np.ndarray,
    bias_compensation: np.ndarray,
) -> np.ndarray:
    """Compute joint torques using PD control."""
    limited_target = np.clip(
        target_positions,
        current_positions - MAX_POS_ERROR,
        current_positions + MAX_POS_ERROR,
    )
    torque = KP * (limited_target - current_positions) - KD * current_velocities
    torque += bias_compensation
    return torque


class DiffIKController:
    """
    Single-file pedagogical implementations of common robotics controllers in MuJoCo.

    Source: https://github.com/kevinzakka/mjctrl?tab=readme-ov-file

    <MODIFIED VERSION>
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_name: str,
        target_name: str,
        joint_names: list[str],
        damping: float = 1e-4,
    ) -> None:
        self.model = model
        self.damping = damping
        self.integration_dt = SMOOTHING_ALPHA

        # IDs
        self.site_id = model.site(site_name).id
        self.target_body_id = model.body(target_name).id
        self.dof_ids = np.array([model.joint(name).id for name in joint_names])
        self.qpos_indices = np.array([model.jnt_qposadr[j] for j in self.dof_ids])

        # Pre-allocate arrays for speed
        self.jac = np.zeros((6, model.nv))
        self.diag = self.damping * np.eye(6)
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.target_quat = np.zeros(4)

        # --- Calculate Initial Rotational Offset ---
        # Ensures the gripper rotates relative to its starting orientation
        target_quat_init = np.zeros(4)
        gripper_quat_init = np.zeros(4)
        target_quat_init_conj = np.zeros(4)
        self.quat_offset = np.zeros(4)

        mujoco.mju_mat2Quat(target_quat_init, data.body(self.target_body_id).xmat)
        mujoco.mju_mat2Quat(gripper_quat_init, data.site(self.site_id).xmat)
        mujoco.mju_negQuat(target_quat_init_conj, target_quat_init)
        mujoco.mju_mulQuat(self.quat_offset, target_quat_init_conj, gripper_quat_init)

    def compute_q_des(self, data: mujoco.MjData) -> np.ndarray:
        """Compute the desired joint positions to reach the target."""
        # 1. Position Error
        self.error_pos[:] = data.body(self.target_body_id).xpos - data.site(self.site_id).xpos

        # 2. Orientation Error (incorporating initial offset)
        target_quat_curr = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat_curr, data.body(self.target_body_id).xmat)
        mujoco.mju_mulQuat(self.target_quat, target_quat_curr, self.quat_offset)

        mujoco.mju_mat2Quat(self.site_quat, data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, self.target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        # 3. Solve Jacobian
        mujoco.mj_jacSite(self.model, data, self.jac[:3], self.jac[3:], self.site_id)
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)

        # 4. Integrate to find new joint positions
        q_full = data.qpos.copy()
        mujoco.mj_integratePos(self.model, q_full, dq, self.integration_dt)
        q_des = q_full[self.qpos_indices]

        # 5. Enforce Joint Limits
        np.clip(
            q_des,
            self.model.jnt_range[self.dof_ids, 0],
            self.model.jnt_range[self.dof_ids, 1],
            out=q_des,
        )

        return q_des
