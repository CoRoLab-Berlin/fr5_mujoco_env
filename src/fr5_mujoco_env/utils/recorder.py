# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

import pathlib
import secrets
import string

import cv2
import h5py
import numpy as np

from ..constants import RECORD_FPS


class DataRecorder:
    def __init__(
        self, instruction: str, save_dir: str = '.', compress_images: bool = True
    ) -> None:
        self.record_fps = RECORD_FPS
        self.record_dt = 1.0 / self.record_fps
        self.save_dir = save_dir
        self.compress_images = compress_images

        self.last_record_time = 0.0
        self.is_recording = False
        self.current_instruction = instruction

        pathlib.Path(self.save_dir).mkdir(exist_ok=True, parents=True)
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self.obs_images_left = []
        self.obs_images_right = []
        self.obs_qpos = []
        self.obs_qvel = []
        self.obs_joint_pos = []
        self.actions = []

    def start_recording(self, instruction: str, start_time: float | None = None) -> None:
        if self.is_recording:
            return

        self.current_instruction = instruction
        self._reset_buffers()
        self.last_record_time = start_time if start_time is not None else 0.0
        self.is_recording = True
        print(f"\n[REC] Recording started. Task: '{self.current_instruction}'")

    def stop_and_save(self) -> None:
        if not self.is_recording:
            return

        self.is_recording = False
        print('\n[REC] Recording stopped. Processing data...')

        if len(self.actions) == 0:
            print('[REC] No data recorded during this episode.')
            return

        alphabet = string.ascii_lowercase + string.digits
        filename = pathlib.Path(self.save_dir) / (
            f'recording_{"".join(secrets.choice(alphabet) for _ in range(5))}.hdf5'
        )
        while filename.exists():
            filename = pathlib.Path(self.save_dir) / (
                f'recording_{"".join(secrets.choice(alphabet) for _ in range(5))}.hdf5'
            )
        print(f'[REC] Saving {len(self.actions)} steps to {filename}...')

        with h5py.File(filename, 'w') as f:
            f.attrs['sim'] = True
            f.attrs['compress'] = self.compress_images  # <-- DYNAMIC ATTRIBUTE

            # Language string definition
            encoded_instruction = np.array([self.current_instruction.encode('utf-8')])
            f.create_dataset('language_raw', data=encoded_instruction)

            # Actions
            f.create_dataset('action', data=np.array(self.actions, dtype=np.float32))

            # Observations
            obs_grp = f.create_group('observations')
            obs_grp.create_dataset('qpos', data=np.array(self.obs_qpos, dtype=np.float32))
            obs_grp.create_dataset('qvel', data=np.array(self.obs_qvel, dtype=np.float32))
            obs_grp.create_dataset(
                'joint_positions', data=np.array(self.obs_joint_pos, dtype=np.float32)
            )

            # --- Image Saving Logic ---
            img_grp = obs_grp.create_group('images')
            num_steps = len(self.actions)

            if self.compress_images:
                # Use variable-length datatype for compressed bytes
                dt = h5py.vlen_dtype(np.uint8)
                left_dset = img_grp.create_dataset('left', (num_steps,), dtype=dt)
                right_dset = img_grp.create_dataset('right', (num_steps,), dtype=dt)

                for i in range(num_steps):
                    # Convert RGB to BGR before JPEG encoding for accurate color subsampling
                    left_bgr = cv2.cvtColor(self.obs_images_left[i], cv2.COLOR_RGB2BGR)
                    right_bgr = cv2.cvtColor(self.obs_images_right[i], cv2.COLOR_RGB2BGR)

                    _, left_enc = cv2.imencode('.jpg', left_bgr)
                    _, right_enc = cv2.imencode('.jpg', right_bgr)

                    # Store flat 1D byte array
                    left_dset[i] = left_enc.flatten()
                    right_dset[i] = right_enc.flatten()
            else:
                # Fallback to uncompressed
                left_frames = np.asarray(self.obs_images_left, dtype=np.uint8)
                right_frames = np.asarray(self.obs_images_right, dtype=np.uint8)
                img_grp.create_dataset('left', data=left_frames, dtype=np.uint8)
                img_grp.create_dataset('right', data=right_frames, dtype=np.uint8)

        print('[REC] Saved successfully!')
        self._reset_buffers()

    def should_record(self, current_sim_time: float) -> bool:
        """Determine if enough simulation time has passed to capture a new frame."""
        if not self.is_recording:
            return False

        if (current_sim_time - self.last_record_time) >= self.record_dt:
            self.last_record_time = current_sim_time
            return True

        return False

    def add_step(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        action: np.ndarray,
    ) -> None:
        """Append a single frame of data to the buffers."""
        self.obs_images_left.append(left_img)
        self.obs_images_right.append(right_img)
        self.obs_qpos.append(qpos)
        self.obs_qvel.append(qvel)
        self.obs_joint_pos.append(qpos)
        self.actions.append(action)
