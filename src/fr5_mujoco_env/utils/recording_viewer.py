# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import cv2
import h5py
import numpy as np


@dataclass(frozen=True)
class StereoRecording:
    instruction: str
    left_frames: np.ndarray
    right_frames: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.left_frames.shape[0])


def _decode_instruction(raw_value: object) -> str:
    if isinstance(raw_value, bytes):
        return raw_value.decode('utf-8')
    return str(raw_value)


def _print_hdf5_tree(group: h5py.Group, indent: str) -> None:
    for name in sorted(group.keys()):
        obj = group[name]
        if isinstance(obj, h5py.Group):
            print(f'{indent}- {name}/')
            _print_hdf5_tree(obj, indent + '  ')
        elif isinstance(obj, h5py.Dataset):
            print(f'{indent}- {name} (dataset, shape={obj.shape}, dtype={obj.dtype})')
        else:
            print(f'{indent}- {name} (datatype, dtype={obj.dtype})')


def _print_recording_header(
    path: pathlib.Path,
    recording: StereoRecording,
    left_exists: bool,
    right_exists: bool,
) -> None:
    height, width = recording.left_frames.shape[1:3]
    print(f'Playing: {path}')
    print(f'Images left/right present: {left_exists}/{right_exists}')
    print(f'Instruction: {recording.instruction or "<empty>"}')
    print(f'Frames: {recording.frame_count}')
    print(f'Resolution: {width}x{height} (W x H)')


def _extract_and_decode_frames(dataset: h5py.Dataset, compressed: bool) -> np.ndarray:
    """Extract frames from an HDF5 dataset, decoding JPEGs to RGB if compressed."""
    frames = []
    num_frames = len(dataset)

    for i in range(num_frames):
        frame_data = dataset[i]

        if compressed:
            # Ensure it is a 1D uint8 array for cv2
            encoded = np.asarray(frame_data, dtype=np.uint8).reshape(-1)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                msg = f'Failed to decode compressed image at index {i}'
                raise ValueError(msg)
            # Convert BGR (OpenCV default) back to RGB (Matplotlib default)
            rgb_frame = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        else:
            # Uncompressed frames are already (H, W, C) RGB uint8
            frames.append(np.asarray(frame_data, dtype=np.uint8))

    return np.stack(frames, axis=0)


def load_recording(recording_path: str | pathlib.Path) -> StereoRecording:
    path = pathlib.Path(recording_path)
    if not path.exists():
        msg = f'Recording file does not exist: {path}'
        raise FileNotFoundError(msg)

    with h5py.File(path, 'r') as h5_file:
        left_exists = 'observations/images/left' in h5_file
        right_exists = 'observations/images/right' in h5_file

        # Check if the dataset was saved with compression
        compressed = h5_file.attrs.get('compress', False)

        try:
            left_dataset = h5_file['observations/images/left']
            right_dataset = h5_file['observations/images/right']

            if not isinstance(left_dataset, h5py.Dataset) or not isinstance(
                right_dataset, h5py.Dataset
            ):
                msg = 'Camera datasets at observations/images/left or right are not datasets.'
                raise TypeError(msg)

            # Decode frames dynamically based on compression flag
            left = _extract_and_decode_frames(left_dataset, compressed)
            right = _extract_and_decode_frames(right_dataset, compressed)

        except KeyError as exc:
            msg = 'Recording is missing camera datasets at observations/images/left or right.'
            raise KeyError(msg) from exc

        instruction = ''
        if 'language_raw' in h5_file:
            language_raw_arr = np.asarray(h5_file['language_raw'])
            if language_raw_arr.shape[0] > 0:
                instruction = _decode_instruction(language_raw_arr[0])

        if left.ndim != 4 or right.ndim != 4:
            msg = (
                'Expected image tensors with 4 dims (T,H,W,C), '
                f'got {left.shape=} and {right.shape=}.'
            )
            raise ValueError(msg)
        if left.shape[0] == 0 or right.shape[0] == 0:
            msg = 'Recording has no frames.'
            raise ValueError(msg)
        if left.shape != right.shape:
            msg = f'Left and right streams differ: {left.shape=} vs {right.shape=}.'
            raise ValueError(msg)

        recording = StereoRecording(
            instruction=instruction,
            left_frames=left,
            right_frames=right,
        )

        _print_recording_header(path, recording, left_exists, right_exists)
        print('HDF5 contents:')
        print('root/')
        _print_hdf5_tree(h5_file, '  ')
        return recording


def find_latest_recording(recordings_dir: str | pathlib.Path = 'recordings') -> pathlib.Path:
    directory = pathlib.Path(recordings_dir)
    if not directory.exists():
        msg = f'Recordings directory does not exist: {directory}'
        raise FileNotFoundError(msg)

    candidates = [
        path
        for pattern in ('recording_*.hdf5', 'recording_*.h5')
        for path in directory.glob(pattern)
        if path.is_file()
    ]
    if not candidates:
        msg = f'No recording_*.hdf5 or recording_*.h5 files found in: {directory}'
        raise FileNotFoundError(msg)

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _make_title(instruction: str, frame_idx: int, frame_count: int) -> str:
    prefix = f'Frame {frame_idx + 1}/{frame_count}'
    if instruction:
        return f'{prefix} | Instruction: {instruction}'
    return prefix


def play_stereo_recording(
    recording: StereoRecording, fps: float = 10.0, loop: bool = True
) -> None:
    if fps <= 0:
        msg = 'fps must be greater than 0.'
        raise ValueError(msg)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.artist import Artist
    except ImportError as exc:
        msg = 'matplotlib is required for playback. Install it with: pip install matplotlib'
        raise RuntimeError(msg) from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    left_axis, right_axis = axes
    left_axis.set_title('Left Camera')
    right_axis.set_title('Right Camera')
    left_axis.axis('off')
    right_axis.axis('off')

    left_artist = left_axis.imshow(recording.left_frames[0])
    right_artist = right_axis.imshow(recording.right_frames[0])
    title = fig.suptitle(_make_title(recording.instruction, 0, recording.frame_count))

    def _update(frame_idx: int) -> tuple[Artist, Artist, Artist]:
        left_artist.set_data(recording.left_frames[frame_idx])
        right_artist.set_data(recording.right_frames[frame_idx])
        title.set_text(_make_title(recording.instruction, frame_idx, recording.frame_count))
        return left_artist, right_artist, title

    animation = FuncAnimation(
        fig,
        _update,
        frames=recording.frame_count,
        interval=1000.0 / fps,
        blit=False,
        repeat=loop,
    )

    fig._recording_animation_ref = animation
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    plt.show()
