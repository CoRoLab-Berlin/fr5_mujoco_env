import pathlib
import sys
from typing import Any

import h5py
import numpy as np

# =====================================================================
# Inspection Configuration
# =====================================================================

# Path to a specific recording .hdf5 file.
# Set to None to automatically find the newest recording.
RECORDING_FILE = None

# Directory to search if RECORDING_FILE is None.
RECORDINGS_DIR = pathlib.Path('recordings')

# =====================================================================


def find_latest_recording(recordings_dir: str | pathlib.Path) -> pathlib.Path:
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


def _format_attr_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return repr(value.item())
        return f'array shape={value.shape}, dtype={value.dtype}'
    if isinstance(value, np.generic):
        return repr(value.item())
    return repr(value)


def _print_attributes(attrs: h5py.AttributeManager, indent: str) -> None:
    if len(attrs) == 0:
        return
    print(f'{indent}attributes:')
    for key in sorted(attrs.keys()):
        value = _format_attr_value(attrs[key])
        print(f'{indent}  - {key}: {value}')


def _dataset_summary(dataset: h5py.Dataset) -> str:
    shape = dataset.shape
    dtype = dataset.dtype
    summary = [f'shape={shape}', f'dtype={dtype}']

    try:
        length = len(dataset)
    except TypeError:
        length = None
    if length is not None:
        summary.append(f'len={length}')

    vlen_base = h5py.check_dtype(vlen=dtype)
    if vlen_base is not None:
        summary.append(f'vlen={vlen_base}')
    else:
        est_bytes = dataset.size * dtype.itemsize
        summary.append(f'est_bytes={est_bytes}')

    if dataset.chunks is not None:
        summary.append(f'chunks={dataset.chunks}')
    if dataset.compression is not None:
        summary.append(f'compression={dataset.compression}')
    if dataset.compression_opts is not None:
        summary.append(f'compression_opts={dataset.compression_opts}')
    if dataset.maxshape is not None:
        summary.append(f'maxshape={dataset.maxshape}')

    return ', '.join(summary)


def _print_hdf5_tree(group: h5py.Group, indent: str) -> None:
    for name in sorted(group.keys()):
        obj = group[name]
        if isinstance(obj, h5py.Group):
            print(f'{indent}- {name}/ (keys={len(obj)})')
            _print_attributes(obj.attrs, indent + '  ')
            _print_hdf5_tree(obj, indent + '  ')
        elif isinstance(obj, h5py.Dataset):
            summary = _dataset_summary(obj)
            print(f'{indent}- {name} (dataset, {summary})')
            _print_attributes(obj.attrs, indent + '  ')
        else:
            print(f'{indent}- {name} (datatype, dtype={obj.dtype})')


def _guess_episode_length(h5_file: h5py.File) -> int | None:
    candidates = [
        'action',
        'observations/qpos',
        'observations/images/left',
        'observations/images/right',
    ]
    for path in candidates:
        if path in h5_file:
            try:
                return len(h5_file[path])
            except TypeError:
                continue
    return None


def _resolve_recording_path(argv: list[str]) -> pathlib.Path:
    if len(argv) > 1:
        return pathlib.Path(argv[1])
    if RECORDING_FILE is not None:
        return pathlib.Path(RECORDING_FILE)
    return find_latest_recording(RECORDINGS_DIR)


def main() -> None:
    try:
        recording_path = _resolve_recording_path(sys.argv)
        if not recording_path.exists():
            msg = f'Recording file does not exist: {recording_path}'
            raise FileNotFoundError(msg)

        file_size = recording_path.stat().st_size
        print(f'File: {recording_path}')
        print(f'Size (bytes): {file_size}')

        with h5py.File(recording_path, 'r') as h5_file:
            length = _guess_episode_length(h5_file)
            if length is not None:
                print(f'Episode length (steps): {length}')

            print('Root attributes:')
            _print_attributes(h5_file.attrs, '  ')

            print('HDF5 contents:')
            print('root/')
            _print_hdf5_tree(h5_file, '  ')

    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f'\nError: {exc}')
        sys.exit(1)


if __name__ == '__main__':
    main()
