import pathlib
import sys

from fr5_mujoco_env.constants import RECORD_FPS
from fr5_mujoco_env.utils import (
    find_latest_recording,
    load_recording,
    play_stereo_recording,
)

# =====================================================================
# Playback Configuration
# =====================================================================

# Path to a specific recording .hdf5 file.
# Set to None to automatically find the newest recording.
RECORDING_FILE = None

# Directory to search if RECORDING_FILE is None.
RECORDINGS_DIR = pathlib.Path('recordings')

# Playback frames per second. Defaults to the imported RECORD_FPS.
PLAYBACK_FPS = RECORD_FPS

# Loop the playback when the last frame is reached.
LOOP_PLAYBACK = True

# =====================================================================


def main() -> None:
    try:
        # Determine the file path based on your configuration
        recording_path = (
            pathlib.Path(RECORDING_FILE)
            if RECORDING_FILE is not None
            else find_latest_recording(RECORDINGS_DIR)
        )

        recording = load_recording(recording_path)

        play_stereo_recording(recording, fps=PLAYBACK_FPS, loop=LOOP_PLAYBACK)

    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f'\nError: {exc}')
        sys.exit(1)


if __name__ == '__main__':
    main()
