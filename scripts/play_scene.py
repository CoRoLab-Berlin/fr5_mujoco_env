from fr5_mujoco_env.simulation import InteractiveSimulation


def main() -> None:
    output_dir = 'recordings'
    record_control = 'auto'  # 'auto', 'manual'

    scene_path = 'tasks/scene.xml'
    instruction = 'Place the yellow cube on the plate.'

    sim = InteractiveSimulation(
        scene_path=scene_path,
        instruction=instruction,
        record_control=record_control,
        recorder_save_dir=str(output_dir),
        compress_images=True,
        deadzone=0.2,
        movement_speed=0.8,
        orbit_speed=0.1,
        right_stick_sens=0.5,
        dpad_sens=5.0,
    )
    sim.run()


if __name__ == '__main__':
    main()
