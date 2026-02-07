"""
dareg CLI entry point

Provides subcommand routing to the existing pipeline modules:
  dareg motion      - 4D motion tracking registration
  dareg register    - Single pair image registration
  dareg postprocess - Post-processing (STL, video, interpolation)
  dareg monitor     - Live progress monitor GUI
  dareg config      - Registration configuration GUI
"""

import sys


USAGE = """\
usage: dareg [-h] [--version] {motion,register,postprocess,monitor,config} ...

DAREG - Medical image registration replicating MIRTK using deepali

subcommands:
  motion       4D motion tracking registration pipeline
  register     Single-pair image registration (rigid/affine/FFD)
  postprocess  Post-processing: STL generation, temporal interpolation, video
  monitor      Live progress monitor GUI (watches output directory)
  config       Interactive registration configuration GUI

examples:
  dareg motion --image4d dynamic.nii --static static.nii --seg seg.nii.gz -o ./output
  dareg register --source moving.nii.gz --target fixed.nii.gz
  dareg postprocess --output_dir ./output --stl --video
  dareg monitor --watch ./motion_output
  dareg config
  dareg config --config registration_config.yaml
  dareg config --json-config config.json
"""

COMMANDS = {"motion", "register", "postprocess", "monitor", "config"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    if sys.argv[1] in ("-V", "--version"):
        print(f"dareg {_get_version()}")
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"dareg: unknown command '{command}'")
        print(USAGE)
        sys.exit(1)

    # Replace sys.argv so the subcommand's own argparse sees the right args
    sys.argv = [f"dareg {command}"] + sys.argv[2:]

    if command == "motion":
        from .main_motion import main as motion_main
        sys.exit(motion_main() or 0)

    elif command == "register":
        from .main import main as register_main
        sys.exit(register_main() or 0)

    elif command == "postprocess":
        from .main_postprocess import main as postprocess_main
        postprocess_main()  # calls sys.exit internally on error

    elif command == "monitor":
        from .gui.monitor import main as monitor_main
        sys.exit(monitor_main() or 0)

    elif command == "config":
        import argparse
        parser = argparse.ArgumentParser(
            prog="dareg config",
            description="Launch interactive registration configuration GUI",
        )
        parser.add_argument(
            "--config", "-c", type=str,
            help="Pre-populate GUI from a YAML config file",
        )
        parser.add_argument(
            "--json-config", "-j", type=str,
            help="Pre-populate GUI from a JSON config file",
        )
        config_args = parser.parse_args()

        initial_config = None
        if config_args.config:
            from .main_motion import _yaml_config_to_gui_dict
            initial_config = _yaml_config_to_gui_dict(config_args.config)
        elif config_args.json_config:
            import json
            with open(config_args.json_config, 'r') as f:
                initial_config = json.load(f)

        from .gui.config_gui import launch_config_gui
        launch_config_gui(initial_config)
        sys.exit(0)


def _get_version():
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
