"""
dareg CLI entry point

Provides subcommand routing to the existing pipeline modules:
  dareg motion      - 4D motion tracking registration
  dareg register    - Single pair image registration
  dareg postprocess - Post-processing (STL, video, interpolation)
"""

import sys


USAGE = """\
usage: dareg [-h] [--version] {motion,register,postprocess} ...

DAREG - Medical image registration replicating MIRTK using deepali

subcommands:
  motion       4D motion tracking registration pipeline
  register     Single-pair image registration (rigid/affine/FFD)
  postprocess  Post-processing: STL generation, temporal interpolation, video

examples:
  dareg motion --image4d dynamic.nii --static static.nii --seg seg.nii.gz -o ./output
  dareg register --source moving.nii.gz --target fixed.nii.gz
  dareg postprocess --output_dir ./output --stl --video
"""

COMMANDS = {"motion", "register", "postprocess"}


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


def _get_version():
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
