"""
DAREG Monitor Entry Point

Standalone progress monitor for watching CLI registration runs.

Usage:
    python -m dareg.monitor --watch ./output_dir
    python -m dareg.monitor --watch ./output_dir --refresh 500

This allows monitoring registration progress from a separate terminal
while the main registration runs in another terminal or in the background.
"""

from .gui.monitor import main

if __name__ == "__main__":
    main()
