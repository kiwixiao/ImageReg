#!/usr/bin/env bash
#
# DAREG setup script
# Automatically creates a Python environment and installs dareg.
#   - Uses conda if available (env name: image-reg)
#   - Falls back to python venv (dir: .venv/)
#
# Usage:
#   bash setup.sh           # standard install
#   bash setup.sh --all     # include optional STL/video extras
#   bash setup.sh --dev     # editable install for development

set -e

ENV_NAME="image-reg"
VENV_DIR=".venv"
EXTRAS=""
EDITABLE=""

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --all)  EXTRAS="[all]" ;;
        --dev)  EDITABLE="-e" ;;
        --help|-h)
            echo "Usage: bash setup.sh [--all] [--dev]"
            echo "  --all   Install with optional extras (STL, video)"
            echo "  --dev   Editable install (changes apply without reinstalling)"
            exit 0
            ;;
    esac
done

# Detect conda
if command -v conda &>/dev/null; then
    echo "==> conda detected"

    # Check if env already exists
    if conda env list | grep -qw "$ENV_NAME"; then
        echo "==> conda env '$ENV_NAME' already exists, activating..."
    else
        echo "==> Creating conda env '$ENV_NAME' (Python 3.12)..."
        conda create -y -n "$ENV_NAME" python=3.12
    fi

    # Activate
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    echo "==> Activated conda env: $ENV_NAME"

    # Install
    echo "==> Installing dareg..."
    pip install $EDITABLE ".${EXTRAS}"

    echo ""
    echo "Done! To use dareg, run:"
    echo "  conda activate $ENV_NAME"
    echo "  dareg --help"

else
    echo "==> conda not found, using python venv"

    # Find python
    PYTHON=""
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    done

    if [ -z "$PYTHON" ]; then
        echo "Error: No python found. Install Python 3.9+ first."
        exit 1
    fi

    # Check python version >= 3.9
    PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
        echo "Error: Python >= 3.9 required (found $PY_VERSION)"
        exit 1
    fi

    echo "==> Using $PYTHON ($PY_VERSION)"

    # Create venv if it doesn't exist
    if [ -d "$VENV_DIR" ]; then
        echo "==> venv '$VENV_DIR' already exists, activating..."
    else
        echo "==> Creating venv '$VENV_DIR'..."
        $PYTHON -m venv "$VENV_DIR"
    fi

    # Activate
    source "$VENV_DIR/bin/activate"
    echo "==> Activated venv: $VENV_DIR"

    # Install
    echo "==> Installing dareg..."
    pip install $EDITABLE ".${EXTRAS}"

    echo ""
    echo "Done! To use dareg, run:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  dareg --help"
fi
