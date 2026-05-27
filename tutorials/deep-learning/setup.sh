#!/bin/bash
# Setup script for NHANES glucose pipeline (uses uv)

set -e

echo "Setting up NHANES glucose pipeline..."

if ! command -v uv &> /dev/null; then
    echo "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "Using uv: $(uv --version)"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.10
else
    echo "Virtual environment already exists"
fi

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Setup complete. Run the pipeline: ./run.sh  or  uv run python run_pipeline.py"
