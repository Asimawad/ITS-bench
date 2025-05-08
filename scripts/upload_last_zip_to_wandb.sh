#!/bin/bash
# Simple wrapper script to upload the latest zip file to wandb
# This can be added to your workflow without modifying run_agent.py

set -e

echo "Finding and uploading the latest runs zip file to W&B..."

# Activate the virtual environment if needed
if [ -d ".aide-ds" ]; then
    echo "Activating virtual environment..."
    source .aide-ds/bin/activate
fi

# Default search directory is the current directory
SEARCH_DIR="${1:-.}"

# Run the Python script to find and upload the latest zip
python scripts/upload_to_wandb.py --zip-dir "$SEARCH_DIR"

echo "Upload to W&B complete!" 