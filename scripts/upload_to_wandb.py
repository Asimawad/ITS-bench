#!/usr/bin/env python3
"""
Upload the latest zipped runs file to Weights & Biases.
This script can be called separately without modifying run_agent.py.

Usage:
  python scripts/upload_to_wandb.py [--zip-dir DIRECTORY] [--project PROJECT_NAME] [--entity ENTITY_NAME]
"""

import argparse
import glob
import os
import time
from datetime import datetime
from pathlib import Path

import wandb

try:
    from environment.config import load_cfg  # revise this later

    cfg = load_cfg()
    WANDB_ENTITY = cfg.wandb.entity if hasattr(cfg.wandb, "entity") else "asim_awad"
    WANDB_PROJECT = cfg.wandb.project if hasattr(cfg.wandb, "project") else "MLE_BENCH"
except (ImportError, AttributeError):
    # Default values if config loading fails
    WANDB_ENTITY = "asim_awad"
    WANDB_PROJECT = "MLE_BENCH"


def find_latest_zip(directory):
    """Find the latest zip file in the given directory."""
    # Get all zip files in the directory
    zip_files = glob.glob(os.path.join(directory, "**/*.zip"), recursive=True)

    if not zip_files:
        return None

    # Sort by modification time (newest first)
    latest_zip = max(zip_files, key=os.path.getmtime)
    return latest_zip


def upload_to_wandb(zip_file_path, project=WANDB_PROJECT, entity=WANDB_ENTITY, run_name=None):
    """Upload the zip file to wandb."""
    if not os.path.exists(zip_file_path):
        print(f"Error: File {zip_file_path} does not exist.")
        return False

    # Extract filename from path
    zip_filename = os.path.basename(zip_file_path)

    # If no run name provided, use the zip filename without extension
    if not run_name:
        run_name = f"runs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Uploading {zip_filename} to W&B project {entity}/{project}")
    try:
        # Initialize a new W&B run
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            job_type="backup",
            tags=["backup", "runs", "automatic"],
        )

        # Log the zip file as an artifact
        artifact = wandb.Artifact(
            name=f"runs_backup_{int(time.time())}",
            type="backup",
            description=f"Backup of runs directory: {zip_filename}",
        )
        artifact.add_file(zip_file_path, name=zip_filename)
        run.log_artifact(artifact)

        # Add some metadata
        run.config.update(
            {
                "backup_time": datetime.now().isoformat(),
                "zip_file": zip_filename,
                "zip_size_mb": os.path.getsize(zip_file_path) / (1024 * 1024),
            }
        )

        print(f"Successfully uploaded {zip_filename} to W&B")
        run.finish()
        return True

    except Exception as e:
        print(f"Error uploading to W&B: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload the latest zip file to W&B")
    parser.add_argument(
        "--zip-dir",
        type=str,
        default=".",
        help="Directory to search for zip files (default: current directory)",
    )
    parser.add_argument(
        "--specific-zip",
        type=str,
        help="Path to a specific zip file to upload (overrides --zip-dir)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=WANDB_PROJECT,
        help=f"W&B project name (default: {WANDB_PROJECT})",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=WANDB_ENTITY,
        help=f"W&B entity name (default: {WANDB_ENTITY})",
    )
    parser.add_argument(
        "--run-name", type=str, help="W&B run name (default: auto-generated based on timestamp)"
    )

    args = parser.parse_args()

    if args.specific_zip:
        zip_file_path = args.specific_zip
        if not os.path.exists(zip_file_path):
            print(f"Error: Specified zip file {zip_file_path} does not exist.")
            return
    else:
        # Find the latest zip file
        zip_file_path = find_latest_zip(args.zip_dir)
        if not zip_file_path:
            print(f"Error: No zip files found in {args.zip_dir}")
            return

    print(f"Found zip file: {zip_file_path}")
    upload_to_wandb(zip_file_path, project=args.project, entity=args.entity, run_name=args.run_name)


if __name__ == "__main__":
    main()
