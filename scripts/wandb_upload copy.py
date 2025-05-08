"""
Utility for uploading files to Weights & Biases (W&B).
This can be used for checkpoint files or final results.
"""
import argparse
import logging
import os
from pathlib import Path

import wandb

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def upload_to_wandb(
    file_path,
    entity=None,
    project="MLE_BENCH",
    run_name=None,
    run_id=None,
    description=None,
    job_type="checkpoint",
    tags=None,
    notes=None,
    reinit=False,
    save_code=False,
):
    """
    Upload a file to Weights & Biases.

    Args:
        file_path (str): Path to the file to upload
        entity (str, optional): W&B entity name. Defaults to None (uses WANDB_ENTITY env var or default user).
        project (str, optional): W&B project name. Defaults to "MLE_BENCH".
        run_name (str, optional): Name for the W&B run. Defaults to the filename if None.
        run_id (str, optional): Specific run ID to use. If None, creates a new run.
        description (str, optional): Description of the file. Defaults to None.
        job_type (str, optional): Job type tag. Defaults to "checkpoint".
        tags (list, optional): List of tags for the run. Defaults to None.
        notes (str, optional): Notes for the run. Defaults to None.
        reinit (bool, optional): Whether to reinitialize W&B for a new run. Defaults to False.
        save_code (bool, optional): Whether to save code alongside the upload. Defaults to False.

    Returns:
        str: URL to the uploaded file in W&B
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # If run_name is not provided, use the file's name
    if run_name is None:
        run_name = file_path.stem

    # Get entity from environment variable if not provided
    if entity is None:
        entity = os.environ.get("WANDB_ENTITY")

    logger.info(f"Uploading {file_path} to W&B project {project} as run {run_name}")

    # Check if wandb is already initialized
    if wandb.run is not None and not reinit:
        logger.info(f"Using existing W&B run: {wandb.run.name} (ID: {wandb.run.id})")
        run = wandb.run
    else:
        # Initialize a new W&B run
        run = wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            id=run_id,
            job_type=job_type,
            tags=tags,
            notes=notes,
            reinit=reinit,
            save_code=save_code,
        )
        logger.info(f"Initialized new W&B run: {run.name} (ID: {run.id})")

    # Upload the file
    artifact = wandb.Artifact(
        name=file_path.stem,
        type="checkpoint" if "checkpoint" in file_path.name else "result",
        description=description or f"Upload of {file_path.name}",
    )
    artifact.add_file(str(file_path), name=file_path.name)

    # Log the artifact
    run.log_artifact(artifact)
    logger.info(f"Uploaded {file_path.name} as artifact {artifact.name}")

    # Get the URL for the artifact
    artifact_url = f"https://wandb.ai/{run.entity}/{run.project}/artifacts/{artifact.type}/{artifact.name}:{artifact.version}"
    logger.info(f"Artifact URL: {artifact_url}")

    # Don't finish the run - allow for multiple uploads to the same run
    # If reinit=True was used, the caller should call wandb.finish() when done

    return artifact_url


def upload_checkpoint(
    checkpoint_path, wandb_entity=None, wandb_project="MLE_BENCH", run_group=None
):
    """
    Convenience function to upload a checkpoint file to W&B.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        wandb_entity (str, optional): W&B entity. Defaults to WANDB_ENTITY env var.
        wandb_project (str, optional): W&B project. Defaults to "MLE_BENCH".
        run_group (str, optional): Run group name to use in the run name. If None, extracts from path if possible.

    Returns:
        str: URL to the uploaded artifact
    """
    checkpoint_path = Path(checkpoint_path)

    # Extract run group from path if not provided
    if run_group is None:
        # Try to extract from path - assume checkpoint is in a runs/<run_group>/... structure
        # or the checkpoint filename contains the run group
        path_parts = checkpoint_path.parts
        if "runs" in path_parts:
            run_group_idx = path_parts.index("runs") + 1
            if run_group_idx < len(path_parts):
                run_group = path_parts[run_group_idx]
        else:
            # Try to extract from filename pattern like checkpoint_TIMESTAMP_RUNGROUP_NNN.zip
            filename_parts = checkpoint_path.stem.split("_")
            if len(filename_parts) >= 3:
                # Assume format is checkpoint_TIMESTAMP_RUNGROUP or similar
                run_group = (
                    "_".join(filename_parts[1:-1]) if len(filename_parts) > 3 else filename_parts[1]
                )

    # Build a run name with format: checkpoint_<run_group>_<filename>
    run_name = f"checkpoint_{run_group or 'unknown'}"

    # Upload with appropriate tags
    return upload_to_wandb(
        file_path=checkpoint_path,
        entity=wandb_entity,
        project=wandb_project,
        run_name=run_name,
        job_type="checkpoint",
        tags=["checkpoint", run_group or "unknown"],
        description=f"Checkpoint for run group {run_group or 'unknown'}",
        reinit=True,  # Create a new run for each checkpoint
        save_code=False,  # Don't save code for checkpoint uploads
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a file to Weights & Biases")
    parser.add_argument("file_path", help="Path to file to upload")
    parser.add_argument("--entity", help="W&B entity name")
    parser.add_argument("--project", default="MLE_BENCH", help="W&B project name")
    parser.add_argument("--run-name", help="Name for the W&B run")
    parser.add_argument("--run-group", help="Run group name")
    parser.add_argument("--job-type", default="checkpoint", help="Job type tag")
    parser.add_argument("--description", help="Description for the artifact")

    args = parser.parse_args()

    if args.job_type == "checkpoint":
        upload_checkpoint(
            args.file_path,
            wandb_entity=args.entity,
            wandb_project=args.project,
            run_group=args.run_group,
        )
    else:
        upload_to_wandb(
            args.file_path,
            entity=args.entity,
            project=args.project,
            run_name=args.run_name,
            job_type=args.job_type,
            description=args.description,
        )
