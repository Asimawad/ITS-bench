from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List  # Keep List for type hinting args

from dotenv import dotenv_values

from mlebench.utils import purple

# Import dotenv_values - Assume .shared_env is in the parent directory of THIS script's directory
# e.g., if run_local.py is in /path/to/repo/environment/, .shared_env is in /path/to/repo/
try:
    shared_env_path = Path(__file__).parent.parent.resolve() / ".shared_env"
    CONSTANTS = dotenv_values(shared_env_path)
except FileNotFoundError:
    logging.error(f"'.shared_env' file not found at {shared_env_path}. Please create it.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Failed to load .shared_env file: {e}")
    sys.exit(1)


# --- Load Default AIDE Configuration from YAML File ---
# Assumes the base aide config template is at project root / environment / aide_config.yaml
# project root is two levels up from this script's directory
try:
    repo_root = Path(__file__).resolve().parents[1]
    aide_config_template_path = repo_root / "environment" / "aide_config.yaml"

    if not aide_config_template_path.exists():
        raise FileNotFoundError(f"AIDE config template not found: {aide_config_template_path}")

    import yaml

    with open(aide_config_template_path, "r", encoding="utf-8") as f:
        DEFAULT_AIDE_CONFIG = yaml.safe_load(f)

except FileNotFoundError as e:
    logging.error(f"Configuration file error: {e}")
    sys.exit(1)
except ImportError:
    logging.error("PyYAML library not found. Please install it: `pip install PyYAML`")
    sys.exit(1)
except Exception as e:
    logging.error(f"Failed to load or process AIDE config template: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1)


# --- Helper function to flatten a nested dictionary ---
def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # Handle boolean values explicitly as aide might expect "true" or "false"
            if isinstance(v, bool):
                value_str = str(v).lower()
            elif v is None:
                value_str = "null"  # Or handle None as needed, maybe skip?
            else:
                value_str = str(v)

            items.append((new_key, value_str))  # Store as string
    return dict(items)


if TYPE_CHECKING:
    from agents.registry import Agent
    from mlebench.registry import Competition


def run_locally(
    competition: "Competition",
    agent: "Agent",
    run_dir: Path,
    port: int,  # Unique port for this run's grading server
    logger: logging.Logger,
    main_logger=logging.Logger,
    retain_workspace: bool = False,
) -> Path:
    """
    Execute `agent` on `competition` locally.

    Simulates a simple container environment structure within `run_dir` (specifically `/home`).
    Configures the agent by parsing a base YAML config and agent kwargs,
    then passing them as command-line arguments to start.sh/aide.

    Args:
        competition: The competition to run.
        agent: The agent to run.
        run_dir: Path to the directory where all assets associated with the run are stored.
        port: The specific network port the grading server should listen on.
        logger: Logger instance for the run.
        retain_workspace: If True, the full temporary workspace is kept.

    Returns:
        Path to the run_dir.
    """
    logger.info(
        f"Starting local execution for competition {competition.id} with agent {agent.name}"
    )
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Port assigned: {port}")

    work = run_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    # --- 1. Setup Simulated Environment Directories ---
    logger.info("Setting up simulated environment directories...")
    ch = work / "home"
    ch.mkdir(parents=True, exist_ok=True)

    # Create standard directories
    for dir_name in ["submission", "logs", "code", "agent", "workspaces", "data"]:
        dir_path = ch / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")

    # Simulate the /private/data mount point
    simulated_private_data_root = work / "private"
    simulated_private_data_root.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created private data root: {simulated_private_data_root}")

    # --- 2. Copy Competition Data ---
    logger.info("Copying competition data...")
    simulated_public_data_dest = ch / "data"
    logger.info(
        f"Copying public data from {competition.public_dir} to {simulated_public_data_dest}"
    )
    shutil.copytree(competition.public_dir, simulated_public_data_dest, dirs_exist_ok=True)

    if competition.private_dir.exists():
        private_target_simulated = (
            simulated_private_data_root / "data" / competition.id / "prepared" / "private"
        )
        logger.info(
            f"Copying private data from {competition.private_dir} to {private_target_simulated}"
        )
        private_target_simulated.mkdir(parents=True, exist_ok=True)
        shutil.copytree(competition.private_dir, private_target_simulated, dirs_exist_ok=True)
    else:
        logger.warning(f"Private data directory not found: {competition.private_dir}")

    # --- 3. Generate AIDE Command-Line Arguments from Config and Agent Kwargs ---
    logger.info("Generating AIDE command-line arguments...")
    aide_config_dict = {}
    try:
        aide_config_dict = DEFAULT_AIDE_CONFIG.copy()
        logger.debug("Successfully loaded default AIDE config")
    except Exception as e:
        logger.error(f"Error copying default AIDE config: {e}")
        logger.debug(traceback.format_exc())
        logger.warning("Using empty config as fallback")

    # Path to data *as seen by the agent* (relative to HOME_DIR)
    aide_config_dict[
        "data_dir"
    ] = f"${{HOME_DIR}}/{simulated_public_data_dest.relative_to(ch).as_posix()}"  # e.g. ${HOME_DIR}/data
    # start.sh creates full_instructions.txt in AGENT_DIR
    aide_config_dict["desc_file"] = f"${{AGENT_DIR}}/full_instructions.txt"  # Match start.sh logic

    # --- 4. Create Symlinks for Submission Files ---
    logger.info("Creating symlinks for submission files...")
    workspaces_dir = ch / "workspaces"
    logs_dir = ch / "logs"

    # Create symlinks for each competition workspace
    for workspace in workspaces_dir.iterdir():
        if workspace.is_dir():
            # Find the best submission file
            best_submission = workspace / "best_submission" / "submission.csv"
            if best_submission.exists():
                # Create corresponding logs directory
                log_subdir = logs_dir / workspace.name
                log_subdir.mkdir(exist_ok=True)

                # Create symlink
                submission_link = log_subdir / "submission.csv"
                if submission_link.exists():
                    submission_link.unlink()
                submission_link.symlink_to(best_submission)
                logger.info(f"Created symlink: {submission_link} -> {best_submission}")

    # --- 5. Run the Agent ---
    try:
        # e (now partially updated) default config into key=value strings
        flattened_aide_config = flatten_dict(aide_config_dict)
        # Remove keys that start.sh already sets explicitly on the aide command line
        # (like data_dir, desc_file, agent.code.model) to avoid duplicates or conflicts
        keys_set_in_start_sh = ["data_dir", "desc_file", "agent.code.model"]  # Add others if needed
        aide_cli_args = [
            f"{k}={v}"
            for k, v in flattened_aide_config.items()
            if k not in keys_set_in_start_sh
            and v != "null"  # Filter out keys handled by start.sh and nulls
        ]

        # Convert agent.kwargs into key=value or --key value arguments
        # These will override the YAML defaults if they have the same keys
        agent_override_args = []
        if agent.kwargs_type == "argparse":
            for key, value in agent.kwargs.items():
                # Format as --key value
                agent_override_args += [f"--{key}", str(value)]
        elif agent.kwargs_type == "omegaconf":
            for key, value in agent.kwargs.items():
                # Format as key=value
                agent_override_args += [f"{key}={value}"]
        else:
            logger.warning(
                f"Unknown agent kwargs_type: {agent.kwargs_type}. Skipping passing agent.kwargs."
            )

        # Combine default config args and agent override args
        # Agent overrides should come LAST so they take precedence
        final_aide_args = aide_cli_args + agent_override_args

        # --- 6. Environment Variables Setup ---
        env = os.environ.copy()  # Inherit environment from the parent process
        env.setdefault("TIME_LIMIT_SECS", "3600")  # Set TIME_LIMIT_SECS as used by start.sh

        # Use hardcoded standard environment variable names and local simulated paths
        env.update(
            {
                "HOME_DIR": str(ch),  # Local path to simulated /home
                "SUBMISSION_DIR": str(
                    ch / "submission"
                ),  # Local path to simulated /home/submission
                "LOGS_DIR": str(ch / "logs"),  # Local path to simulated /home/logs
                "CODE_DIR": str(ch / "code"),  # Local path to simulated /home/code
                "AGENT_DIR": str(ch / "agent"),  # Local path to simulated /home/agent
                "COMPETITION_ID": competition.id,
                "MBX_GRADE_PORT": str(
                    port
                ),  # Pass the assigned unique port via MBX_GRADE_PORT env var
                "PRIVATE_DATA_DIR": str(
                    simulated_private_data_root.resolve()
                ),  # Local path to simulated /private
                # You might need to add other env vars your agent expects here
            }
        )

        # Ensure paths assigned to env vars are absolute and resolved
        for key in [
            "HOME_DIR",
            "SUBMISSION_DIR",
            "LOGS_DIR",
            "CODE_DIR",
            "AGENT_DIR",
            "PRIVATE_DATA_DIR",
        ]:
            if key in env and env[key]:
                try:
                    env[key] = str(Path(env[key]).resolve())
                except Exception as e:
                    logger.warning(f"Could not resolve path for env var {key}: {env[key]} - {e}")

        # --- 7. Start Grading Server ---
        logger.info(purple(f"Starting grading server on port {port}…"))
        server_log_path = work / "grading_server.log"
        server_log_file = None
        server = None

        try:
            server_log_file = server_log_path.open("w", encoding="utf-8")
            repo_root = Path(__file__).resolve().parents[1]
            server_script = repo_root / "environment" / "grading_server.py"
            assert server_script.exists(), f"Server script missing: {server_script}"

            logger.debug(
                f"Running server: {sys.executable} {server_script} --port {port} from CWD {work}"
            )
            server = subprocess.Popen(
                [sys.executable, str(server_script), "--port", str(port)],
                cwd=work,  # Server runs from the main run_dir
                env=env,  # Pass the environment (might contain COMPETITION_ID etc. needed by server)
                stdout=server_log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Wait for server to become ready
            ready = False
            logger.info(f"Waiting for server on port {port}...")
            timeout_seconds = 60
            start_time = time.monotonic()
            while time.monotonic() - start_time < timeout_seconds:
                if server.poll() is not None:
                    server_log_file.close()
                    server_log = server_log_path.read_text(encoding="utf-8", errors="ignore")
                    raise RuntimeError(
                        f"Server exited unexpectedly (code {server.poll()}) on port {port}. Logs: {server_log[-1000:]}..."
                    )  # Log last part of output
                try:
                    import http.client

                    conn = http.client.HTTPConnection("localhost", port, timeout=1)
                    conn.request("GET", "/health")
                    response = conn.getresponse()
                    if response.status == 200:
                        ready = True
                        break
                    conn.close()
                except ConnectionRefusedError:
                    pass  # Server not ready, keep looping
                except Exception as e:
                    logger.debug(f"Server check failed on port {port}: {e}. Retrying...")
                time.sleep(1)

            if not ready:
                server_log_file.close()
                server_log = server_log_path.read_text(encoding="utf-8", errors="ignore")
                raise RuntimeError(
                    f"Server did not start on port {port} within {timeout_seconds}s. Logs: {server_log[-1000:]}..."
                )

            logger.info(purple(f"Grading server on port {port} is ready."))

            # --- 8. Run Agent Start Script ---
            # This section is inside the main try block, after server starts
            # Copy agent start.sh into its simulated directory (/home/agent)
            agent_src_dir = agent.agents_dir / agent.name
            start_script_name = "start.sh"
            start_script_src = agent_src_dir / start_script_name
            start_script_dest = (
                ch / "agent" / start_script_name
            )  # Path within simulated /home/agent
            logger.info(
                f"Copying agent start script from {start_script_src} to {start_script_dest}"
            )
            shutil.copy(start_script_src, start_script_dest)

            if not start_script_dest.exists():
                raise FileNotFoundError(
                    f"Agent start script missing after copy: {start_script_dest}"
                )

            logger.info(purple("Running agent start.sh ..."))
            # Pass the combined config and agent args directly to start.sh
            # start.sh uses "$@" to pass these to the `aide` command.
            logger.debug(
                f"Running bash {start_script_dest} with CWD {ch}. Env: {env}. Args: {final_aide_args}"
            )

            agent_timeout = int(env.get("TIME_LIMIT_SECS", 3600))  # Use TIME_LIMIT_SECS env var

            result = subprocess.run(
                ["bash", str(start_script_dest)] + final_aide_args,  # Pass the args here!
                cwd=ch,  # Set CWD to simulated /home/
                env=env,
                text=True,
                capture_output=True,
                timeout=agent_timeout,
            )

            logger.info(purple("===== agent stdout ====="))
            logger.info(result.stdout.strip() if result.stdout else "(empty)")
            logger.info(purple("===== agent stderr ====="))
            logger.info(result.stderr.strip() if result.stderr else "(empty)")

            if result.returncode != 0:
                if result.returncode == 124:
                    raise TimeoutError(f"Agent timed out after {agent_timeout} seconds.")
                else:
                    raise RuntimeError(f"Agent failed with exit code {result.returncode}.")

            logger.info(purple("Agent completed successfully."))
            main_logger.info(
                (purple(f"Agent completed successfully for competition {competition.id}."))
            )
            # --- 9. Save Selected Outputs ---
            # This section is inside the main try block, after agent finishes
            if not retain_workspace:
                logger.info("Saving selected outputs...")
                # Define paths to save *from* the simulated home directories
                # These are already Path objects
                items_to_save = {
                    "submission": ch / "submission",
                    "logs": ch / "logs",
                    "code": ch / "code",
                    "wandb": ch / "wandb",  # Assuming wandb is created under /home
                    "workspaces": ch / "workspaces",
                }

                for name, src_path in items_to_save.items():
                    dest_path = (
                        work / name
                    )  # Save to /path/to/run/submission, /path/to/run/logs etc. (root of work dir)
                    if src_path.exists():  # Now src_path is a Path object, .exists() works
                        try:
                            logger.info(f"Saving '{name}' from '{src_path}' to '{dest_path}'")
                            if dest_path.exists():
                                if dest_path.is_dir():
                                    shutil.rmtree(dest_path)
                                else:
                                    os.remove(dest_path)
                            shutil.move(src_path, dest_path)
                        except Exception as e:
                            logger.error(f"Error saving '{name}': {e}")
                    else:
                        logger.debug(
                            f"Source directory for '{name}' not found: '{src_path}'. Skipping save."
                        )

            logger.info(purple("Run completed."))
            main_logger.info(purple("Run completed."))

            return work

        except Exception as e:
            # This catches errors from server startup, agent run, or saving
            logger.error(purple(f"Run failed: {type(e).__name__}: {e}"))
            logger.error(traceback.format_exc())
            raise e

        finally:
            # --- 10. Cleanup Temporary Workspace ---
            # This block runs regardless of success or failure

            # Ensure server is stopped first
            if server and server.poll() is None:
                logger.info(f"Stopping grading server on port {port}...")
                try:
                    server.terminate()
                    server.wait(timeout=10)
                except Exception as e:
                    logger.error(f"Error during server shutdown on port {port}: {e}")

            if server_log_file and not server_log_file.closed:
                try:
                    server_log_file.close()
                except Exception as e:
                    logger.error(f"Error closing server log: {e}")

            if not retain_workspace:
                logger.info(f"Cleaning up temporary workspace in: {work}")
                # Define items that should *remain* in the main run_dir
                items_to_keep_resolved = {
                    (work / "run.log").resolve(),
                    (work / "grading_server.log").resolve(),
                    # Saved items that should now be at the root of work
                    (work / "submission").resolve(),
                    (work / "logs").resolve(),
                    (work / "code").resolve(),
                    (work / "wandb").resolve(),  # Ensure wandb is included if it might be saved
                    (
                        work / "workspaces"
                    ).resolve(),  # Ensure wandb is included if it might be saved
                }
                # Filter the set to only include paths that actually exist after saving attempts
                items_to_keep_resolved = {item for item in items_to_keep_resolved if item.exists()}
                try:
                    # Iterate through *all* items in the main run_dir (work)
                    for item in list(
                        work.iterdir()
                    ):  # Use list() to avoid issues while deleting during iteration
                        if item.resolve() not in items_to_keep_resolved:
                            logger.debug(f"Removing temporary item: {item}")
                            try:
                                if item.is_dir():
                                    shutil.rmtree(item)
                                else:
                                    os.remove(item)
                            except Exception as item_e:
                                logger.error(f"Error removing {item}: {item_e}")

                    workspaces = (work / "workspaces").resolve()

                    for input_dir in workspaces.glob("*/input"):
                        if input_dir.is_dir():
                            logger.info(f"Removing temporary item: {input_dir}")
                            shutil.rmtree(input_dir, ignore_errors=True)

                    # These are the top-level temporary folders we created
                    if ch.exists():  # The simulated '/home' directory
                        logger.debug(f"Removing simulated home directory: {ch}")
                        try:
                            shutil.rmtree(ch)
                        except Exception as e:
                            logger.error(f"Error removing simulated home {ch}: {e}")

                    # The simulated '/private' directory (containing private data)
                    simulated_private_data_root = (
                        work / "private"
                    )  # Re-define path for clarity in cleanup
                    if simulated_private_data_root.exists():
                        logger.debug(
                            f"Removing simulated private data root: {simulated_private_data_root}"
                        )
                        try:
                            shutil.rmtree(simulated_private_data_root)
                        except Exception as e:
                            logger.error(
                                f"Error removing simulated private data {simulated_private_data_root}: {e}"
                            )

                    logger.info("Temporary workspace cleanup complete.")
                except Exception as e:
                    logger.error(
                        purple(
                            f"An unexpected error occurred during workspace cleanup: {e}. Manual cleanup may be required for {work}"
                        )
                    )

            else:
                logger.info(f"Retaining full workspace: {work}")

    except Exception as e:
        # This catches errors from server startup, agent run, or saving
        logger.error(purple(f"Run failed: {type(e).__name__}: {e}"))
        logger.error(traceback.format_exc())
        raise e
