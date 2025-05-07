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
    # Ensure __file__ is defined (e.g., not running in an interactive session where it might be missing)
    if "__file__" not in globals():
        raise NameError(
            "__file__ not defined. This script might be running in an environment where __file__ is not available."
        )
    shared_env_path = Path(__file__).parent.parent.resolve() / ".shared_env"
    CONSTANTS = dotenv_values(shared_env_path)
except FileNotFoundError:
    logging.error(f"'.shared_env' file not found at {shared_env_path}. Please create it.")
    sys.exit(1)
except NameError as e:
    logging.error(f"Could not determine shared_env_path: {e}")
    sys.exit(1)
except Exception as e:
    logging.error(f"Failed to load .shared_env file: {e}")
    sys.exit(1)


# --- Load Default AIDE Configuration from YAML File ---
# Assumes the base aide config template is at project root / environment / aide_config.yaml
# project root is two levels up from this script's directory
try:
    if "__file__" not in globals():
        raise NameError(
            "__file__ not defined. This script might be running in an environment where __file__ is not available."
        )
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
except NameError as e:
    logging.error(f"Could not determine aide_config_template_path: {e}")
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
    main_logger=logging.Logger,  # type: ignore
    retain_workspace: bool = False,
    seed: int = 0,
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
        main_logger: Main logger from the calling process.
        retain_workspace: If True, the full temporary workspace is kept.
        seed: The seed number for this run.

    Returns:
        Path to the run_dir.
    """
    logger.info(
        f"Starting local execution for competition {competition.id} with agent {agent.name} (seed {seed})"
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

    # --- 5. Run the Agent (and related setup) ---
    try:
        # e (now partially updated) default config into key=value strings
        flattened_aide_config = flatten_dict(aide_config_dict)
        # Remove keys that start.sh already sets explicitly on the aide command line
        keys_set_in_start_sh = ["data_dir", "desc_file", "agent.code.model"]
        aide_cli_args = [
            f"{k}={v}"
            for k, v in flattened_aide_config.items()
            if k not in keys_set_in_start_sh and v != "null"
        ]

        agent_override_args = []
        if agent.kwargs_type == "argparse":
            for key, value in agent.kwargs.items():
                agent_override_args += [f"--{key}", str(value)]
        elif agent.kwargs_type == "omegaconf":
            for key, value in agent.kwargs.items():
                agent_override_args += [f"{key}={value}"]
        else:
            logger.warning(
                f"Unknown agent kwargs_type: {agent.kwargs_type}. Skipping passing agent.kwargs."
            )

        final_aide_args = aide_cli_args + agent_override_args

        # --- 6. Environment Variables Setup ---
        env = os.environ.copy()
        env.setdefault("TIME_LIMIT_SECS", "3600")

        env.update(
            {
                "HOME_DIR": str(ch),
                "SUBMISSION_DIR": str(ch / "submission"),
                "LOGS_DIR": str(ch / "logs"),
                "CODE_DIR": str(ch / "code"),
                "AGENT_DIR": str(ch / "agent"),
                "COMPETITION_ID": competition.id,
                "MBX_GRADE_PORT": str(port),
                "PRIVATE_DATA_DIR": str(simulated_private_data_root.resolve()),
                "SEED": str(seed),
            }
        )

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

        # --- 7. Start Grading Server & Run Agent ---
        server_log_path = work / "grading_server.log"
        server_log_file = None
        server = None

        try:  # This try is for server startup, agent run, symlink creation, and saving outputs
            server_log_file = server_log_path.open("w", encoding="utf-8")
            if "__file__" not in globals():
                raise NameError("__file__ not defined for server script path.")
            repo_root_server = (
                Path(__file__).resolve().parents[1]
            )  # Re-evaluate for safety if structure changes
            server_script = repo_root_server / "environment" / "grading_server.py"
            assert server_script.exists(), f"Server script missing: {server_script}"

            logger.debug(
                f"Running server: {sys.executable} {server_script} --port {port} from CWD {work}"
            )
            server = subprocess.Popen(
                [sys.executable, str(server_script), "--port", str(port)],
                cwd=work,
                env=env,
                stdout=server_log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

            ready = False
            logger.info(f"Waiting for server on port {port}...")
            timeout_seconds = 60
            start_time = time.monotonic()
            while time.monotonic() - start_time < timeout_seconds:
                if server.poll() is not None:
                    if server_log_file and not server_log_file.closed:
                        server_log_file.close()
                    server_log_text = server_log_path.read_text(encoding="utf-8", errors="ignore")
                    raise RuntimeError(
                        f"Server exited unexpectedly (code {server.poll()}) on port {port}. Logs: {server_log_text[-1000:]}..."
                    )
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
                    pass
                except Exception as e_conn:
                    logger.debug(f"Server check failed on port {port}: {e_conn}. Retrying...")
                time.sleep(1)

            if not ready:
                if server_log_file and not server_log_file.closed:
                    server_log_file.close()
                server_log_text = server_log_path.read_text(encoding="utf-8", errors="ignore")
                raise RuntimeError(
                    f"Server did not start on port {port} within {timeout_seconds}s. Logs: {server_log_text[-1000:]}..."
                )
            logger.info(purple(f"Grading server on port {port} is ready."))

            # --- 8. Run Agent Start Script ---
            agent_src_dir = agent.agents_dir / agent.name
            start_script_name = "start.sh"
            start_script_src = agent_src_dir / start_script_name
            start_script_dest = ch / "agent" / start_script_name
            logger.info(
                f"Copying agent start script from {start_script_src} to {start_script_dest}"
            )
            shutil.copy(start_script_src, start_script_dest)

            if not start_script_dest.exists():
                raise FileNotFoundError(
                    f"Agent start script missing after copy: {start_script_dest}"
                )

            logger.info(purple("Running agent start.sh ..."))
            logger.debug(f"Running bash {start_script_dest} with CWD {ch}. Args: {final_aide_args}")
            agent_timeout = int(env.get("TIME_LIMIT_SECS", 3600))

            result = subprocess.run(
                ["bash", str(start_script_dest)] + final_aide_args,
                cwd=ch,
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
                if (
                    result.returncode == 124
                ):  # Timeout error code from `timeout` utility often used in bash scripts
                    raise TimeoutError(f"Agent timed out after {agent_timeout} seconds.")
                else:
                    raise RuntimeError(f"Agent failed with exit code {result.returncode}.")

            logger.info(purple("Agent completed successfully."))
            if main_logger:
                main_logger.info(
                    purple(f"Agent completed successfully for competition {competition.id}.")
                )

            # --- CREATE/FIX SYMLINKS POST-AGENT-RUN ---
            logger.info("Creating/Fixing symlinks for submission files post-agent-run...")
            post_agent_workspaces_dir = ch / "workspaces"  # Agent should have populated this
            post_agent_logs_dir = ch / "logs"

            for workspace in post_agent_workspaces_dir.iterdir():
                if workspace.is_dir():
                    best_submission = workspace / "best_submission" / "submission.csv"
                    if best_submission.exists():
                        log_subdir = post_agent_logs_dir / workspace.name
                        log_subdir.mkdir(exist_ok=True)
                        submission_link = log_subdir / "submission.csv"

                        if submission_link.exists():
                            if submission_link.is_symlink():
                                logger.debug(f"Removing existing symlink: {submission_link}")
                                submission_link.unlink()
                            else:
                                logger.warning(
                                    f"File {submission_link} exists and is not a symlink. Removing to create new symlink."
                                )
                                submission_link.unlink()  # Or os.remove(submission_link) if not a dir

                        relative_target_path = os.path.relpath(
                            best_submission.resolve(), start=submission_link.parent.resolve()
                        )
                        submission_link.symlink_to(relative_target_path)
                        logger.info(
                            f"Created/Updated symlink: {submission_link} -> {relative_target_path} (points to {best_submission.resolve()})"
                        )
                    else:
                        logger.warning(
                            f"Expected submission file not found at {best_submission} after agent run for workspace {workspace.name}."
                        )

            # --- 9. Save Selected Outputs ---
            if not retain_workspace:
                logger.info("Saving selected outputs...")
                items_to_save = {
                    "submission": ch / "submission",
                    "logs": ch / "logs",  # Now contains corrected symlinks
                    "code": ch / "code",
                    "wandb": ch / "wandb",
                    "workspaces": ch / "workspaces",  # Contains the actual submission files
                }

                for name, src_path in items_to_save.items():
                    dest_path = work / name
                    if src_path.exists():
                        try:
                            logger.info(f"Saving '{name}' from '{src_path}' to '{dest_path}'")
                            if dest_path.exists():
                                if dest_path.is_dir():
                                    shutil.rmtree(dest_path)
                                else:
                                    os.remove(dest_path)
                            shutil.move(
                                str(src_path), str(dest_path)
                            )  # Ensure strings for older shutil
                        except Exception as e_save:
                            logger.error(f"Error saving '{name}': {e_save}")
                    else:
                        logger.debug(f"Source for '{name}' not found: '{src_path}'. Skipping save.")

            logger.info(purple("Run completed."))
            if main_logger:
                main_logger.info(purple(f"Run completed for competition {competition.id}."))
            return work

        except Exception as e_inner:  # Catches errors from server startup, agent run, symlinks, saving
            logger.error(purple(f"Inner run block failed: {type(e_inner).__name__}: {e_inner}"))
            logger.error(traceback.format_exc())
            raise e_inner  # Re-raise to be caught by outer try-except if necessary, or handled by caller

        finally:
            # --- 10. Cleanup Server and Potentially Workspace ---
            if server and server.poll() is None:
                logger.info(f"Stopping grading server on port {port}...")
                try:
                    server.terminate()
                    server.wait(timeout=10)
                except Exception as e_server_stop:
                    logger.error(f"Error during server shutdown on port {port}: {e_server_stop}")
            if server_log_file and not server_log_file.closed:
                try:
                    server_log_file.close()
                except Exception as e_log_close:
                    logger.error(f"Error closing server log: {e_log_close}")

            if not retain_workspace:
                logger.info(f"Cleaning up temporary workspace components in: {work}")
                # Items that were explicitly moved to `work` or are logs should remain.
                # `ch` and `simulated_private_data_root` are the main temporary structures to remove.

                # Remove simulated '/home' if it wasn't moved entirely and still exists
                if (
                    ch.exists() and ch.is_relative_to(work) and (work / ch.name).exists()
                ):  # Check if ch itself is still there (e.g. if retain_workspace=True previously)
                    logger.debug(f"Removing simulated home directory contents: {ch}")

                    # Remove the simulated '/home' directory structure if it's still under `work`
                    # This path was work / "home"
                    simulated_home_to_remove = work / "home"
                    if simulated_home_to_remove.exists():
                        logger.debug(
                            f"Removing full simulated home directory: {simulated_home_to_remove}"
                        )
                        try:
                            shutil.rmtree(simulated_home_to_remove)
                        except Exception as e_rm_home:
                            logger.error(
                                f"Error removing simulated home {simulated_home_to_remove}: {e_rm_home}"
                            )

                # Remove simulated '/private'
                if simulated_private_data_root.exists():
                    logger.debug(
                        f"Removing simulated private data root: {simulated_private_data_root}"
                    )
                    try:
                        shutil.rmtree(simulated_private_data_root)
                    except Exception as e_rm_private:
                        logger.error(
                            f"Error removing simulated private data {simulated_private_data_root}: {e_rm_private}"
                        )

                # Specific cleanup for items not caught by moving `ch` subdirectories
                # e.g. if `ch/agent` was not in `items_to_save`
                if (work / "home" / "agent").exists():
                    shutil.rmtree(work / "home" / "agent", ignore_errors=True)

                # Clean up any remaining empty directories or specific files in work if necessary,
                # but the `items_to_keep_resolved` logic from the original script was more for when
                # `ch` itself was the main `work` directory. Here `work` is the parent.
                # The main cleanup is `work/"home"` and `work/"private"`.
                logger.info("Temporary workspace component cleanup attempt complete.")

            else:  # if retain_workspace is True
                logger.info(f"Retaining full workspace: {work}")
                # Even if retaining, the symlinks should now be relative and robust
                # within the work/logs and work/workspaces structure if they were moved.
                # If they were not moved (i.e. ch/logs, ch/workspaces are still there),
                # the relative symlinks will work within ch.

    except Exception as e_outer:  # Catches errors from pre-agent setup or re-raised from inner block
        logger.error(purple(f"Outer run_locally failed: {type(e_outer).__name__}: {e_outer}"))
        logger.error(traceback.format_exc())
        # Ensure cleanup is attempted even if outer setup fails, though server might not be running.
        # The `finally` block for server/workspace cleanup is tied to the inner `try`.
        # If failure is before server `try`, that `finally` won't run.
        # However, primary cleanup candidates (ch, private_data_root) are created early.
        if not retain_workspace:
            if ch.exists():
                shutil.rmtree(ch, ignore_errors=True)
            if simulated_private_data_root.exists():
                shutil.rmtree(simulated_private_data_root, ignore_errors=True)
        raise e_outer

    return work  # Should return work whether successful or if retain_workspace is true on failure.
