from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import (  # Keep List for type hinting args
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

import psutil
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
    heartbeat_callback: Optional[Callable] = None,
    no_output_timeout_mins: int = 15,  # Kill after 15 minutes of no output (configurable)
    quiet_mode: bool = False,  # Whether to reduce logging verbosity
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
        heartbeat_callback: Optional callback function to call periodically during long operations.
        no_output_timeout_mins: Kill agent if no output for this many minutes (default 15).
        quiet_mode: Whether to reduce logging verbosity.

    Returns:
        Path to the run_dir.
    """
    process_id = os.getpid()
    start_time = time.monotonic()
    child_processes = []  # Track child processes for cleanup

    # Helper function to call heartbeat and flush logs
    def check_heartbeat():
        if heartbeat_callback:
            try:
                heartbeat_callback()
            except Exception as e:
                logger.warning(f"Heartbeat callback failed: {e}")
        # Always flush logs
        for handler in logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass

    # Helper function to kill a process and all its children
    def kill_process_tree(pid, include_parent=True):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # First try graceful termination
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Give them a moment to terminate
            gone, alive = psutil.wait_procs(children, timeout=3)

            # Force kill anything still alive
            for child in alive:
                try:
                    logger.warning(f"Force killing child process {child.pid}")
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            # Kill the parent if requested
            if include_parent:
                try:
                    parent.terminate()
                    try:
                        parent.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        parent.kill()
                except psutil.NoSuchProcess:
                    pass

            return True
        except psutil.NoSuchProcess:
            return False
        except Exception as e:
            logger.error(f"Error killing process tree for PID {pid}: {e}")
            return False

    # Log system resource usage
    def log_resource_usage(prefix=""):
        if quiet_mode:
            return  # Skip resource logging in quiet mode
        try:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            process = psutil.Process(os.getpid())
            process_mem = process.memory_info().rss / (1024 * 1024)  # MB
            logger.info(
                f"{prefix} CPU: {cpu}%, Memory: {mem.percent}%, Process memory: {process_mem:.1f} MB"
            )
        except Exception as e:
            logger.debug(f"Failed to log resource usage: {e}")

    logger.info(
        f"Starting local execution for competition {competition.id} with agent {agent.name} (seed {seed}) [PID:{process_id}]"
    )
    if not quiet_mode:
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Port assigned: {port}")
        logger.info(f"No-output timeout set to {no_output_timeout_mins} minutes")
        log_resource_usage("Initial resources:")

    work = run_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    # --- 1. Setup Simulated Environment Directories ---
    if not quiet_mode:
        logger.info("Setting up simulated environment directories...")
    ch = work / "home"
    ch.mkdir(parents=True, exist_ok=True)

    # Create standard directories
    for dir_name in ["submission", "logs", "code", "agent", "workspaces", "data"]:
        dir_path = ch / dir_name
        dir_path.mkdir(exist_ok=True)
        if not quiet_mode:
            logger.debug(f"Created directory: {dir_path}")

    # Simulate the /private/data mount point
    simulated_private_data_root = work / "private"
    simulated_private_data_root.mkdir(parents=True, exist_ok=True)
    if not quiet_mode:
        logger.debug(f"Created private data root: {simulated_private_data_root}")
    check_heartbeat()

    # --- 2. Copy Competition Data ---
    logger.info("Copying competition data...")
    copy_start_time = time.monotonic()
    simulated_public_data_dest = ch / "data"
    logger.info(
        f"Copying public data from {competition.public_dir} to {simulated_public_data_dest}"
    )
    try:
        # Copy with progress updates for large directories
        public_dir_size = sum(
            f.stat().st_size for f in competition.public_dir.glob("**/*") if f.is_file()
        )
        if not quiet_mode:
            logger.info(f"Public data size: {public_dir_size / (1024*1024):.2f} MB")

        # Do the actual copy
        shutil.copytree(competition.public_dir, simulated_public_data_dest, dirs_exist_ok=True)

        copy_time = time.monotonic() - copy_start_time
        if not quiet_mode:
            logger.info(f"Public data copy completed in {copy_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error copying public data: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to copy public data: {e}")

    check_heartbeat()

    if competition.private_dir.exists():
        private_copy_start = time.monotonic()
        private_target_simulated = (
            simulated_private_data_root / "data" / competition.id / "prepared" / "private"
        )
        logger.info(
            f"Copying private data from {competition.private_dir} to {private_target_simulated}"
        )
        try:
            private_target_simulated.mkdir(parents=True, exist_ok=True)

            # Get size for logging
            private_dir_size = sum(
                f.stat().st_size for f in competition.private_dir.glob("**/*") if f.is_file()
            )
            if not quiet_mode:
                logger.info(f"Private data size: {private_dir_size / (1024*1024):.2f} MB")

            # Do the actual copy
            shutil.copytree(competition.private_dir, private_target_simulated, dirs_exist_ok=True)

            private_copy_time = time.monotonic() - private_copy_start
            if not quiet_mode:
                logger.info(f"Private data copy completed in {private_copy_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error copying private data: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Continuing without private data...")
    else:
        logger.warning(f"Private data directory not found: {competition.private_dir}")

    check_heartbeat()
    if not quiet_mode:
        log_resource_usage("Resources after data copy:")

    # --- 3. Generate AIDE Command-Line Arguments from Config and Agent Kwargs ---
    if not quiet_mode:
        logger.info("Generating AIDE command-line arguments...")
    aide_config_dict = {}
    try:
        aide_config_dict = DEFAULT_AIDE_CONFIG.copy()
        if not quiet_mode:
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
        if not quiet_mode:
            logger.debug(f"Final AIDE args: {final_aide_args}")
        check_heartbeat()

        # --- 6. Environment Variables Setup ---
        env = os.environ.copy()
        env.setdefault("TIME_LIMIT_SECS", "7200")

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
                "PYTHONUNBUFFERED": "1",  # Force unbuffered Python output for better logging
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
            logger.info(f"Starting grading server on port {port}...")
            server_log_file = server_log_path.open("w", encoding="utf-8")
            if "__file__" not in globals():
                raise NameError("__file__ not defined for server script path.")
            repo_root_server = (
                Path(__file__).resolve().parents[1]
            )  # Re-evaluate for safety if structure changes
            server_script = repo_root_server / "environment" / "grading_server.py"
            assert server_script.exists(), f"Server script missing: {server_script}"

            if not quiet_mode:
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

            # Log server PID for debugging
            if server and server.pid:
                # Add server to tracked processes
                child_processes.append(server.pid)
                if not quiet_mode:
                    logger.info(f"Grading server started with PID: {server.pid}")

            ready = False
            logger.info(f"Waiting for server on port {port}...")
            timeout_seconds = 60
            start_time_server = time.monotonic()
            check_count = 0
            while time.monotonic() - start_time_server < timeout_seconds:
                check_count += 1
                if check_count % 5 == 0:  # Every 5 checks (5 seconds)
                    check_heartbeat()

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
                        # No need for a break here - we'll exit the loop naturally
                    conn.close()
                except ConnectionRefusedError:
                    pass
                except Exception as e_conn:
                    if not quiet_mode:
                        logger.debug(f"Server check failed on port {port}: {e_conn}. Retrying...")

                # Exit loop if server is ready
                if ready:
                    break

                time.sleep(1)

            if not ready:
                if server_log_file and not server_log_file.closed:
                    server_log_file.close()
                server_log_text = server_log_path.read_text(encoding="utf-8", errors="ignore")
                raise RuntimeError(
                    f"Server did not start on port {port} within {timeout_seconds}s. Logs: {server_log_text[-1000:]}..."
                )
            logger.info(purple(f"Grading server on port {port} is ready."))
            check_heartbeat()
            if not quiet_mode:
                log_resource_usage("Resources after server start:")

            # --- 8. Run Agent Start Script ---
            agent_src_dir = agent.agents_dir / agent.name
            start_script_name = "start.sh"
            start_script_src = agent_src_dir / start_script_name
            start_script_dest = ch / "agent" / start_script_name
            if not quiet_mode:
                logger.info(
                    f"Copying agent start script from {start_script_src} to {start_script_dest}"
                )
            shutil.copy(start_script_src, start_script_dest)

            if not start_script_dest.exists():
                raise FileNotFoundError(
                    f"Agent start script missing after copy: {start_script_dest}"
                )

            # Make sure the script is executable
            os.chmod(start_script_dest, 0o755)

            logger.info(purple("Running agent start.sh ..."))
            if not quiet_mode:
                logger.debug(
                    f"Running bash {start_script_dest} with CWD {ch}. Args: {final_aide_args}"
                )
            agent_timeout = int(env.get("TIME_LIMIT_SECS", 7200))

            # Log timeout and start time
            if not quiet_mode:
                logger.info(f"Agent timeout set to {agent_timeout} seconds")
            agent_start_time = time.monotonic()

            # Create a file to record start time for debugging
            with open(work / "agent_start_time.txt", "w") as f:
                f.write(f"Agent started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"PID: {os.getpid()}\n")
                f.write(f"Timeout: {agent_timeout} seconds\n")

            try:
                # Use Popen with streaming output for real-time logs instead of subprocess.run
                agent_log_path = work / "agent_output.log"
                with open(agent_log_path, "w", encoding="utf-8") as agent_log_file:
                    agent_process = subprocess.Popen(
                        ["bash", str(start_script_dest)] + final_aide_args,
                        cwd=ch,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,  # Line buffered
                        preexec_fn=os.setsid,  # Create a new process group for easier killing
                    )

                    # Add to tracked processes
                    child_processes.append(agent_process.pid)

                    if not quiet_mode:
                        logger.info(f"Agent process started with PID: {agent_process.pid}")

                    # Set up non-blocking I/O for stdout and stderr
                    import fcntl
                    import select

                    # Set stdout and stderr to non-blocking mode
                    for pipe in [agent_process.stdout, agent_process.stderr]:
                        fd = pipe.fileno()
                        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

                    # Process output in real-time with timeout
                    stdout_data, stderr_data = "", ""
                    start_time_agent = time.monotonic()
                    last_heartbeat_time = start_time_agent

                    # Keep track of whether we've seen output recently
                    last_output_time = time.monotonic()
                    no_output_warning_threshold = 300  # 5 minutes (warning only)
                    no_output_kill_threshold = no_output_timeout_mins * 60  # Kill threshold

                    while agent_process.poll() is None:
                        # Check for timeout
                        current_time = time.monotonic()

                        # Check for overall timeout
                        if current_time - start_time_agent > agent_timeout:
                            logger.error(
                                f"Agent timed out after {agent_timeout} seconds (global timeout)"
                            )
                            kill_process_tree(agent_process.pid)
                            raise TimeoutError(
                                f"Agent timed out after {agent_timeout} seconds (global timeout)"
                            )

                        # Check for no-output timeout
                        output_silence_time = current_time - last_output_time
                        if output_silence_time > no_output_kill_threshold:
                            logger.error(
                                f"Agent produced no output for {no_output_timeout_mins} minutes. Killing it."
                            )
                            kill_process_tree(agent_process.pid)
                            raise TimeoutError(
                                f"Agent killed after {no_output_timeout_mins} minutes of no output"
                            )

                        # Call heartbeat periodically
                        if current_time - last_heartbeat_time >= 60:  # Every minute
                            check_heartbeat()
                            last_heartbeat_time = current_time

                            # Check for no output warning (but not killing yet)
                            if output_silence_time > no_output_warning_threshold and not quiet_mode:
                                silence_mins = output_silence_time / 60
                                logger.warning(
                                    f"No output from agent for {silence_mins:.1f} minutes (will kill at {no_output_timeout_mins})"
                                )
                                log_resource_usage("Resources during agent silence:")

                        # Check for output from the process (without blocking)
                        ready_pipes = select.select(
                            [agent_process.stdout, agent_process.stderr], [], [], 1.0
                        )

                        if agent_process.stdout in ready_pipes[0]:
                            output = agent_process.stdout.read()
                            if output:
                                last_output_time = current_time
                                stdout_data += output
                                # Only log agent output to console if not in quiet mode
                                if not quiet_mode:
                                    logger.info(output.strip())
                                # Always write to log file
                                agent_log_file.write(output)
                                agent_log_file.flush()

                        if agent_process.stderr in ready_pipes[0]:
                            output = agent_process.stderr.read()
                            if output:
                                last_output_time = current_time
                                stderr_data += output
                                # Always log stderr (it's usually important even in quiet mode)
                                logger.warning(output.strip())
                                agent_log_file.write(f"STDERR: {output}")
                                agent_log_file.flush()

                    # Process any remaining output
                    remaining_stdout = agent_process.stdout.read()
                    if remaining_stdout:
                        stdout_data += remaining_stdout
                        if not quiet_mode:
                            logger.info(remaining_stdout.strip())
                        agent_log_file.write(remaining_stdout)

                    remaining_stderr = agent_process.stderr.read()
                    if remaining_stderr:
                        stderr_data += remaining_stderr
                        logger.warning(remaining_stderr.strip())
                        agent_log_file.write(f"STDERR: {remaining_stderr}")

                # Check return code
                return_code = agent_process.returncode
                if not quiet_mode:
                    logger.info(f"Agent process completed with return code: {return_code}")

                if return_code != 0:
                    if return_code == 124:  # Timeout error code from `timeout` utility
                        raise TimeoutError(f"Agent timed out after {agent_timeout} seconds.")
                    else:
                        raise RuntimeError(f"Agent failed with exit code {return_code}.")

            except TimeoutError:
                # This catches both our manual timeout and subprocess.run timeout
                logger.error(f"Agent timed out after {agent_timeout} seconds")
                raise
            except Exception as e:
                logger.error(f"Error running agent: {e}")
                logger.error(traceback.format_exc())
                raise

            logger.info(purple("Agent completed successfully."))

            # Log elapsed time
            agent_elapsed_time = time.monotonic() - agent_start_time
            logger.info(f"Agent completed in {agent_elapsed_time:.2f} seconds")

            if main_logger:
                main_logger.info(
                    purple(f"Agent completed successfully for competition {competition.id}.")
                )

            check_heartbeat()
            if not quiet_mode:
                log_resource_usage("Resources after agent completion:")

            # --- CREATE/FIX SYMLINKS POST-AGENT-RUN ---
            logger.info("Creating/Fixing symlinks for submission files post-agent-run...")
            post_agent_workspaces_dir = ch / "workspaces"  # Agent should have populated this
            post_agent_logs_dir = ch / "logs"

            submission_count = 0
            for workspace in post_agent_workspaces_dir.iterdir():
                if workspace.is_dir():
                    best_submission = workspace / "best_submission" / "submission.csv"
                    if best_submission.exists():
                        submission_count += 1
                        log_subdir = post_agent_logs_dir / workspace.name
                        log_subdir.mkdir(exist_ok=True)
                        submission_link = log_subdir / "submission.csv"

                        if submission_link.exists():
                            if submission_link.is_symlink():
                                if not quiet_mode:
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

            logger.info(f"Total submission files found: {submission_count}")
            check_heartbeat()

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
                        if not quiet_mode:
                            logger.debug(
                                f"Source for '{name}' not found: '{src_path}'. Skipping save."
                            )

            # This is the one place we return, so include the timing code here
            logger.info(purple("Run completed."))
            if main_logger:
                main_logger.info(purple(f"Run completed for competition {competition.id}."))

            total_time = time.monotonic() - start_time
            logger.info(f"Total run_locally execution time: {total_time:.2f} seconds")
            return work

        except Exception as e_inner:  # Catches errors from server startup, agent run, symlinks, saving
            logger.error(purple(f"Inner run block failed: {type(e_inner).__name__}: {e_inner}"))
            logger.error(traceback.format_exc())
            raise e_inner  # Re-raise to be caught by outer try-except if necessary, or handled by caller

    except Exception as e_outer:  # Catches errors from pre-agent setup or re-raised from inner block
        total_time = time.monotonic() - start_time
        logger.error(
            purple(
                f"Outer run_locally failed after {total_time:.2f}s: {type(e_outer).__name__}: {e_outer}"
            )
        )
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

    finally:
        # Attempt to clean up any child processes we created
        for pid in child_processes:
            try:
                kill_process_tree(pid, include_parent=True)
            except Exception as e:
                logger.warning(f"Failed to clean up process {pid}: {e}")

        # Original cleanup code for server and workspace...
        if server and server.poll() is None:
            logger.info(f"Stopping grading server on port {port}...")
            try:
                server.terminate()
                server.wait(timeout=10)
            except Exception as e_server_stop:
                logger.error(f"Error during server shutdown on port {port}: {e_server_stop}")
                try:
                    logger.warning(f"Attempting to force kill server on port {port}")
                    server.kill()
                except Exception as e_kill:
                    logger.error(f"Failed to kill server: {e_kill}")

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
                if not quiet_mode:
                    logger.debug(f"Removing simulated home directory contents: {ch}")

                # Remove the simulated '/home' directory structure if it's still under `work`
                # This path was work / "home"
                simulated_home_to_remove = work / "home"
                if simulated_home_to_remove.exists():
                    if not quiet_mode:
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
                if not quiet_mode:
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


# mlebench grade --submission  RedHatAI_DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic_data_None_25_steps/submission_paths_seed_all.jsonl  --output-dir RedHatAI_DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic_data_None_25_steps --data-dir lite_dataset
