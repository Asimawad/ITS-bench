#!/usr/bin/env python3

import argparse
import json
import logging
import multiprocessing
import os
import random
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psutil

from agents.registry import Agent
from agents.registry import registry as agent_registry
from agents.run_local import run_locally
from environment.upload_results import upload_to_s3
from environment.utils_zip import make_filtered_zip
from mlebench.registry import Competition, registry
from mlebench.utils import create_run_dir, get_logger, get_runs_dir, get_timestamp, purple

# Global flag for quiet mode
QUIET_MODE = False

# Custom logger initialization
def setup_logging(quiet_mode=False):
    global QUIET_MODE
    QUIET_MODE = quiet_mode

    root_logger = logging.getLogger()
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set appropriate level based on quiet mode
    log_level = logging.WARNING if quiet_mode else logging.INFO
    root_logger.setLevel(log_level)

    # Create a handler that logs to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(process)d] %(levelname)s %(name)s - %(message)s")
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(log_level)
    root_logger.addHandler(stdout_handler)

    return get_logger(__name__)


# Initialize with default (will be updated in main)
logger = get_logger(__name__)


@dataclass(frozen=True)
class Task:
    run_id: str
    seed: int
    path_to_run_group: Path
    path_to_run: Path
    agent: Agent
    competition: Competition
    port: int  # Unique port for this run's grading server
    retry_count: int = 0  # Track how many times this task has been retried


def worker_process(task: Task):
    """
    Worker process that runs a single task. Enhanced with robust logging, heartbeats, and exception handling.

    Args:
        task (Task): The task to run.

    Returns:
        tuple: (run_id, task_output) where task_output contains success status and error details if applicable
    """
    process_id = os.getpid()
    start_time = time.time()
    heartbeat_interval = 60  # Log a heartbeat every minute
    last_heartbeat = start_time
    task_output = {
        "success": False,
        "error": None,
        "process_id": process_id,
        "retry_count": task.retry_count,
    }

    # Create and configure run-specific logger
    run_logger = get_logger(f"{task.run_id}_{process_id}")
    # Always log to file at INFO level, regardless of quiet mode
    run_logger.setLevel(logging.INFO)

    # Create log directory if it doesn't exist
    task.path_to_run.mkdir(parents=True, exist_ok=True)
    log_file_path = task.path_to_run / "run.log"

    # For retries, rename the existing log file instead of overwriting
    if task.retry_count > 0 and log_file_path.exists():
        old_log_file = task.path_to_run / f"run.log.attempt{task.retry_count - 1}"
        try:
            log_file_path.rename(old_log_file)
            run_logger.info(f"Previous log file moved to {old_log_file}")
        except Exception as e:
            run_logger.warning(f"Failed to rename previous log file: {e}")

    # Add file handler - always at INFO level for complete logs
    file_handler = logging.FileHandler(log_file_path)
    file_formatter = logging.Formatter("%(asctime)s [PID:%(process)d] %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)  # Always INFO for file logs
    run_logger.addHandler(file_handler)

    # Add stream handler with level dependent on quiet mode
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(file_formatter)
    stream_handler.setLevel(logging.WARNING if QUIET_MODE else logging.INFO)
    run_logger.addHandler(stream_handler)

    # Prevent propagation to avoid duplicate logs
    run_logger.propagate = False

    try:
        # Log worker start with detailed information - minimal logging if in quiet mode
        run_logger.info(f"=== WORKER STARTED [PID:{process_id}] ===")
        if task.retry_count > 0:
            # Always show retry information, even in quiet mode
            run_logger.warning(f"*** RETRY ATTEMPT #{task.retry_count} ***")

        # More verbose logging only if not in quiet mode
        if not QUIET_MODE:
            run_logger.info(
                f"Competition: {task.competition.id}, Agent: {task.agent.name}, Seed: {task.seed}, Port: {task.port}"
            )
            run_logger.info(f"Run directory: {task.path_to_run}")

            # Log system resource info at start
            memory_info = psutil.virtual_memory()
            run_logger.info(
                f"System memory: {memory_info.percent}% used, {memory_info.available / (1024**3):.2f} GB available"
            )

        # Run the task with periodic heartbeats
        def heartbeat_callback():
            nonlocal last_heartbeat
            current_time = time.time()
            if current_time - last_heartbeat >= heartbeat_interval:
                last_heartbeat = current_time
                elapsed_minutes = (current_time - start_time) / 60

                if not QUIET_MODE:
                    memory_info = psutil.virtual_memory()
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    run_logger.info(
                        f"HEARTBEAT - Task running for {elapsed_minutes:.1f} minutes - CPU: {cpu_percent}%, Mem: {memory_info.percent}%"
                    )

                # Flush logs to ensure they're written
                for handler in run_logger.handlers:
                    handler.flush()

        # Call the run_locally function with heartbeat and enhanced logging
        if not QUIET_MODE:
            run_logger.info(f"Initializing environment for task...")
        heartbeat_callback()  # Initial heartbeat

        # Set a more aggressive no-output timeout based on retry count
        # First attempt: 15 minutes, Second: 10 minutes, Third+: 5 minutes
        no_output_timeout = max(5, 15 - (task.retry_count * 5))
        if not QUIET_MODE:
            run_logger.info(f"Setting no-output timeout to {no_output_timeout} minutes")

        # Run the task with custom logger
        run_locally(
            competition=task.competition,
            agent=task.agent,
            run_dir=task.path_to_run,
            logger=run_logger,
            main_logger=logger,
            port=task.port,
            seed=task.seed,
            heartbeat_callback=heartbeat_callback,  # Pass the heartbeat callback
            no_output_timeout_mins=no_output_timeout,  # Pass configurable timeout
            quiet_mode=QUIET_MODE,  # Pass the quiet mode flag
        )

        # Task completed successfully
        elapsed_time = time.time() - start_time
        run_logger.info(purple(f"Task completed successfully in {elapsed_time:.2f} seconds"))
        logger.info(
            purple(
                f"Task completed successfully - Competition: {task.competition.id}, Agent: {task.agent.name}, Seed: {task.seed}"
            )
        )

        task_output["success"] = True
        task_output["elapsed_time"] = elapsed_time

    except KeyboardInterrupt:
        run_logger.error("Task interrupted by user")
        task_output["error"] = {
            "type": "KeyboardInterrupt",
            "message": "Task was interrupted by user",
            "stack_trace": traceback.format_exc(),
            "retry_suggested": False,  # Don't retry on manual interruption
        }

    except TimeoutError as e:
        run_logger.error(f"Task timed out: {e}")
        task_output["error"] = {
            "type": "TimeoutError",
            "message": str(e),
            "stack_trace": traceback.format_exc(),
            "retry_suggested": True,  # Retry on timeout
        }

    except Exception as e:
        stack_trace = traceback.format_exc()
        run_logger.error(f"Task failed with error type: {type(e).__name__}")
        run_logger.error(f"Error details: {str(e)}")

        # Only log full stack trace if not in quiet mode
        if not QUIET_MODE:
            run_logger.error(f"Stack trace:\n{stack_trace}")

        # Log additional system info for debugging, but only if not in quiet mode
        if not QUIET_MODE:
            try:
                memory_info = psutil.virtual_memory()
                run_logger.error(
                    f"System memory at failure: {memory_info.percent}% used, {memory_info.available / (1024**3):.2f} GB available"
                )
                cpu_percent = psutil.cpu_percent(interval=0.1)
                run_logger.error(f"CPU usage at failure: {cpu_percent}%")
            except Exception as e_info:
                run_logger.error(f"Failed to get system info: {e_info}")

        # Determine if this error should be retried
        # Generally retry network, resource, and timeout issues
        retry_error_types = [
            "ConnectionError",
            "TimeoutError",
            "ResourceError",
            "MemoryError",
            "RuntimeError",
            "OSError",
            "IOError",
        ]
        retry_suggested = any(err_type in type(e).__name__ for err_type in retry_error_types)

        # Also suggest retry if error message contains specific keywords
        retry_keywords = ["timeout", "connection", "reset", "refused", "failed", "hang", "deadlock"]
        if not retry_suggested:
            retry_suggested = any(keyword in str(e).lower() for keyword in retry_keywords)

        task_output["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "stack_trace": stack_trace,
            "retry_suggested": retry_suggested,
        }

    finally:
        # Always log completion and clean up resources
        end_time = time.time()
        elapsed_time = end_time - start_time
        task_output["elapsed_time"] = elapsed_time

        run_logger.info(
            f"=== WORKER FINISHED [PID:{process_id}] === (elapsed: {elapsed_time:.2f}s)"
        )

        # Ensure all logs are flushed before returning
        for handler in run_logger.handlers:
            handler.flush()
            handler.close()
            run_logger.removeHandler(handler)

        # Write a completion marker file to indicate this process finished (even if with error)
        try:
            completion_marker = task.path_to_run / f"process_{process_id}_completed.txt"
            with open(completion_marker, "w") as f:
                f.write(f"Process completed at {datetime.now().isoformat()}\n")
                f.write(f"Success: {task_output['success']}\n")
                f.write(f"Retry attempt: {task.retry_count}\n")
                if task_output["error"]:
                    f.write(
                        f"Error: {task_output['error']['type']} - {task_output['error']['message']}\n"
                    )
        except Exception as e_marker:
            # If we can't write the marker, log it but don't fail
            print(f"Failed to write completion marker: {e_marker}")

    return task.run_id, task_output


def zip_and_upload_runs(every_minutes: int = 10):
    """
    Periodically zips the runs directory and uploads it to S3 for checkpointing.
    Uses make_filtered_zip to exclude large data directories.

    Args:
        every_minutes: How often to checkpoint in minutes
    """

    def task():
        logger.info(f"Starting checkpointing thread (every {every_minutes} minutes)")

        checkpoint_counter = 1
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        while True:
            try:
                time.sleep(every_minutes * 60)

                # Skip if runs directory doesn't exist yet
                if not os.path.isdir("runs"):
                    logger.warning("Checkpoint skipped: 'runs' directory does not exist yet")
                    continue

                # Create timestamp for the checkpoint
                timestamp = get_timestamp()
                checkpoint_name = f"checkpoint_{timestamp}_{checkpoint_counter:03d}"
                checkpoint_file = checkpoint_dir / f"{checkpoint_name}.zip"

                logger.info(
                    f"[Checkpoint] Creating checkpoint {checkpoint_counter}: {checkpoint_file}"
                )

                # Zip the runs folder with filtering (excludes data/input dirs)
                result_path = make_filtered_zip(checkpoint_file, "runs")

                # Check if Aichor environment variables are available for S3 upload
                if "AICHOR_OUTPUT_PATH" in os.environ and "S3_ENDPOINT" in os.environ:
                    # Upload to S3
                    remote_key = f"{os.environ['AICHOR_OUTPUT_PATH'].rstrip('/')}/checkpoints/{checkpoint_name}.zip"
                    upload_to_s3(result_path, remote_key)
                    logger.info(f"[Checkpoint] Uploaded {checkpoint_file} to {remote_key}")
                else:
                    logger.info(f"[Checkpoint] Created local checkpoint: {checkpoint_file}")
                    logger.warning(
                        "S3 upload skipped: AICHOR_OUTPUT_PATH environment variable not set"
                    )

                # Increment counter for next checkpoint
                checkpoint_counter += 1

            except Exception as e:
                logger.error(f"[Checkpoint] Error during checkpointing: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue the loop even after error

    # Start the checkpointing thread as daemon (will exit when main thread exits)
    checkpoint_thread = threading.Thread(target=task, daemon=True)
    checkpoint_thread.start()
    return checkpoint_thread


def init_worker():
    """
    Initializes worker process. This helps handle keyboard interrupts properly.
    """
    # Handle signals at process level
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore keyboard interrupt in worker processes


# Move queue_worker outside of main function to make it picklable
def queue_worker(task_queue, result_queue, active_workers):
    """Worker function that processes tasks from the queue until empty."""
    # Set up signal handling
    init_worker()

    while True:
        try:
            # Try to get a task, wait for 1 second if queue is empty
            try:
                task = task_queue.get(timeout=1)
            except Exception:
                # If queue is empty and we've been waiting, check if we should exit
                if task_queue.empty() and active_workers.value <= 0:
                    # No more tasks and no active workers, time to exit
                    break
                continue

            # Increment active worker count - use value property directly without lock
            active_workers.value += 1

            try:
                # Process the task
                result = worker_process(task)
                # Put result in result queue
                result_queue.put(result)
            except Exception as e:
                # If worker_process itself fails (should be rare), log and continue
                logger.error(f"Worker process raised unhandled exception: {e}")
                logger.error(traceback.format_exc())
                # Put a failure result in the queue
                error_result = (
                    task.run_id,
                    {
                        "success": False,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "stack_trace": traceback.format_exc(),
                            "retry_suggested": True,  # Always retry framework-level errors
                        },
                        "retry_count": task.retry_count,
                    },
                )
                result_queue.put(error_result)
            finally:
                # Decrement active worker count - use value property directly without lock
                active_workers.value -= 1

        except KeyboardInterrupt:
            # Handle interrupt
            logger.warning("Worker received keyboard interrupt, exiting...")
            break


def main(args):
    # Initialize logging based on quiet mode
    global logger
    logger = setup_logging(args.quiet)

    # Set up a custom data directory that uses lite_dataset
    lite_data_dir = Path("./lite_dataset")
    if not lite_data_dir.exists():
        raise ValueError(f"Lite dataset directory not found: {lite_data_dir.resolve()}")

    global registry
    registry = registry.set_data_dir(lite_data_dir)

    agent = agent_registry.get_agent(args.agent_id)

    run_group = f"{get_timestamp()}_run-group_{agent.name}"

    # Start the checkpointing thread
    if args.enable_checkpoints:
        logger.info(f"Enabling automatic checkpointing every {args.checkpoint_interval} minutes")
        checkpoint_thread = zip_and_upload_runs(every_minutes=args.checkpoint_interval)
    else:
        logger.info("Automatic checkpointing is disabled")

    # Load competition ids from the file
    with open(args.competition_set, "r") as f:
        competition_ids = [line.strip() for line in f.read().splitlines() if line.strip()]

    # Create tasks for each competition and seed
    logger.info(f"Launching run group: {run_group}")

    # Tasks to run initially
    initial_tasks = []
    # Port range to use - ensure we have enough for all tasks and retries
    base_port = args.base_port if hasattr(args, "base_port") and args.base_port else 5000
    max_retries = args.max_retries if hasattr(args, "max_retries") else 3

    # Calculate how many ports we need in total (including possible retries)
    total_possible_tasks = len(competition_ids) * args.n_seeds * (max_retries + 1)
    logger.info(
        f"Reserving ports {base_port} to {base_port + total_possible_tasks - 1} for tasks and retries"
    )

    # Port assignment function - gives each task a deterministic port even after retries
    def get_port_for_task(comp_id, seed, retry_count=0):
        # Create a deterministic "task index" based on competition and seed
        comp_idx = competition_ids.index(comp_id)
        task_idx = comp_idx * args.n_seeds + seed
        # Add retry offset
        port = base_port + task_idx * (max_retries + 1) + retry_count
        return port

    # Create initial task list
    for seed in range(args.n_seeds):
        for competition_id in competition_ids:
            try:
                # Get competition from registry
                competition = registry.get_competition(competition_id)

                # Create a run directory for this task
                run_dir = create_run_dir(competition.id, agent.id, run_group, seed)
                run_id = run_dir.stem
                port = get_port_for_task(competition.id, seed, 0)  # Initial run uses retry_count=0

                # Override the public and private dirs with our lite dataset paths
                competition_path = lite_data_dir / competition.id / "prepared"

                # If the directory structure doesn't exist as expected, skip this competition
                if not competition_path.exists():
                    logger.warning(
                        f"Skipping competition {competition_id} - directory not found: {competition_path}"
                    )
                    continue

                # Create a task for this competition and seed
                task = Task(
                    run_id=run_id,
                    seed=seed,
                    agent=agent,
                    competition=competition,
                    path_to_run_group=run_dir.parent,
                    path_to_run=run_dir,
                    port=port,
                    retry_count=0,  # Initial run
                )
                initial_tasks.append(task)
                logger.info(f"Added task for competition {competition_id}, seed {seed}")
            except Exception as e:
                logger.error(f"Failed to create task for competition {competition_id}: {e}")
                if not QUIET_MODE:
                    traceback.print_exc()

    if not initial_tasks:
        logger.error("No valid tasks found. Exiting.")
        return

    # Number of workers = number of processes (we will run each worker in a separate process)
    logger.info(f"Creating {args.n_workers} workers to serve {len(initial_tasks)} tasks...")
    logger.info(f"Each task can be retried up to {max_retries} times if it fails or hangs")

    # Safely determine allowed CPU count
    try:
        cpu_count = multiprocessing.cpu_count()
        if args.n_workers > cpu_count:
            logger.warning(
                f"Warning: Requested {args.n_workers} workers but only {cpu_count} CPUs available"
            )
            logger.warning(f"This may cause performance issues due to CPU contention")
    except Exception as e:
        logger.warning(f"Could not determine CPU count: {e}")

    # Log system memory information before starting tasks
    if not QUIET_MODE:
        try:
            memory_info = psutil.virtual_memory()
            logger.info(
                f"System memory: {memory_info.percent}% used, {memory_info.available / (1024**3):.2f} GB available"
            )
        except Exception as e:
            logger.warning(f"Could not determine system memory: {e}")

    # Initialize an atomic counter for active workers
    # This helps us track how many workers are currently running
    manager = multiprocessing.Manager()
    active_workers = manager.Value("i", 0)
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # Put initial tasks in the queue
    for task in initial_tasks:
        task_queue.put(task)

    # Initialize results tracking
    tasks_outputs = {}
    retry_counts = defaultdict(int)

    # Create worker processes
    processes = []
    try:
        # Use 'spawn' method for better process isolation and stability
        mp_context = multiprocessing.get_context("spawn")

        # Start worker processes
        for i in range(args.n_workers):
            p = mp_context.Process(
                target=queue_worker, args=(task_queue, result_queue, active_workers)
            )
            p.daemon = True  # Set as daemon so they exit when main process exits
            p.start()
            processes.append(p)
            logger.info(f"Started worker process {i+1}/{args.n_workers} with PID {p.pid}")

        # Process results as they come in, and handle retries
        completed_tasks = 0
        total_tasks = len(initial_tasks)

        while completed_tasks < total_tasks:
            try:
                # Get result with a timeout to allow checking for keyboard interrupts
                try:
                    run_id, task_output = result_queue.get(timeout=5)
                except Exception:
                    # Check if any workers are still alive
                    if all(not p.is_alive() for p in processes) and result_queue.empty():
                        logger.error("All worker processes died unexpectedly")
                        break
                    continue

                # Process the result
                tasks_outputs[run_id] = task_output

                # Check if task succeeded
                if task_output.get("success", False):
                    logger.info(f"Task {run_id} completed successfully")
                    completed_tasks += 1
                else:
                    # Get error details
                    error = task_output.get("error", {})
                    retry_suggested = error.get("retry_suggested", False)
                    retry_count = task_output.get("retry_count", 0)

                    # Check if we should retry
                    if retry_suggested and retry_count < max_retries:
                        # Create a new task with incremented retry count
                        for task in initial_tasks:
                            if task.run_id == run_id:
                                # Get new port for the retry
                                new_port = get_port_for_task(
                                    task.competition.id, task.seed, retry_count + 1
                                )

                                # Create retry task
                                retry_task = Task(
                                    run_id=task.run_id,
                                    seed=task.seed,
                                    agent=task.agent,
                                    competition=task.competition,
                                    path_to_run_group=task.path_to_run_group,
                                    path_to_run=task.path_to_run,
                                    port=new_port,
                                    retry_count=retry_count + 1,
                                )

                                # Add jitter to avoid all retries starting at once
                                jitter = random.uniform(2, 10)
                                logger.info(
                                    f"Will retry task {run_id} in {jitter:.1f} seconds (attempt {retry_count + 1}/{max_retries})"
                                )
                                time.sleep(jitter)

                                # Put retry task in queue
                                task_queue.put(retry_task)
                                retry_counts[run_id] += 1
                                break
                    else:
                        if not retry_suggested:
                            logger.info(f"Task {run_id} failed, but retry not suggested")
                        elif retry_count >= max_retries:
                            logger.warning(
                                f"Task {run_id} failed after {retry_count} retries, giving up"
                            )

                        # Mark as completed even though it failed
                        completed_tasks += 1

                # Log progress
                logger.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed")

            except KeyboardInterrupt:
                logger.critical("Main process received keyboard interrupt, terminating workers...")
                break

        # Signal workers to exit - removed task_queue.close() since manager.Queue doesn't have that method
        # Instead, we rely on the queue_worker loop to exit when task_queue is empty and active_workers is 0

    except Exception as e:
        logger.critical(f"Error in main process: {e}")
        if not QUIET_MODE:
            logger.critical(traceback.format_exc())
    finally:
        # Clean up processes
        for p in processes:
            if p.is_alive():
                logger.info(f"Terminating worker process {p.pid}")
                p.terminate()
                time.sleep(0.5)
                if p.is_alive():
                    logger.warning(f"Killing worker process {p.pid}")
                    p.kill()

        # Wait for processes to exit
        for p in processes:
            p.join(timeout=5)

    # Print retry statistics
    if retry_counts and not QUIET_MODE:
        logger.info("Retry statistics:")
        for run_id, count in retry_counts.items():
            logger.info(f"  {run_id}: {count} retries")

    # Generate metadata.json
    metadata = {
        "run_group": run_group,
        "created_at": get_timestamp(),
        "runs": tasks_outputs,
        "retry_counts": dict(retry_counts),
    }

    run_group_dir = get_runs_dir() / run_group
    with open(run_group_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=False, default=str)

    # Log summary of results
    successful_runs = sum(1 for output in tasks_outputs.values() if output.get("success", False))
    failed_runs = len(initial_tasks) - successful_runs
    logger.info(f"Run group completed: {successful_runs} successful, {failed_runs} failed")
    logger.info(f"Total spookyretries: {sum(retry_counts.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent on a set of competitions in a Docker container."
    )
    parser.add_argument(
        "--agent-id",
        help="Agent ID of the agent to run.",
        type=str,
    )
    parser.add_argument(
        "--competition-set",
        type=str,
        required=True,
        help="Path to a text file with a single competition ID on each line",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to run in parallel",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=False,
        default=1,
        help="Number of seeds to run for each competition",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        required=False,
        default=3,
        help="Maximum number of retries for failed or hanging tasks",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        required=False,
        default=5000,
        help="Base port number to use for grading servers",
    )
    parser.add_argument(
        "--enable-checkpoints",
        action="store_true",
        help="Enable automatic checkpointing of runs directory",
        default=False,
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="How often to checkpoint in minutes",
    )
    parser.add_argument(
        "--retain",
        help="Whether to retain the container after the run instead of removing it.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the directory where all assets associated with the run are stored.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--quiet",
        help="Reduce logging verbosity to show only warnings and errors",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser.parse_args()
    logger = get_logger(__name__)

    try:
        main(args)
    except KeyboardInterrupt:
        logger.critical("Program interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        if not args.quiet:
            logger.critical(traceback.format_exc())
        sys.exit(1)
