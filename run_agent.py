import argparse
import json
import logging
import multiprocessing
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from agents.registry import Agent
from agents.registry import registry as agent_registry
from agents.run_local import run_locally
from environment.upload_results import upload_to_s3
from environment.utils_zip import make_filtered_zip

# from agents.run import run_in_container, run_locally
from mlebench.data import is_dataset_prepared
from mlebench.registry import Competition, registry
from mlebench.utils import create_run_dir, get_logger, get_runs_dir, get_timestamp

logger = get_logger(__name__)


@dataclass(frozen=True)
class Task:
    run_id: str
    seed: int
    path_to_run_group: Path
    path_to_run: Path
    agent: Agent
    competition: Competition
    port: int  # Add this line


def worker_process(task: Task):
    # Same logic for running a task, but now in a separate process
    run_logger = get_logger(str(task.path_to_run))
    log_file_handler = logging.FileHandler(task.path_to_run / "run.log")
    log_file_handler.setFormatter(logging.getLogger().handlers[0].formatter)
    # log_file_handler.setFormatter(logging.getLogger().handlers[0].formatter())  # match the formatting we have
    run_logger.addHandler(log_file_handler)
    run_logger.propagate = False

    run_logger.info(
        f"Running seed {task.seed} for {task.competition.id}, agent {task.agent.name}, on port {task.port}"
    )

    logger.info(
        f"Running seed {task.seed} for {task.competition.id}, agent {task.agent.name}, on port {task.port}"
    )
    task_output = {}
    try:
        run_locally(
            competition=task.competition,
            agent=task.agent,
            run_dir=task.path_to_run,
            logger=run_logger,
            main_logger=logger,
            port=task.port,  # Pass the port
        )
        task_output["success"] = True

        run_logger.info(
            f"Finished running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
        )

        logger.info(
            f"Finished running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        run_logger.error(type(e))
        run_logger.error(stack_trace)
        run_logger.error(
            f"Run failed for seed {task.seed}, agent {task.agent.id} and competition {task.competition.id}"
        )
        logger.error(type(e))
        logger.error(stack_trace)
        logger.error(
            f"Run failed for seed {task.seed}, agent {task.agent.id} and competition {task.competition.id}"
        )
        task_output["success"] = False

    # Ensure the run_dir is returned or identified for the main process if needed
    return task.run_id, task_output


def zip_and_upload_runs(every_minutes: int = 10):
    def task():
        i = 1
        while True:
            time.sleep(every_minutes * 60)
            timestamp = f"{i * every_minutes}min"
            zip_name = f"runs_{timestamp}.zip"
            zip_path = Path(zip_name)

            # Zip the `runs/` folder
            # shutil.make_archive(zip_path.stem, "zip", "runs")
            make_filtered_zip(zip_path, "runs")

            # Upload (reuse your logic here)
            upload_to_s3(zip_path, f"{os.environ['AICHOR_OUTPUT_PATH'].rstrip('/')}/{zip_name}")
            # upload_to_local(zip_path)
            logger.info(f"[Checkpoint] Uploaded {zip_name} to output bucket")
            i += 1

    threading.Thread(target=task, daemon=True).start()


def main(args):
    global registry
    registry = registry.set_data_dir(Path(args.data_dir))

    agent = agent_registry.get_agent(args.agent_id)

    run_group = f"{get_timestamp()}_run-group_{agent.name}"

    # Load competition ids and check if all are prepared
    with open(args.competition_set, "r") as f:
        competition_ids = [line.strip() for line in f.read().splitlines() if line.strip()]

    for competition_id in competition_ids:
        competition = registry.get_competition(competition_id)
        if not is_dataset_prepared(competition):
            raise ValueError(f"Dataset for competition `{competition.id}` is not prepared!")

    # Create tasks for each competition and seed
    logger.info(f"Launching run group: {run_group}")
    tasks = []
    base_port = 5000  # Or make this configurable
    task_index = 0
    for seed in range(args.n_seeds):
        for competition_id in competition_ids:
            competition = registry.get_competition(competition_id)
            run_dir = create_run_dir(competition.id, agent.id, run_group)
            run_id = run_dir.stem
            task = Task(
                run_id=run_id,
                seed=seed,
                agent=agent,
                competition=competition,
                path_to_run_group=run_dir.parent,
                path_to_run=run_dir,
                port=base_port + task_index,  # Assign unique port
            )
            tasks.append(task)
            task_index += 1  # Increment for the next task

    # Number of workers = number of processes (we will run each worker in a separate process)
    logger.info(f"Creating {args.n_workers} workers to serve {len(tasks)} tasks...")
    logger.info(f"Assigning ports from {base_port} to {base_port + len(tasks) - 1}")

    # Using multiprocessing Pool to process tasks in parallel
    with multiprocessing.Pool(processes=args.n_workers) as pool:
        results = pool.map(worker_process, tasks)

    # Collect the results and handle output
    tasks_outputs = {}
    for run_id, task_output in results:
        tasks_outputs[run_id] = task_output

    # Generate metadata.json
    metadata = {
        "run_group": run_group,
        "created_at": get_timestamp(),
        "runs": tasks_outputs,
    }

    run_group_dir = get_runs_dir() / run_group
    with open(run_group_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=False, default=str)

    logger.info(f"{args.n_workers} workers ran for tasks in total")


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
        "--data-dir",
        help="Path to the directory containing the competition data.",
        type=str,
        required=False,
        default=registry.get_data_dir(),
    )
    args = parser.parse_args()
    logger = get_logger(__name__)
    # zip_and_upload_runs(every_minutes=10)

    main(args)
