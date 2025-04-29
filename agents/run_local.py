"""
Run a single competition locally – no inner Docker containers.

This file parallels `agents/run_in_container` but:
  • copies data into a temp workspace
  • starts the grading server as a subprocess
  • invokes the agent’s start.sh directly
  • copies logs / submission back into run_dir
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mlebench.utils import purple

if TYPE_CHECKING:  # only for type hints; avoids circular runtime import
    from agents.registry import Agent
    from mlebench.registry import Competition


def run_locally(
    competition: "Competition",
    agent: "Agent",
    run_dir: Path,
    logger,
) -> Path:
    """
    Execute `agent` on `competition` without Docker.

    * Creates a workspace inside run_dir
    * Starts the grading server (`environment.grading_server`)
    * Runs the agent's start.sh
    * On success, leaves submission / logs / code in run_dir

    Returns
    -------
    Path to run_dir (for symmetry with run_in_container)
    """

    work = run_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. copy public + (if present) private data to workspace
    # ------------------------------------------------------------------
    shutil.copytree(competition.public_dir, work / "data", dirs_exist_ok=True)
    if competition.private_dir.exists():
        private_target = work / "private" / "data" / competition.id / "prepared" / "private"
        private_target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(competition.private_dir, private_target, dirs_exist_ok=True)

    # ------------------------------------------------------------------
    # 2. environment variables expected by grading server & agent
    # ------------------------------------------------------------------
    env = os.environ.copy()
    env.update(
        {
            "COMPETITION_ID": competition.id,
            "SUBMISSION_DIR": str(work / "submission"),
            "LOGS_DIR": str(work / "logs"),
            "CODE_DIR": str(work / "code"),
            "AGENT_DIR": str(work / "agent"),
        }
    )

    # ------------------------------------------------------------------
    # 3. start grading server (background)
    # ------------------------------------------------------------------
    logger.info("Starting grading server ...")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "environment.grading_server",
            "--port",
            "5000",
        ],
        cwd=work,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # wait until health endpoint is up (max 60 s)
    ready = False
    for _ in range(60):
        try:
            import http.client

            conn = http.client.HTTPConnection("localhost", 5000, timeout=1)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                ready = True
                break
        except Exception:
            time.sleep(1)
    if not ready:
        server.terminate()
        raise RuntimeError("Grading server did not start within 60 s")

    # ------------------------------------------------------------------
    # 4. run agent start.sh
    # ------------------------------------------------------------------
    start_script = Path(agent.path) / "start.sh"
    if not start_script.exists():
        raise FileNotFoundError(f"{start_script} not found inside container image")

    logger.info("Running agent start.sh ...")
    result = subprocess.run(
        ["bash", str(start_script)],
        cwd=work,
        env=env,
        text=True,
        capture_output=True,
    )

    # always shut server down
    server.terminate()
    server.wait(timeout=10)

    # stream agent stdout/err to main logger
    logger.info(purple("===== agent stdout ====="))
    logger.info(result.stdout)
    logger.info(purple("===== agent stderr ====="))
    logger.info(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Agent failed with exit code {result.returncode}")

    logger.info("Outputs saved in %s", work)
    return work
