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
    (work / "code").mkdir(exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    (work / "submission").mkdir(exist_ok=True)
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
            "HOME_DIR": str(work),
        }
    )
    PORT = int(os.getenv("MBX_GRADE_PORT", "5000"))

    # ------------------------------------------------------------------
    # container-style compatibility shims for start.sh
    # ------------------------------------------------------------------
    # 1) Provide /home/data  →  pointing to our copied dataset
    # compatibility shims
    # container_home = work / "container_home"
    # container_home.mkdir(parents=True, exist_ok=True)          # <-- create first

    # # now safe to add links and helper files
    # (container_home / "data").symlink_to(
    #     work / "data", target_is_directory=True
    # )

    # # empty instructions.txt so the cp in start.sh succeeds
    # (container_home / "instructions.txt").write_text(
    #     "Local run – no global instructions.\n"
    # )

    # 3) TIME_LIMIT env expected by script
    env.setdefault("TIME_LIMIT", "3600")  # seconds; tweak as you like

    # 4) Put that fake /home/ on PYTHONPATH for any relative imports
    env.setdefault("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{work}:{env['PYTHONPATH']}"

    # ------------------------------------------------------------------
    # 3. start grading server (background)
    # ------------------------------------------------------------------
    logger.info("Starting grading server …")

    # >>> new lines start
    server_log_path = work / "grading_server.log"
    server_log_file = server_log_path.open("w")  # create …/grading_server.log
    # >>> new lines end

    from importlib.util import spec_from_file_location

    repo_root = Path(__file__).resolve().parents[1]  # project root
    server_script = repo_root / "environment" / "grading_server.py"
    assert server_script.exists(), f"{server_script} missing"

    server = subprocess.Popen(
        [sys.executable, str(server_script), "--port", str(PORT)],
        cwd=work,
        env=env,
        stdout=server_log_file,
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
    # start_script = Path(agent.path) / "start.sh"
    # if not start_script.exists():
    #     raise FileNotFoundError(f"{start_script} not found inside container image")
    agent_src_dir = agent.agents_dir / agent.name
    agent_dst_dir = work / "agent"
    shutil.copytree(agent_src_dir, agent_dst_dir, dirs_exist_ok=True)
    start_script = agent_dst_dir / "start.sh"
    if not start_script.exists():
        raise FileNotFoundError(f"{start_script} missing — check YAML start path")

    logger.info("Running agent start.sh ...")
    data_dir_flag = ["data_dir", str(work / "data")]

    # make these flags available to start.sh via an env-var; start.sh can
    # simply append them when it calls `aide`
    env["AIDE_EXTRA_FLAGS"] = " ".join(f"{k}={v}" for k, v in [data_dir_flag])
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
