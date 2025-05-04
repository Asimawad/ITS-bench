from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from agents.utils import parse_env_var_values
from mlebench.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    agents_dir: Path
    start: Path
    kwargs: dict
    env_vars: dict
    kwargs_type: Optional[str] = None

    def __post_init__(self):
        assert isinstance(self.start, Path), "Agent start script must be a pathlib.Path object."
        assert isinstance(self.kwargs, dict), "Agent kwargs must be a dictionary."

        if self.kwargs_type is not None:
            assert isinstance(self.kwargs_type, str), "Agent kwargs_type must be a string."
        else:  # i.e., self.kwargs_type is None
            assert self.kwargs == {}, "Agent kwargs_type must be set if kwargs are provided."

        assert isinstance(self.env_vars, dict), "Agent env_vars must be a dictionary."

        assert self.start.exists(), f"start script {self.start} does not exist."

    @staticmethod
    def from_dict(data: dict) -> "Agent":
        agents_dir = Path(data["agents_dir"])
        try:
            return Agent(
                id=data["id"],
                name=data["name"],
                agents_dir=agents_dir,
                start=agents_dir / data["start"],
                kwargs=data.get("kwargs", {}),
                kwargs_type=data.get("kwargs_type", None),
                env_vars=data.get("env_vars", {}),
            )
        except KeyError as e:
            raise ValueError(f"Missing key {e} in agent config!")


class Registry:
    def get_agents_dir(self) -> Path:
        """Retrieves the agents directory within the registry."""

        return Path(__file__).parent

    def get_agent(self, agent_id: str) -> Agent:
        """Fetch the agent from the registry."""

        agents_dir = self.get_agents_dir()

        for fpath in agents_dir.glob("**/config.yaml"):
            with open(fpath, "r") as f:
                contents = yaml.safe_load(f)

            if agent_id not in contents:
                continue

            logger.debug(f"Fetching {fpath}")

            kwargs = contents[agent_id].get("kwargs", {})
            kwargs_type = contents[agent_id].get("kwargs_type", None)
            env_vars = contents[agent_id].get("env_vars", {})

            # env vars can be used both in kwargs and env_vars
            kwargs = parse_env_var_values(kwargs)
            env_vars = parse_env_var_values(env_vars)

            return Agent.from_dict(
                {
                    **contents[agent_id],
                    "id": agent_id,
                    "name": fpath.parent.name,  # this will be 'aide-deepseek' I guess
                    "agents_dir": agents_dir,
                    "kwargs": kwargs,
                    "kwargs_type": kwargs_type,
                    "env_vars": env_vars,
                }
            )

        raise ValueError(f"Agent with id {agent_id} not found")


registry = Registry()
