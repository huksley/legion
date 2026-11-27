"""The small, deliberately strict experiment interface."""

import os
import re
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from huggingface_hub import get_token
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legion.models import NO_KEY, Inference

PI_VERSION = "1.5.0"
# Legion's own resource providers and tools, shipped inside the package.
BUILTIN = Path(__file__).resolve().parent / "builtin"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Selector = Annotated[str, Field(min_length=1)]


class TaskSettings(Settings):
    repo: str
    dataset: str | None = None
    # one task for every agent, or a list: one per agent by position (sources.pin_tasks)
    selector: Selector | Annotated[list[Selector], Field(min_length=1)]
    revision: str = Field(default="main", min_length=1)

    @field_validator("repo")
    @classmethod
    def hub_repository(cls, value):
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or url.netloc != "huggingface.co"
            or not re.fullmatch(r"/datasets/[\w.-]+/[\w.-]+", url.path)
            or url.query
            or url.fragment
        ):
            raise ValueError("task.repo must be an HF dataset repository URL")
        return value

    @field_validator("dataset")
    @classmethod
    def versioned_dataset(cls, value):
        if value is not None and not re.fullmatch(r"[^@\s]+@[^@\s]+", value):
            raise ValueError(
                "task.dataset must contain a name and version: name@version"
            )
        return value


def inference_key(inference: Inference, environ=None) -> str:
    """The inference key for `inference`. HF_TOKEN falls back to the logged-in token
    (`hf auth login`), as for the Space board; servers that take no key get a placeholder, since
    Pi needs a key all the same."""
    environ = os.environ if environ is None else environ
    name = inference.key_env
    if name is None:
        return NO_KEY
    key = environ.get(name) or (get_token() if name == "HF_TOKEN" else None)
    if not key:
        raise ValueError(
            f"Set {name}"
            + (" or run `hf auth login`" if name == "HF_TOKEN" else "")
            + " before running an experiment"
        )
    return key


class RoleSettings(Settings):
    count: int = Field(default=1, ge=1, strict=True)
    role_file: Path | None = None


RoleName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]


class AgentSettings(Settings):
    count: int | None = Field(default=None, ge=1, strict=True)
    roles: dict[RoleName, RoleSettings] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def count_or_roles(self):
        if self.roles is not None and self.count is not None:
            raise ValueError("Set agents.count or agents.roles, not both")
        if self.roles is None and self.count is None:
            self.count = 2
        return self

    def roster(self):
        roles = (
            [name for name, role in self.roles.items() for _ in range(role.count)]
            if self.roles is not None
            else [None] * self.count
        )
        return {f"agent-{i + 1:03d}": role for i, role in enumerate(roles)}


# Backends whose Trials run in a remote VM: no mounts, no local network, no private URLs.
SANDBOXES = ("hf-sandbox", "e2b")
# An e2b sandbox's lifetime is fixed at creation: it also covers the Trial's setup (Pi's install,
# the session's restore), which runs before LegionPi's trial_hours clock starts.
E2B_SETUP_S = 600


class EnvironmentSettings(Settings):
    backend: Literal["docker", "podman", "hf-sandbox", "e2b"] = "docker"
    # Hugging Face Jobs hardware for hf-sandbox, e.g. cpu-basic or cpu-upgrade.
    flavor: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def sandbox_flavor(self):
        if self.backend == "hf-sandbox":
            self.flavor = self.flavor or "cpu-basic"
        elif self.flavor is not None:
            raise ValueError("environment.flavor is only valid with hf-sandbox")
        return self


class Limits(Settings):
    run_timeout_seconds: float = Field(default=900, gt=0)
    agent_max_turns: int = Field(default=30, ge=1, strict=True)
    agent_timeout_seconds: float | None = Field(default=None, gt=0)


class SuccessCheck(Settings):
    """Operator-owned check in a task sidecar, never in the agent container."""

    service: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    command: str = Field(min_length=1)

    @field_validator("service")
    @classmethod
    def sidecar_only(cls, value):
        if value == "main":
            raise ValueError("success_check must run in a trusted sidecar, not main")
        return value


class Config(Settings):
    name: str = Field(default="pi-team", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    # the agent in each Trial: Pi, or OpenCode (legion.harbor.harnesses.opencode)
    harness: Literal["pi", "opencode"] = "pi"
    success_check: SuccessCheck | None = None
    model: str = Field(min_length=1)
    inference: Inference = Field(default_factory=Inference)
    task: TaskSettings
    team_file: Path | None = None
    resources_dir: Path = BUILTIN / "resources"
    # `message_board:` with no settings means the defaults.
    resources: dict[str, dict | None] = Field(default_factory=dict)
    tools_dir: Path = BUILTIN / "tools"
    tools: list[str] = Field(default_factory=list)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    limits: Limits = Field(default_factory=Limits)
    output_dir: Path = Path("results")

    @model_validator(mode="after")
    def resource_defaults(self):
        self.resources = {
            name: options or {} for name, options in self.resources.items()
        }
        board = self.resources.get("message_board")
        # A board comes with all its tools unless `tools` is set; `tools: []`
        # means none.
        if board is not None and "tools" not in self.model_fields_set:
            self.tools = ["message_board.*"]
        if (
            self.environment.backend in SANDBOXES
            and board is not None
            and not board.get("space_id")
            and not board.get("url")  # legion: a board that is up (message_board.py)
        ):
            # Agents reach a local board through a mounted Unix socket, and a
            # sandbox can't mount it. A run without a board is fine.
            raise ValueError(
                f"environment.backend {self.environment.backend} can't reach a local board: set "
                "message_board.space_id to put the board on a Space, or remove "
                "resources.message_board"
            )
        return self

    @classmethod
    def load(cls, path: Path):
        """Paths set in the YAML resolve relative to it. By default, resources and
        tools are Legion's built-in ones and results go to ./results."""
        path = path.resolve()
        spec = cls.model_validate(yaml.safe_load(path.read_text()))
        for field in ("output_dir", "tools_dir", "resources_dir"):
            value = getattr(spec, field)
            if field in spec.model_fields_set:
                value = path.parent / value.expanduser()
            setattr(spec, field, value.resolve())
        if spec.team_file is not None:
            spec.team_file = (path.parent / spec.team_file.expanduser()).resolve()
        for role in (spec.agents.roles or {}).values():
            if role.role_file is not None:
                role.role_file = (path.parent / role.role_file.expanduser()).resolve()
        return spec

    def api_key(self):
        """The inference credential (`inference_key`)."""
        return inference_key(self.inference)

    def check_model_connection(self, api_key):
        from harbor.agents.installed.opencode import OpenCode
        from harbor.agents.installed.pi import Pi

        from legion.models import NO_KEY_ENV

        name = self.inference.key_env or NO_KEY_ENV
        environment = {**os.environ, name: api_key}
        agent, model = Pi, self.inference.request_model(self.model)
        if self.harness == "opencode":
            from .harnesses.opencode import PROVIDER

            agent, model = OpenCode, f"{PROVIDER}/{model}"
        agent.resolve_model_connection_config(
            model,
            lambda key, *others: next(
                (
                    (name, environment[name])
                    for name in (key, *others)
                    if environment.get(name)
                ),
                None,
            ),
            skip_model_parsing=True,
            base_url=self.inference.endpoints[0],
            model_api_key_name=name,
            model_api_format="openai_chat_completions",
        )
