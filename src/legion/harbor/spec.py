"""legion's run spec: the upstream `Config` plus legion's blocks (loop, ramps, workers, compute, spawn).

One `spec.yaml` (plus its prompt files) per run. An upstream config loads unchanged.
`Spec.load` fills `${VAR}` / `${VAR:-default}` in string values and inlines prompt files.
`compile_spec` turns a Spec into the `SwarmSpec` legion's controller and workers run: one type
per role, `harness: harbor`, each agent one Harbor Trial on the upstream Harbor task path.

Each role is a legion type, so a loop's `while.budget_usd` is per role; `limits.budget_usd` is
the run's.
"""

import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from harbor.utils.env import is_sensitive_env_key
from pydantic import AfterValidator, BeforeValidator, Field, field_validator, model_validator

from legion.buckets import BucketId
from legion.models import (
    ComputeSpec,
    DashboardSpec,
    Inference,
    LoopSpec,
    Namespace,
    SpawnRule,
    SwarmSpec,
    TypeSpec,
    WorkerKnobs,
    load_spec_yaml,
    max_workers,
    parse_duration,
)
from legion.world.protocol import ADMIN_TOKEN_ENV

from .config import (
    E2B_SETUP_S,
    SANDBOXES,
    AgentSettings,
    Config,
    EnvironmentSettings,
    Limits,
    RoleName,
    RoleSettings,
    Settings,
)

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
# the tools a role without `spawn:` rules is not shown (it may still prompt its parent)
SPAWNING = ("spawn.spawn", "spawn.remove")
# an `hf_jobs` server's lifetime without `timeout`: its startup_timeout, the run's
# run_timeout_seconds and this margin (the run's end, the verifiers); the run cancels it sooner
SERVER_MARGIN_S = 900


def seconds(value):
    """'30s', '10m', '48h', '2d' or a number of seconds -> seconds."""
    if isinstance(value, int | float):
        return value
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", str(value))
    if not m:
        raise ValueError(f"bad duration {value!r}; use e.g. 30s, 10m, 48h")
    return float(m[1]) * _UNITS[m[2]]


Duration = Annotated[float, BeforeValidator(seconds), Field(gt=0)]
# [input, output, cached] USD per 1M tokens; Harbor reports no cost for router models.
Pricing = Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=3, max_length=3)]


def legion_duration(value: float) -> str:
    """Seconds as legion writes durations: a whole number of a unit."""
    return f"{value:.0f}s" if value == int(value) else f"{round(value * 1000)}ms"


class Ramp(Settings):
    """Start `start` agents of the role, then `add` more every `every`, up to the role's count."""

    start: int = Field(ge=0, strict=True)
    add: int = Field(ge=1, strict=True)
    every: Duration


class Loop(LoopSpec):
    """legion's loop (docs/loop.md) plus `resume_prompt`, the type's prompt after a restart.
    Board wakes are `message_board.<kind>`."""

    resume_prompt: str = (
        "Your session was restarted. Check your notes and state, and continue your work."
    )


class Workers(Settings):
    """How legion packs agents on workers: main's per-type keys (unset: main's defaults). A
    role's `workers:` overrides keys. `initial` + `max_extra`: worker slots for spawned agents."""

    agents_per_worker: int | None = Field(default=None, ge=1, strict=True)
    max_working_per_worker: int | None = Field(default=None, ge=1, strict=True)
    flavor: str | None = None
    worker_lifetime: str | None = None
    lifetime_jitter: str | None = None
    drain_window: str | None = None
    idle_grace: str | None = None
    turn_timeout: str | None = None
    worker: WorkerKnobs | None = None
    # the workers' own env: what a task's `[environment.env]` `${VAR}` reads, as Harbor fills
    # it from the worker's environment. Stored in the spec, so compile_spec refuses
    # secret-looking names (a task secret goes in the worker host's env, e.g. Slurm's run.env)
    env: dict[str, str] | None = None
    initial: int | None = Field(default=None, ge=0, strict=True)
    max_extra: int | None = Field(default=None, ge=0, strict=True)

    @field_validator(
        "worker_lifetime", "lifetime_jitter", "drain_window", "idle_grace", "turn_timeout"
    )
    @classmethod
    def _duration(cls, value):
        if value is not None:
            parse_duration(value)
        return value


class Role(RoleSettings):
    count: int = Field(default=1, ge=0, strict=True)  # 0: spawned agents only (`spawn:`)
    model: str | None = Field(default=None, min_length=1)  # the run's model otherwise
    inference: Inference | None = None  # its own endpoints (a whole block); the run's otherwise
    ramp: Ramp | None = None
    loop: Loop | None = None  # replaces the run's loop for this role
    workers: Workers | None = None  # overrides keys of the run's workers
    pricing: Pricing | None = None  # its own `model`'s; budgets need it (else the run's)
    # `naming: words` only: put before every name of the role, e.g. "steward-" (TypeSpec's)
    name_prefix: Annotated[str | None, AfterValidator(TypeSpec._name_prefix)] = None

    @model_validator(mode="after")
    def ramp_within_count(self):
        if self.ramp is not None and self.ramp.start > self.count:
            raise ValueError("ramp.start must be <= count")
        return self


class Agents(AgentSettings):
    roles: dict[RoleName, Role] | None = Field(default=None, min_length=1)


class SpecLimits(Limits):
    trial_hours: float = Field(default=23, gt=0)  # a new Trial after this; sandboxes cap at 24 h
    budget_usd: float | None = Field(default=None, ge=0)  # the whole run
    alive: int | None = Field(default=None, ge=0, strict=True)  # agents not removed
    spawned_total: int | None = Field(default=None, ge=0, strict=True)


class Spec(Config):
    stop_on_success: bool = False
    agents: Agents = Field(default_factory=Agents)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    limits: SpecLimits = Field(default_factory=SpecLimits)
    loop: Loop | None = None
    pricing: Pricing | None = None
    workers: Workers = Field(default_factory=Workers)
    # Where legion's workers run (docs/local.md); an upstream config runs here, on its engine.
    compute: ComputeSpec = Field(default_factory=lambda: ComputeSpec(kind="local"))
    namespace: Namespace = ""  # bills Jobs, sandboxes and router calls (unset: each token's account)
    bucket_id: BucketId = ""
    dashboard: DashboardSpec | None = None
    spawn: dict[str, SpawnRule] = Field(default_factory=dict)
    # agent names: legion's random `<adjective>-<noun>` (words), or the upstream roster() names
    # (`agent-001`, ... numbered across the roles in order) for a task that relies on them
    naming: Literal["words", "consecutive-id"] = "words"

    @field_validator("compute", mode="before")
    @classmethod
    def _compute_short(cls, value):
        return {"kind": value} if isinstance(value, str) else value

    def roles(self):
        """Role name -> Role; `agents.count` is one role, `agent`, whose agents have no role."""
        return self.agents.roles or {"agent": Role(count=self.agents.count)}

    def type_loops(self):
        """Each role's loop: its own, else the run's. Roles sharing the run's loop split its
        `final.reserve_usd`, which is held back once (legion sums its types' reserves)."""
        roles = self.roles()
        shared = sum(r.loop is None for r in roles.values())
        out = {}
        for name, role in roles.items():
            loop = role.loop or self.loop
            if loop is not None and role.loop is None and loop.final is not None:
                final = loop.final.model_copy(
                    update={"reserve_usd": loop.final.reserve_usd / shared}
                )
                loop = loop.model_copy(update={"final": final})
            if loop is not None:
                out[name] = loop
        return out

    @model_validator(mode="after")
    def checks(self):
        limits, backend = self.limits, self.environment.backend
        if self.stop_on_success and self.success_check is None:
            raise ValueError("stop_on_success requires success_check")
        if self.success_check and backend not in ("docker", "podman"):
            raise ValueError("success_check requires a docker or podman sidecar")
        loops = self.type_loops().values()
        for loop in loops:
            try:
                until = parse_duration(loop.while_.until).total_seconds()
            except (TypeError, ValueError):  # unset, or a time of day
                until = None
            if until is not None and limits.run_timeout_seconds <= until:
                raise ValueError("limits.run_timeout_seconds must be longer than loop.while.until")
            if loop.while_.until is None and "run_timeout_seconds" not in limits.model_fields_set:
                raise ValueError(
                    "a loop without while.until runs for limits.run_timeout_seconds: set it "
                    "(the default is 900 s, 15 minutes) or while.until"
                )
        shared = sum(role.loop is None for role in self.roles().values())
        if self.loop is not None and self.loop.while_.budget_usd is not None and shared > 1:
            raise ValueError(
                f"loop.while.budget_usd would apply to each of the {shared} roles sharing the "
                "run's loop: set limits.budget_usd for the run, or a budget in each role's loop"
            )
        reserve = sum(loop.final.reserve_usd for loop in loops if loop.final)
        if limits.budget_usd is not None and reserve > limits.budget_usd:
            raise ValueError(
                f"loop final.reserve_usd ({reserve:g}) is more than limits.budget_usd"
            )
        for name, role in self.roles().items() if backend in SANDBOXES else ():
            turn = ((role.workers and role.workers.turn_timeout) or self.workers.turn_timeout
                    or TypeSpec.model_fields["turn_timeout"].default)
            setup = E2B_SETUP_S if backend == "e2b" else 0  # in an e2b sandbox's lifetime
            if limits.trial_hours * 3600 + parse_duration(turn).total_seconds() + setup >= 86400:
                raise ValueError(
                    f"{backend} caps a sandbox at 24 h: limits.trial_hours plus {name}'s "
                    f"turn_timeout ({turn}, a Trial's last turn)"
                    f"{' and 10 min of setup' if setup else ''} must stay below it"
                )
        if self.naming == "consecutive-id" and (
                named := [n for n, r in self.roles().items() if r.name_prefix]):
            raise ValueError(f"agents.roles.{named[0]}.name_prefix needs naming: words "
                             "(consecutive-id names are agent-001, ...)")
        if backend in SANDBOXES and "spawn" in self.resources:
            raise ValueError("resources.spawn needs a mount: use backend docker or podman")
        if backend in SANDBOXES and (self.resources.get("message_board") or {}).get("rules"):
            # no mounts, no rule proxy: the board runs its rules unenforced and warns
            self.resources["message_board"].setdefault("transport", "http")
        budgets = limits.budget_usd is not None or any(
            loop.while_.agent_usd is not None or loop.while_.budget_usd is not None
            for loop in loops
        )
        if budgets and self.pricing is None:
            raise ValueError("budgets need pricing: [input, output, cached] USD per 1M tokens")
        for name, role in self.roles().items():
            if budgets and role.model not in (None, self.model) and role.pricing is None:
                raise ValueError(f"agents.roles.{name} has its own model: budgets need its pricing")
        if self.compute.kind == "hf-jobs" and backend not in SANDBOXES:
            raise ValueError("compute hf-jobs cannot run containers: use backend hf-sandbox or e2b")
        if self.compute.kind == "slurm" and backend == "docker":
            raise ValueError("compute slurm runs rootless: use backend podman or hf-sandbox")
        if self.compute.kind == "slurm" and backend == "e2b":
            # shortcut: the E2B client reads E2B_API_KEY from os.environ only, and a Slurm worker
            # has it in LEGION_WORKER_SECRETS_FILE; pass api_key from secret_env() to lift this
            raise ValueError("compute slurm keeps E2B_API_KEY out of the worker's environment, "
                             "where the E2B client reads it: use backend podman or hf-sandbox")
        return self

    @classmethod
    def load(cls, path: Path, env=None):
        """`env` fills `${VAR}` (default: the process environment)."""
        path = Path(path).resolve()
        here = path.parent
        # YAML 1.2 booleans, as legion's specs: `on:` stays a key
        data = load_spec_yaml(path.read_text())
        spec = cls.model_validate(substitute(data, os.environ if env is None else env))
        # as the upstream Config.load: a path set in the YAML is relative to it
        for field in ("output_dir", "tools_dir", "resources_dir"):
            value = getattr(spec, field)
            if field in spec.model_fields_set:
                value = here / value.expanduser()
            setattr(spec, field, value.resolve())
        if spec.team_file is not None:
            spec.team_file = (here / spec.team_file.expanduser()).resolve()
        roles = (spec.agents.roles or {}).values()
        for role in roles:
            if role.role_file is not None:
                role.role_file = (here / role.role_file.expanduser()).resolve()
        for loop in (spec.loop, *(role.loop for role in roles)):
            for prompted in (loop, loop and loop.final):
                if prompted and prompted.prompt_file is not None:
                    prompted.prompt = (here / Path(prompted.prompt_file).expanduser()).read_text(
                        encoding="utf-8"
                    )
                    prompted.prompt_file = None  # run.json carries the text
        return spec


def compile_spec(spec: Spec, task: dict, bundle=None, folders: str | None = None) -> SwarmSpec:
    """The SwarmSpec legion runs `spec` as: one type per role, `harness: harbor`, whose `harbor:`
    block holds what a worker needs to start each agent's Trial. `task` is the pinned task
    (`ResolvedTask.manifest()`); `bundle` the built tool bundle (None: no tools); `folders` the
    version of `pack_folders` (None: workers read tools_dir and resources_dir where they are)."""
    team = spec.team_file.read_text(encoding="utf-8") if spec.team_file is not None else ""
    roles, loops = spec.roles(), spec.type_loops()
    label = {name: name if spec.agents.roles else None for name in roles}
    # the folders: a worker builds the tool bundle and loads the resource modules from them
    # inference is the types' (`resolved_model`), outside the harbor block: `legion update` may
    # change it while the run's Trials keep the rest
    config = spec.model_dump(mode="json", include={
        "task", "environment", "tools_dir", "tools", "resources_dir", "namespace", "success_check"})
    if spec.success_check is None:
        config.pop("success_check", None)
    if spec.harness != "pi":  # the Trial's agent (legion.worker.adapters.harbor.AGENTS)
        config["harness"] = spec.harness
    if spec.stop_on_success:
        config["stop_on_success"] = True
    config["limits"] = spec.limits.model_dump(
        include={"agent_max_turns", "agent_timeout_seconds", "trial_hours"}
    )
    # the inference key goes into every agent's container: never the token a remote worker
    # syncs the bucket and starts sandboxes with
    remote = spec.environment.backend in SANDBOXES or spec.compute.kind in ("slurm", "hf-jobs")
    # the blocks the roles run on: a role's own, else the run's (unused when every role has one)
    blocks = dict((f"agents.roles.{n}.inference", r.inference) if r.inference
                  else ("inference", spec.inference) for n, r in roles.items())
    for where, inference in blocks.items():
        if remote and inference.key_env == "HF_TOKEN":
            raise ValueError(f"{where}.api_key_env: HF_TOKEN is the workers' Hub token (bucket, "
                             "sandboxes; the router's default key) and would reach every "
                             "agent's container: name an inference-only token, e.g. "
                             "HF_INFERENCE_TOKEN")
    board = spec.resources.get("message_board") or {}
    if remote and board.get("space_id") and board.get("private") is not False:
        raise ValueError("resources.message_board.private: set it to false: a private Space's "
                         "gate needs the workers' Hub token in every agent's container")
    for where, w in [("workers", spec.workers)] + [
            (f"agents.roles.{n}.workers", r.workers) for n, r in roles.items() if r.workers]:
        if bad := sorted(k for k in w.env or {} if is_sensitive_env_key(k)):
            # Harbor's check is a loose substring match: KEY, TOKEN, SECRET, PASSWORD,
            # CREDENTIAL, AUTH anywhere in a name refuse it, harmless or not
            raise ValueError(f"{where}.env: {', '.join(bad)} cannot be set here: the block is "
                             "stored in the spec and lands in the worker job's plain env, so "
                             "names containing KEY/TOKEN/SECRET/PASSWORD/CREDENTIAL/AUTH are "
                             "refused (even if harmless). Put a task secret in the worker host's "
                             "environment instead: on Slurm add NAME=value to run.env, which "
                             "every worker job sources, and the task's ${NAME} reads it there. "
                             "On hf-jobs only the inference key and resource tokens are passed "
                             "as secrets")
    for name, options in spec.resources.items():
        if options.get("token_env") == ADMIN_TOKEN_ENV:
            raise ValueError(f"resources.{name}.token_env: {ADMIN_TOKEN_ENV} is the world's "
                             "admin secret, never a worker's: name the world's agent secret "
                             "(its LEGION_WORLD_TOKEN)")
    # besides the inference key (the provider's): what the workers read on their own host
    # sandboxes start with the worker's HF token (hf-sandbox) or E2B key (e2b); an HF Jobs
    # worker needs the HF token for its bucket either way
    backend = spec.environment.backend
    secrets = resource_secrets(spec) + (
        ["HF_TOKEN"] if backend == "hf-sandbox" or spec.compute.kind == "hf-jobs" else []) + (
        ["E2B_API_KEY"] if backend == "e2b" else [])
    secrets = list(dict.fromkeys(secrets))  # a Space board's HF_TOKEN may be the sandboxes' too
    def lifetime(inference: Inference | None) -> Inference | None:
        """An `hf_jobs` block's `timeout` filled in: the job outlives the run by a margin."""
        hj = inference.hf_jobs if inference is not None else None
        if hj is None or hj.timeout is not None:
            return inference
        s = (parse_duration(hj.startup_timeout).total_seconds() + spec.limits.run_timeout_seconds
             + SERVER_MARGIN_S)
        return inference.model_copy(update={"hf_jobs": hj.model_copy(
            update={"timeout": legion_duration(math.ceil(s))})})

    # [input, output, cached] by model: a role's own model at its own pricing, any other at the
    # run's (none: $0)
    prices = {"*": spec.pricing} | {
        role.model: role.pricing for role in roles.values() if role.model and role.pricing}
    types = {}
    for name, role in roles.items():
        model = role.model or spec.model
        packing = spec.workers.model_dump(exclude_none=True)
        packing |= role.workers.model_dump(exclude_none=True) if role.workers else {}
        slots = {k: packing.pop(k) for k in ("initial", "max_extra") if k in packing}
        ramp, loop = role.ramp, loops.get(name)
        if loop is not None and loop.while_.until is None:  # the run's timeout ends the loop
            until = legion_duration(spec.limits.run_timeout_seconds)
            loop = loop.model_copy(update={"while_": loop.while_.model_copy(
                update={"until": until})})
        tools = bundle.manifest if bundle is not None else {}
        if tools and name not in spec.spawn:
            tools = {**tools, "tools": {k: v for k, v in tools["tools"].items()
                                        if k not in SPAWNING}}
        types[name] = {
            **packing,
            "harness": "harbor",
            "model": model,
            "inference": lifetime(role.inference),
            "pricing": prices.get(model) or prices["*"],
            "name_prefix": role.name_prefix,
            "replicas": 0 if ramp else role.count,  # the ramp raises them
            "replicas_ramp": ramp and {
                "start": ramp.start, "add": ramp.add, "every": legion_duration(ramp.every),
                "max": role.count,
            },
            "workers": slots or None,
            "max_turns": spec.limits.agent_max_turns,
            "secrets": secrets,
            "loop": loop and loop.model_dump(exclude={"resume_prompt"}),
            "resume_prompt": loop and loop.resume_prompt,
            "harbor": {
                "task": task,
                "config": {**config, "model": model},
                "team": team,
                "role": label[name],
                "role_text": role.role_file.read_text(encoding="utf-8") if role.role_file else "",
                "tools": tools,
                "tools_sha256": bundle.sha256 if bundle is not None else None,
                "resources": spec.resources,
                # consecutive names and roles; words: none here, each assign carries the run's
                "roster": spec.agents.roster() if spec.naming == "consecutive-id" else {},
                "folders": folders,
            },
        }
    swarm = SwarmSpec.model_validate({
        "swarm": spec.name,
        "namespace": spec.namespace,
        "bucket_id": spec.bucket_id,
        "inference": lifetime(spec.inference),
        "types": types,
        "dashboard": spec.dashboard,
        "stop_on_success": spec.stop_on_success,
        "spawn": spec.spawn,
        "limits": spec.limits.model_dump(include={"alive", "spawned_total", "budget_usd"}),
        "compute": spec.compute,
        "naming": spec.naming,
    })
    most = [max_workers(ts.model_copy(update={"replicas": ts.replicas_ramp.max}) if ts.replicas_ramp
                        else ts) for ts in swarm.types.values()]  # a ramp's max, not its start
    if board.get("space_id") and sum(most) > 1:
        # each worker runs its own resources, and each deploy() rotates the board's operator token
        raise ValueError("resources.message_board.space_id: the worker that starts the board "
                         "deploys it, so a run of more than one worker (a role each, or "
                         "agents_per_worker below the count) cannot: deploy the board once and "
                         "set message_board.url and token_env, or fit the run on one worker")
    return swarm


def pack_folders(tools_dir, resources_dir):
    """The run's tools and resources folders (`tools/`, `resources/`; a missing one left out,
    a link inside either refused) as one legion tools bundle (legion.tools.bundle): `legion launch`
    uploads it to the controller as the swarm's tools version, and every worker fetches it by
    that content hash and unpacks it in its own run dir (legion.worker.harbor)."""
    from legion.tools.bundle import pack
    from legion.tools.loader import Manifest

    with tempfile.TemporaryDirectory() as d:
        for name, src in (("tools", Path(tools_dir)), ("resources", Path(resources_dir))):
            if src.is_dir():  # a link inside is kept, so `pack` refuses it (the folder may be one)
                shutil.copytree(src, Path(d) / name, symlinks=True,
                                ignore=shutil.ignore_patterns("__pycache__"))
        return pack(d, Manifest())  # no legion folder tools: Harbor's bundle carries the tools


def resource_secrets(spec: Spec) -> list[str]:
    """The env names resources read a token from on the worker: a plugin's `token_env` (its
    default too) when it connects to a `url` or `repo_id`, HF_TOKEN when it deploys
    a Space (`space_id`). Missing modules are refused before workers or sandboxes start."""
    from .resources import load_module

    names = []
    for name, options in spec.resources.items():
        path = spec.resources_dir / f"{name}.py"
        if not path.is_file():
            raise ValueError(f"resources.{name}: no resource module {path}")
        schema = getattr(load_module(path), "Options", None)
        settings = schema.model_validate(options) if schema is not None else None
        if (getattr(settings, "url", None) or getattr(settings, "repo_id", None)) and getattr(
                settings, "token_env", None):
            names.append(settings.token_env)
        if getattr(settings, "space_id", None):
            names.append("HF_TOKEN")
    return names


# From legion's specrender: only UPPERCASE names are variables, so `{spend}` and `${spend}` in
# prompts pass through; `$${` is a literal `${`.
_VAR = re.compile(r"\$\$(?=\{)|\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def substitute(data, env):
    """Fill placeholders in every string value. A string that is one placeholder takes the
    value's JSON type (`"${N:-20}"` -> 20, `"0123"` stays a string). Raises naming every unset
    variable without a default."""
    missing = set()

    def rep(m):
        name, default = m.groups()
        if name is None:
            return "$"
        value = env.get(name)
        if value or (value is not None and default is None):  # `:-` as in sh: empty = unset
            return value
        if default is not None:
            return default
        missing.add(name)
        return m.group(0)

    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if not isinstance(v, str):
            return v
        out = _VAR.sub(rep, v)
        whole = _VAR.fullmatch(v)
        if whole and whole.group(1) and out:
            try:
                typed = json.loads(out)
            except ValueError:
                return out
            if isinstance(typed, bool | int | float):
                return typed
        return out

    result = walk(data)
    if missing:
        raise ValueError("unset variables without a default: " + ", ".join(sorted(missing)))
    return result
