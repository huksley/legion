"""Experiment resources around ordinary concurrent Harbor trials."""

import asyncio
import importlib.metadata
import math
import shutil
import signal
import sys
import threading
import uuid
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

from legion.models import NO_KEY_ENV, loopback_url

from .config import PI_VERSION, SANDBOXES, Config
from .harnesses.pi import attach_pi_setup, terminal_pi_error
from .instructions import build_agent_instructions
from .networking import TrialNetworks
from .resources import Resources
from .results import now, safe_error, write_json
from .sources import download_task, resolve_task
from .tools.bundle import build_tool_bundle
from .tools.setup import TOOLS_MANIFEST, attach_tool_setup


def environment_config(config):
    backend = config.environment.backend
    if backend == "hf-sandbox":
        # The sandbox's idle timeout doesn't count a running foreground command as
        # activity, and Pi's whole run is one command, so it must outlast the run.
        # delete=True kills the sandbox (an HF Job) when the trial ends.
        return EnvironmentConfig(
            type=backend,
            kwargs={
                "flavor": config.environment.flavor,
                "job_timeout": math.ceil(config.limits.run_timeout_seconds) + 60,
            },
            delete=True,
        )
    if backend == "e2b":
        # E2B's timeout is the sandbox's whole lifetime; stop kills it when the trial ends.
        return EnvironmentConfig(
            type=backend,
            kwargs={"sandbox_timeout_secs": math.ceil(config.limits.run_timeout_seconds) + 60},
            delete=True,
        )
    # Podman Compose force-removes shared images with `down --rmi local`.
    # delete=False still removes trial containers; job cleanup owns images.
    return EnvironmentConfig(type=backend, delete=backend != "podman")


# Where a podman container reaches the machine it runs on: podman puts the name in every
# container's /etc/hosts (the host's address; a server must listen there, not only on 127.0.0.1)
PODMAN_HOST = "host.containers.internal"
DOCKER_DESKTOP_HOST = "host.docker.internal"  # the Mac, from a Docker Desktop container


def container_host(backend):
    """The name a container of `backend` reaches this machine by, or None where `localhost`
    cannot be rewritten (docker on Linux shares no such name by default)."""
    if backend == "podman":
        return PODMAN_HOST
    if backend == "docker" and sys.platform == "darwin":
        return DOCKER_DESKTOP_HOST
    return None


def agent_url(url, backend):
    """`url` as an agent's container reaches it: with podman, `localhost` (this machine) is
    PODMAN_HOST; with Docker Desktop on macOS, DOCKER_DESKTOP_HOST. Other URLs, and other
    backends, as written."""
    host = container_host(backend)
    if host is None or not loopback_url(url):
        return url
    parts = urlsplit(url)
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit(parts._replace(netloc=host + port))


def localize_resources(trial_config, backend):
    """After the resources bound an agent: their `*_URL` variables that name this machine, as
    the agent's container reaches it (agent_url), and that host allowed."""
    host = container_host(backend)
    if host is None:
        return
    env = trial_config.agent.env
    for name, value in list(env.items()):
        if name.endswith("_URL") and isinstance(value, str) and loopback_url(value):
            env[name] = agent_url(value, backend)
            if host not in trial_config.agent.extra_allowed_hosts:
                trial_config.agent.extra_allowed_hosts.append(host)


def build_trial_config(
    config, resolved, run_dir, name, agent_instructions, api_key, inference=None, run=None
):
    """`inference`: the agent's endpoints (default: the config's); the agent `name` of `run`
    (default: the run dir's name, the run id) calls one of its URLs
    (`Inference.endpoint_for`), and its container may reach all of them."""
    inference = inference or config.inference
    run = run_dir.name if run is None else run
    backend = config.environment.backend
    key_name = inference.key_env or NO_KEY_ENV
    # the id as the servers take it (`<model>:<provider>` on the router), unparsed
    model_name = inference.request_model(config.model)
    if config.harness == "opencode":
        from .harnesses.opencode import OPENCODE_VERSION, PROVIDER

        # OpenCode names a model `<provider>/<model>`: one generated OpenAI-compatible
        # provider for the run's endpoint (legion.harbor.harnesses.opencode)
        agent_name, model_name = "opencode", f"{PROVIDER}/{model_name}"
        agent_kwargs = {"version": OPENCODE_VERSION}
    else:
        agent_name = "pi"
        agent_kwargs = {"version": PI_VERSION, "max_turns": config.limits.agent_max_turns}
        if inference.reasoning_effort is not None:
            agent_kwargs["reasoning_effort"] = inference.reasoning_effort
    return TrialConfig(
        task=(
            TaskConfig(path=resolved.local_path)
            if resolved.local_path is not None
            else TaskConfig(
                **resolved.task_id.model_dump(),
                download_dir=run_dir / "state" / "sources",
            )
        ),
        trial_name=f"{run_dir.name}-{name}",
        trials_dir=run_dir / "trials",
        agent=AgentConfig(
            name=agent_name,
            override_timeout_sec=config.limits.agent_timeout_seconds,
            model_name=model_name,
            skip_model_parsing=True,
            base_url=agent_url(inference.endpoint_for(run, name), backend),
            model_api_key_name=key_name,
            model_api_format="openai_chat_completions",
            env={
                key_name: api_key,
                "LEGION_TOOLS_MANIFEST": TOOLS_MANIFEST,
            },
            extra_allowed_hosts=sorted(
                {urlsplit(agent_url(u, backend)).hostname for u in inference.endpoints}
            ),
            kwargs=agent_kwargs,
        ),
        environment=environment_config(config),
        extra_instructions=[agent_instructions],
    )


async def run_experiment(config: Config):
    """Return (result directory, exit code); always finalize owned resources."""
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    # Embedded callers retain ownership of any existing signal handler.
    handle_sigterm = (
        threading.current_thread() is threading.main_thread()
        and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    )

    def cancel():
        if not current.cancelling():
            current.cancel()

    if handle_sigterm:
        loop.add_signal_handler(signal.SIGTERM, cancel)
    try:
        return await _run_experiment(config)
    finally:
        if handle_sigterm:
            loop.remove_signal_handler(signal.SIGTERM)


async def _run_experiment(config: Config):
    api_key = config.api_key()
    config.check_model_connection(api_key)
    instructions = (
        config.team_file.read_text(encoding="utf-8")
        if config.team_file is not None
        else ""
    )
    role_instructions = {
        name: role.role_file.read_text(encoding="utf-8")
        if role.role_file is not None
        else ""
        for name, role in (config.agents.roles or {}).items()
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{config.name[:24]}-{timestamp}-{uuid.uuid4().hex[:6]}"
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, mode=0o700)
    state = run_dir / "state"
    state.mkdir(mode=0o700)
    bundle = build_tool_bundle(config.tools_dir, config.tools, state / "tools.tar.gz")
    inputs = run_dir / "inputs"
    inputs.mkdir()
    roster = config.agents.roster()
    names = list(roster)
    participants = [
        {"name": name, "role": role, "status": "pending", "rewards": None}
        for name, role in roster.items()
    ]
    secrets = [api_key]
    resources = Resources(
        config.resources_dir,
        config.resources,
        run_dir=run_dir,
        agents=names,
        secrets=secrets,
        required=bundle.manifest["requires"],
    )
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": now(),
        "configuration": config.model_dump(mode="json"),
        "versions": {
            name: importlib.metadata.version(name) for name in ("legion", "harbor")
        },
        "pi_version": PI_VERSION,
        "tools": {"sha256": bundle.sha256, **bundle.manifest},
        "resources": resources.records,
        "participants": participants,
        "artifacts": {"tool_bundle": "state/tools.tar.gz"},
    }
    if config.team_file is not None:
        (inputs / "instructions.md").write_text(instructions, encoding="utf-8")
        manifest["artifacts"]["instructions"] = "inputs/instructions.md"
    for name, role in (config.agents.roles or {}).items():
        if role.role_file is not None:
            snapshot = inputs / "roles" / f"{name}.md"
            snapshot.parent.mkdir(exist_ok=True)
            snapshot.write_text(role_instructions[name], encoding="utf-8")
            manifest["artifacts"].setdefault("role_instructions", {})[name] = str(
                snapshot.relative_to(run_dir)
            )
    manifest_path = run_dir / "run.json"
    write_json(manifest_path, manifest)
    print(f"Results: {run_dir}", flush=True)
    networks = None
    tasks = []
    group = None
    code = 0

    def save():
        write_json(manifest_path, manifest)

    async def run_participant(record, trial_config):
        record.update(status="running", started_at=now())
        save()
        trial = None
        result = None
        try:
            trial = await Trial.create(trial_config)
            if config.harness == "pi":
                attach_pi_setup(trial, config.inference)
            if config.tools:
                attach_tool_setup(trial, bundle)
            result = await trial.run()
            record["status"] = (
                "failed" if result.failure_cause is not None else "completed"
            )
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:
            record.update(status="failed", error=safe_error(exc, secrets))
        finally:
            if result is None and trial is not None:
                try:
                    result = trial.result
                except (AttributeError, RuntimeError):
                    # Harbor may fail before initializing its result object.
                    pass
            if result is not None:
                result_data = result.model_dump(mode="json")
                verifier = result_data.get("verifier_result")
                record["rewards"] = verifier.get("rewards") if verifier else None
                record["trial_id"] = result_data.get("id")
                record["agent_execution"] = result_data.get("agent_execution")
                if result_data.get("step_results"):
                    record["steps"] = [
                        {
                            "name": step["step_name"],
                            "rewards": (step.get("verifier_result") or {}).get(
                                "rewards"
                            ),
                            "agent_execution": step.get("agent_execution"),
                            "error": safe_error(
                                RuntimeError(
                                    step["exception_info"]["exception_message"]
                                ),
                                secrets,
                            )
                            if step.get("exception_info")
                            else None,
                        }
                        for step in result_data["step_results"]
                    ]
                if result.exception_info:
                    record["error"] = safe_error(
                        RuntimeError(result.exception_info.exception_message), secrets
                    )
                result_path = (
                    run_dir / "trials" / trial_config.trial_name / "result.json"
                )
                if result_path.exists():
                    record["result_path"] = str(result_path.relative_to(run_dir))
            if record["rewards"] is None and not record.get("steps"):
                record["missing_reward_reason"] = record.get(
                    "error", "Verification did not produce a reward"
                )
            if record["status"] == "completed":
                pi_error = terminal_pi_error(
                    run_dir / "trials" / trial_config.trial_name / "agent" / "pi.txt",
                    max_turns=config.limits.agent_max_turns,
                )
                if pi_error:
                    record.update(
                        status="failed",
                        error=safe_error(RuntimeError(pi_error), secrets),
                    )
                elif record.get("missing_reward_reason"):
                    record.update(
                        status="failed", error=record["missing_reward_reason"]
                    )
            record["finished_at"] = now()
            save()

    try:
        # Resources start under their own limits (a Space build takes minutes),
        # so startup doesn't count against run_timeout_seconds.
        await resources.start()
        async with asyncio.timeout(config.limits.run_timeout_seconds):
            resolved = await resolve_task(config.task)
            manifest["source"] = resolved.manifest()
            source_path = await download_task(resolved, state / "sources")
            shutil.copy2(source_path / "instruction.md", inputs / "task.md")
            manifest["artifacts"]["task_instruction"] = "inputs/task.md"
            # Sandboxes are separate VMs, with no local networks to create.
            if config.environment.backend not in SANDBOXES:
                networks = TrialNetworks(run_id, engine=config.environment.backend)
                await networks.start()
                manifest["networks"] = networks.allocations
            for record in participants:
                name = record["name"]
                agent_instructions = build_agent_instructions(
                    name,
                    roster,
                    bundle.manifest,
                    instructions,
                    role_instructions.get(record["role"], ""),
                    # The run ID only matters to agents sharing resources like the message board
                    run_id=run_id if config.resources else None,
                )
                instructions_file = inputs / f"{name}-instructions.md"
                instructions_file.write_text(agent_instructions, encoding="utf-8")
                record["instructions_path"] = str(
                    instructions_file.relative_to(run_dir)
                )
                trial_config = build_trial_config(
                    config, resolved, run_dir, name, agent_instructions, api_key
                )
                record["resources"] = resources.bind(name, trial_config)
                if (
                    config.environment.backend in SANDBOXES
                    and trial_config.environment.mounts
                ):
                    # Harbor's sandbox backends would silently ignore them.
                    raise ValueError(
                        f"environment.backend {config.environment.backend} can't mount the files or "
                        "sockets a resource asked for"
                    )
                if networks:
                    overlay = await networks.create(
                        name, state / f"{name}-network.json"
                    )
                    trial_config.environment.extra_docker_compose = [overlay]
                tasks.append(
                    asyncio.create_task(
                        run_participant(record, trial_config), name=name
                    )
                )
            save()
            group = asyncio.gather(*tasks)
            done, _ = await asyncio.wait(
                [group, *resources.watchers], return_when=asyncio.FIRST_COMPLETED
            )
            for watcher in resources.watchers:
                if watcher in done:
                    await watcher
            await group
            failed = [
                record["name"]
                for record in participants
                if record["status"] == "failed"
            ]
            code = int(bool(failed))
            manifest["status"] = "failed" if failed else "completed"
            if failed:
                manifest["error"] = "Participants failed: " + ", ".join(failed)
    except TimeoutError:
        code = 1
        manifest.update(status="timed_out", error="Experiment deadline exceeded")
    except asyncio.CancelledError:
        code = 130
        manifest.update(status="interrupted", error="Experiment interrupted")
    except Exception as exc:
        code = 1
        manifest.update(status="failed", error=safe_error(exc, secrets))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if group:
            await asyncio.gather(group, return_exceptions=True)
        for record in participants:
            if record["status"] in ("pending", "running", "cancelled"):
                record.update(
                    status="timed_out"
                    if manifest["status"] == "timed_out"
                    else "cancelled",
                    finished_at=now(),
                    missing_reward_reason=manifest.get("error", "Run stopped"),
                )
        if await resources.finish():
            code = code or 1
            if manifest["status"] == "completed":
                manifest["status"] = "failed"
        try:
            if networks:
                await networks.stop()
        except Exception as exc:
            code = code or 1
            if manifest["status"] == "completed":
                manifest["status"] = "failed"
            manifest["network_cleanup_error"] = safe_error(exc, secrets)
        manifest["finished_at"] = now()
        manifest["exit_code"] = code
        save()
    return run_dir, code
