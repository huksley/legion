"""Harness `harbor`: each agent is one long-lived Harbor Trial on the upstream Harbor task path
(legion.harbor), in which LegionPi (legion.harbor.agent) runs one `pi` turn per prompt.

The Trial is built as the upstream launcher builds it (build_trial_config, the resource bind,
Trial.create, attach_pi_setup, attach_tool_setup) with LegionPi as the agent, and `trial.run()` runs in the
background for as long as the agent lives here:
- `start` reuses the Trial when it is up; otherwise it starts one, putting back the saved
  sessions when there are any (none for a session to resume: ResumeError), and waits until
  Pi is installed.
- `send` queues the prompt for the next turn (prompts sent during a turn join the next one);
  a Trial that ended on its own (limits.trial_hours) is replaced by a new one for it.
- `stop` kills a turn in flight and ends the burst's event stream; the Trial stays up. When
  /logs/agent is not mounted (hf-sandbox), it also downloads the sessions for the next push.
- `close` ends the Trial: LegionPi returns, Harbor runs the verifier and stops the environment;
  `close(0)` cancels it unverified, and a Trial still starting is cancelled.
On docker and podman each Trial gets its own network, as the upstream launch gives each
participant one (TrialNetworks), removed when the next Trial starts or at `close`.
A Trial that fails (an idle probe, 3 failed turns in a row, a crash) is reported as `error`
and `exited`, after a failed turn for any prompt queued for it; the next prompt starts a new
one. When its first turn on restored sessions failed, the next `start` of that session raises
ResumeError (the runtime then starts a fresh session).

The session files live in `home/sessions` (LegionPi copies them there after each turn and the
adapter again when a Trial ends), where BucketSync pushes and restores them. Files in the
container do not survive a new Trial.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shlex
import shutil
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legion.models import TypeSpec, parse_duration
from legion.secrets_file import exposed
from legion.worker.adapters.base import AdapterEvent, EventQueue, ResumeError
from legion.worker.harbor import TRIAL_RESULTS, keep_results, results_dir

log = logging.getLogger(__name__)

CHECK_START = "\nLEGION_CHECK_START\n"
CHECK_END = "\nLEGION_CHECK_END\n"

CANCEL_WAIT = 60.0  # s close() waits for a cancelled Trial's environment to stop
# the Harbor agent class of each `harness` of a Harbor spec (legion.harbor.config.Config)
AGENTS = {"pi": "legion.harbor.agent:LegionPi",
          "opencode": "legion.harbor.harnesses.opencode:LegionOpenCode"}


class HarborAdapter:
    harness = "harbor"

    def __init__(self, workdir: Path, name: str, spec: TypeSpec, *, resources: Any = None,
                 bundle: Any = None, host: Any = None) -> None:
        self.workdir, self.agent, self.spec = Path(workdir), name, spec
        self.root = self.workdir.parent / "_harbor" / name  # trials/<name>-tN, state/sources
        self.home = self.root / "pi"
        self.session_dir = self.root  # what purge() deletes for an ephemeral agent
        self.resources = resources  # the worker's legion.harbor Resources (bind); None: none
        self.bundle = bundle  # the run's ToolBundle; None: no tools
        self.host = host  # the worker's HarborHost: resources, bundle, task (legion.worker.harbor)
        self.session_id: str | None = None
        self.attempt = 0  # Trials started for this agent here
        self.trial: Any = None
        self._success: dict[str, Any] | None = None
        self._success_error: str | None = None
        self._success_watcher: asyncio.Task | None = None
        self.result: Any = None  # the last ended Trial's TrialResult
        self._run: asyncio.Task | None = None
        self._starting: asyncio.Task | None = None  # a Trial started by send() or a rollover
        self._creating = False  # a Trial is being created (no `_run` for it yet)
        self._prompts: asyncio.Queue = asyncio.Queue()
        self._queue = EventQueue()  # one stream; each burst reads it up to its `exited`
        self._open = False  # a burst is reading events (an `exited` is owed)
        self._closing = False
        self._resume_failed: str | None = None  # the last Trial's --continue failed: why
        self._closes = 0  # close() calls: after one, no Trial starts (the adapter is done)
        self._network: Any = None  # the latest Trial's TrialNetworks (docker, podman)
        self._network_id = uuid.uuid4().hex[:8]  # names and labels this adapter's networks

    def session_files(self, session_id: str | None) -> list[Path]:
        """Every session file of the agent: Pi's `*.jsonl` (one Pi session dir per agent, and
        `--continue` takes up the newest), or OpenCode's data dir as a tree."""
        sessions = self.home / "sessions"
        if self._harness_name() == "opencode":
            return sorted(p for p in sessions.rglob("*") if p.is_file()
                          and not p.name.startswith("."))
        return sorted(sessions.glob("*.jsonl"))

    def _harness_name(self) -> str:
        h = self.spec.harbor
        return (h.config.get("harness") if h is not None else None) or "pi"

    def artifact_files(self) -> list[Path]:
        return [p for p in (self.home / "results").rglob("*") if p.is_file() and not p.is_symlink()]

    def _keep_result(self) -> None:
        """Keep grading evidence in the synced harness home, without credential-bearing config."""
        from legion.harbor.results import write_json

        trial = self.trial
        destination = self.home / "results" / trial.config.trial_name
        destination.mkdir(parents=True, exist_ok=True)
        if self.result is not None:
            summary = self.result.model_dump(mode="json", include={
                "trial_name", "task_name", "task_id", "task_checksum", "verifier_result",
                "verifier_environment_mode", "started_at", "finished_at"})
            summary["exception_type"] = (self.result.exception_info.exception_type
                                         if self.result.exception_info else None)
            write_json(destination / "result.json", summary)
        write_json(destination / "success.json", self._success)
        source = trial.config.trials_dir / trial.config.trial_name / "artifacts"
        for path in source.rglob("*"):
            if path.is_file() and not path.is_symlink():
                target = destination / "artifacts" / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def alive(self) -> bool:
        return self._run is not None and not self._run.done()

    def trial_status(self) -> dict[str, Any] | None:
        """The heartbeat's `trial`: the agent's latest Trial here, if any."""
        if self.trial is None:
            return None
        status = {"id": self.trial.config.trial_name, "attempt": self.attempt, "alive": self.alive()}
        if self.spec.harbor.config.get("success_check"):
            status.update(success=self._success, success_check_error=self._success_error)
        return status

    # ------------------------------------------------------------------ adapter
    async def start(self, workdir: Path, session_id: str | None, spec: TypeSpec, *,
                    env: dict[str, str] | None = None) -> str:
        """`env` is not passed on: a Trial's environment is the task's, plus the inference key,
        the tools and the resources, as the upstream launcher sets them."""
        self.spec = spec
        files = self.session_files(session_id)
        try:
            if session_id and not files:
                raise ResumeError(f"no Pi session files under {self.home / 'sessions'}")
            if session_id and self._resume_failed:
                why, self._resume_failed = self._resume_failed, None
                raise ResumeError(why)
            if self._starting is not None and not self._starting.done():
                try:
                    await asyncio.shield(self._starting)  # its failure is reported in this burst
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
                    # close() cancelled that start: a failed start, not a cancelled burst
                    raise RuntimeError("the Trial start was cancelled: the agent closed") from None
            if not self.alive():
                await self._new_trial(restore=bool(session_id and files))
        except ResumeError:
            self.session_id = None  # the runtime's fresh start gets a new id
            raise
        self._open = True
        self.session_id = session_id or self.session_id or str(uuid.uuid4())
        return self.session_id

    async def send(self, text: str) -> None:
        self._prompts.put_nowait(text)
        self._ensure()

    async def stop(self) -> None:
        """Kill a turn in flight (reported `interrupted`), drop queued prompts and end the
        burst's event stream. The Trial stays up. Idempotent."""
        while not self._prompts.empty():
            self._prompts.get_nowait()
        agent = getattr(self.trial, "agent", None)
        if agent is not None and getattr(agent, "busy", False):
            await agent.kill()
        # unmounted: the next push gets this burst's sessions (a closing Trial's come with
        # Harbor's download as it ends)
        if self._open and self.alive() and not self._closing and agent.channel.ready.is_set():
            await agent.pull_sessions()
        self._exit()

    async def close(self, timeout: float | None = None) -> Any:
        """End the Trial: LegionPi returns, Harbor runs the verifier and stops the environment;
        returns its TrialResult (None if none). After `timeout` s (0: at once; None: no limit),
        or when this call is cancelled, the Trial is cancelled instead: Harbor stops the
        environment unverified, waited for up to CANCEL_WAIT s (a further cancel stops the
        wait). A Trial still starting is cancelled at once, and no Trial starts after a close
        (the worker never reuses a closed adapter)."""
        self._closes += 1
        if self._success_watcher is not None:
            self._success_watcher.cancel()
            await asyncio.gather(self._success_watcher, return_exceptions=True)
        starting = self._starting
        if starting is not None and not starting.done():
            starting.cancel()
            await asyncio.wait({starting})
        run = self._run
        if run is not None and not run.done():
            self._closing = True
            try:
                if timeout != 0 and self.trial.agent.channel.ready.is_set():
                    self._prompts.put_nowait(None)
                    await asyncio.wait({run}, timeout=timeout)
            finally:
                try:
                    if not run.done():
                        if not run.cancelling():  # a second cancel would cut Harbor's cleanup
                            run.cancel()
                        await asyncio.wait({run}, timeout=CANCEL_WAIT)
                finally:
                    self._closing = False
                    self._prompts = asyncio.Queue()
        if run is not None and run.done() and self.spec.harbor.config.get("success_check"):
            try:
                await asyncio.to_thread(self._keep_result)
            except Exception as exc:  # noqa: BLE001 - cleanup must still stop the environment
                error = f"grading artifact preservation failed ({type(exc).__name__})"
                if self._success is not None:
                    self._success["artifact_error"] = error
                self._emit(AdapterEvent("error", error=error))
                log.warning("%s: %s", self.agent, error)
        if run is None or run.done():
            await self._drop_network(self._network)
        if self.resources is not None:  # it leaves this worker (drain, unassign)
            await self.resources.remove([self.agent])
        return self.result

    async def events(self) -> AsyncIterator[AdapterEvent]:
        async for ev in self._queue:
            yield ev

    # ------------------------------------------------------------------ Trials
    def _emit(self, ev: AdapterEvent) -> None:
        self._queue.put(ev)  # between bursts it waits for the next one (a failure, usage)

    def _exit(self) -> None:
        if self._open:
            self._open = False
            self._emit(AdapterEvent("exited"))

    def _ensure(self) -> None:
        """A Trial for the queued prompts, unless one is up or starting (or the session must
        restart: the runtime's next start gets the ResumeError)."""
        if (self.alive() or self._creating or self._closing or self._resume_failed
                or self._prompts.empty()
                or (self._success and self.spec.harbor.config.get("stop_on_success"))):
            return
        self._creating = True  # until the task has the new Trial's `_run`
        self._starting = asyncio.create_task(self._restart(), name=f"trial-{self.agent}")

    async def _restart(self) -> None:
        try:
            await self._new_trial(restore=bool(self.session_id and self.session_files(None)))
        except Exception as e:  # noqa: BLE001 - reported; the next start tries again
            log.warning("%s: new Trial failed: %s", self.agent, e)
            self._fail(f"new Trial failed: {e}")
        finally:
            self._creating = False

    def _fail(self, error: str) -> None:
        """A Trial failed or did not start: prompts queued for it end as a failed turn (the
        runtime's loop sends them again), then `error` and `exited`."""
        queued = False
        while not self._prompts.empty():
            queued = self._prompts.get_nowait() is not None or queued
        if queued and self._open:
            self._emit(AdapterEvent("turn_started"))
            self._emit(AdapterEvent("turn_ended", status="error", error=error))
        self._emit(AdapterEvent("error", error=error))
        self._exit()

    def _instructions(self) -> str:
        from legion.harbor.instructions import build_agent_instructions

        h = self.spec.harbor
        roster = h.roster if self.agent in h.roster else {**h.roster, self.agent: h.role}
        return build_agent_instructions(
            self.agent, roster, {"tools": h.tools.get("tools", {})}, h.team, h.role_text,
            run_id=os.environ.get("SWARM"))  # as the upstream launcher: the run (swarm) a board spans

    def _endpoint(self, channel: Any, base_url: str) -> None:
        """The Trial's endpoint (`base_url`, as build_trial_config picked it) into its channel,
        for Pi's models.json. A changed block restarts the type's workers (controller), so a
        Trial keeps its endpoint."""
        r = self.spec.resolved_model
        channel.base_url = base_url
        channel.context_window = r.context_window
        channel.headers = r.headers

    async def _new_trial(self, restore: bool) -> None:
        """Create the agent's next Trial, run it in the background and wait until LegionPi
        is ready. ResumeError: the sessions could not be put back."""
        from harbor.models.task.id import GitTaskId

        from legion.harbor import launch
        from legion.harbor.agent import Channel
        from legion.harbor.config import E2B_SETUP_S, SANDBOXES, inference_key
        from legion.harbor.envs import ENVIRONMENTS
        from legion.harbor.sources import ResolvedTask
        from legion.harbor.spec import Spec

        h = self.spec.harbor
        if h is None:
            raise RuntimeError("harness harbor needs the type's `harbor:` block")
        if self._closes:  # the exit path closed this agent while its burst got here
            raise RuntimeError("agent closed")
        # A final in-flight check can report success as the old Trial expires.
        # Keep that Trial (and its verifier result) until the controller releases us.
        if self._success_watcher is not None:
            await self._success_watcher
        if self._success and h.config.get("stop_on_success"):
            return
        if self.host is not None:
            await self.host.prepare(self.spec)
            self.resources, self.bundle = self.host.resources, self.host.bundle
        if h.tools.get("tools") and self.bundle is None:
            raise RuntimeError(f"the tool bundle {h.tools_sha256} is not on this worker")
        config = Spec.model_validate(h.config)
        if (per_agent := h.task.get("tasks")) and self.agent not in per_agent:
            raise RuntimeError(f"{self.agent} has no task: the spec gives one to each of "
                               f"{', '.join(per_agent)}")
        own = per_agent[self.agent] if per_agent else h.task["task"]  # only this agent's task
        path = (getattr(self.host, "task_paths", {}).get(self.agent) if per_agent
                else getattr(self.host, "task_path", None))
        task = ResolvedTask(GitTaskId.model_validate(own), h.task["registry_commit"],
                            h.task["dataset"], path)
        trial_seconds = config.limits.trial_hours * 3600
        self.attempt += 1
        inference = self.spec.resolved_model.inference  # the type spec's, not the config's
        swarm = os.environ.get("SWARM")  # the worker's run: with the agent's name, its server
        with exposed():  # the inference key from os.environ; a Slurm worker has a file
            tc = launch.build_trial_config(config, task, self.root, self.agent,
                                           self._instructions(), inference_key(inference),
                                           inference=inference, run=swarm or "")
        # Harbor's compose project is `<trial_name>__env` and its start runs `compose down` on
        # it: the swarm (the worker's SWARM) keeps another run's <agent>-t1 on this host
        # alive; a worker relaunched in this run reuses, and so cleans up, its own names.
        # shortcut: two runs of one swarm name on one host still share names; add a run id then
        tc.trial_name = f"{swarm + '-' if swarm else ''}{self.agent}-t{self.attempt}"
        tc.agent.name, tc.agent.import_path = None, AGENTS[config.harness]
        # LegionPi serves turns for trial_hours; its last turn may take a turn_timeout more
        tc.agent.override_timeout_sec = (
            trial_seconds + parse_duration(self.spec.turn_timeout).total_seconds())
        if config.environment.backend == "hf-sandbox":
            # job_timeout is the sandbox's idle timeout, and a running command is no activity:
            # the upstream launcher's outlasts its one Pi command (the run) by 60 s, legion's one
            # turn (LegionPi probes between turns). Billed to the spec's namespace (unset: the token's owner).
            tc.environment.kwargs.update(
                job_timeout=math.ceil(parse_duration(self.spec.turn_timeout).total_seconds()) + 60,
                namespace=config.namespace or None)
        elif config.environment.backend == "e2b":  # the sandbox's lifetime: the Trial's and setup
            tc.environment.kwargs["sandbox_timeout_secs"] = (
                math.ceil(tc.agent.override_timeout_sec) + E2B_SETUP_S)
        # legion's subclasses: no secret in an argv (docker, podman); hf-sandbox's namespace
        tc.environment.import_path = ENVIRONMENTS[config.environment.backend]
        if self.resources is not None:
            await self.resources.add([self.agent])  # joins here (again: after a move)
            if self._closes:  # a close during the add removed the agent before it was added
                await self.resources.remove([self.agent])
                raise RuntimeError("agent closed")
            bound = self.resources.bind(self.agent, tc)
            launch.localize_resources(tc, config.environment.backend)
            if tc.environment.mounts and config.environment.backend in SANDBOXES:
                raise RuntimeError(f"{config.environment.backend} has no mounts: use the "
                                   "resources' http transport")
            for name, meta in bound.items():  # e.g. board rules not enforced without mounts
                if meta.get("warning") and getattr(self.host, "emit", None):
                    self.host.emit(self.agent, {"resource": name, "kind": "warning",
                                                "data": {"warning": meta["warning"]}})
        await self._drop_network(self._network)  # the last Trial's: it has ended
        networks = None
        if config.environment.backend not in SANDBOXES:  # a remote VM: no local network
            # the agent's env (inference key, resource tokens): kept out of every exec's argv
            tc.environment.kwargs["secret_names"] = sorted(tc.agent.env)
            # one network per Trial, outside host and VPN routes: docker's default pools run
            # out after about 30 compose projects on a host
            self._network = networks = launch.TrialNetworks(
                self._network_id, engine=config.environment.backend)
            await networks.start()
            self.root.mkdir(parents=True, exist_ok=True)
            tc.environment.extra_docker_compose = [await networks.create(
                tc.trial_name, self.root / f"{tc.trial_name}-network.json")]
        self._creating = True
        try:
            trial = await launch.Trial.create(tc)
            if self._closes:  # closed while this Trial was being created
                await self._drop_network(networks)  # a close may have taken self._network
                raise RuntimeError("agent closed")
            if config.harness == "pi":
                launch.attach_pi_setup(trial, inference)
            if self.bundle is not None:
                launch.attach_tool_setup(trial, self.bundle)
            channel = Channel(emit=self._emit, prompts=self._prompts, home=self.home,
                              model=self.spec.harness_model, trial_seconds=trial_seconds,
                              max_turns=config.limits.agent_max_turns, restore=restore)
            self._endpoint(channel, tc.agent.base_url)
            trial.agent.channel = channel
            self.trial = trial
            self._run = run = asyncio.create_task(trial.run(), name=tc.trial_name)
            run.add_done_callback(lambda task: self._ended(trial, task))
        finally:
            self._creating = False
        ready = asyncio.create_task(channel.ready.wait())
        try:
            await asyncio.wait({ready, run}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            run.cancel()
            raise
        finally:
            ready.cancel()
        if channel.ready.is_set() and config.success_check is not None:
            self._success = None
            self._success_error = None
            self._success_watcher = asyncio.create_task(
                self._watch_success(trial, config.success_check))
        if not channel.ready.is_set():
            why = channel.restore_error or _why(trial, run)
            raise (ResumeError if channel.restore_error else RuntimeError)(
                f"Trial {tc.trial_name} did not start: {why}")

    async def _watch_success(self, trial, check) -> None:
        # Harbor merges compose stderr into stdout; Podman prints a provider banner there.
        # Frame only the trusted command's output instead of parsing that transport noise.
        command = (f"printf %s {shlex.quote(CHECK_START)}; sh -c {shlex.quote(check.command)} ; "
                   f'legion_check_rc=$?; printf %s {shlex.quote(CHECK_END)}; exit "$legion_check_rc"')
        while trial is self.trial and self.alive() and not self._closing:
            try:
                async with asyncio.timeout(5):
                    result = await trial.agent_environment.service_exec(
                        command, service=check.service, timeout_sec=5)
                if result.return_code != 0:
                    raise ValueError(f"check exited {result.return_code}")
                if len(result.stdout or "") > 16384:
                    raise ValueError("check response exceeds 16 KiB")
                output = (result.stdout or "").replace("\r\n", "\n")
                if output.count(CHECK_START) != 1 or output.count(CHECK_END) != 1:
                    raise ValueError("missing or ambiguous success check frame")
                body = json.loads(output.split(CHECK_START, 1)[1].split(CHECK_END, 1)[0])
                if not isinstance(body, dict) or type(body.get("success")) is not bool:
                    raise ValueError("check must return a JSON boolean success")
                self._success_error = None
                if body["success"]:
                    self._success = {"trial_id": trial.config.trial_name,
                                     "observed_at": datetime.now(UTC).isoformat(),
                                     "evidence": body.get("evidence"),
                                     "verification": "pending"}
                    if self._run.done():
                        self._verify_success(_result(self._run))
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry failed probes, never infer success
                # Do not include command stdout or exception text: it may contain task secrets.
                error = f"success check failed ({type(exc).__name__})"
                if error != self._success_error:
                    log.warning("%s: %s", self.agent, error)
                    self._emit(AdapterEvent("error", error=error,
                                            data={"reason": "success_check"}))
                self._success_error = error
            await asyncio.sleep(30)

    async def _drop_network(self, networks: Any) -> None:
        """Remove a Trial's network once its environment has stopped."""
        if networks is None:
            return
        if self._network is networks:
            self._network = None
        try:
            await networks.stop()
        except Exception as e:  # noqa: BLE001 - a leftover network holds a subnet, not the run
            log.warning("%s: Trial network not removed: %s", self.agent, e)

    def _verify_success(self, result: Any) -> None:
        if self._success is not None:
            verifier = getattr(result, "verifier_result", None)
            rewards = getattr(verifier, "rewards", None)
            self._success.update(verification=("passed" if rewards and
                rewards.get("reward") == 1 and not result.exception_info else "failed"),
                rewards=rewards)

    def _ended(self, trial: Any, task: asyncio.Task) -> None:
        """Done callback of a Trial's run(): roll over after trial_hours, report a failure
        (now, or at the next burst). Its results go to the results dir too, when there is
        one (legion.worker.harbor)."""
        self.result = _result(task)
        if (results := results_dir()) is not None:
            name = trial.config.trial_name
            keep_results(trial.config.trials_dir / name, results / self.agent / "trials" / name,
                         TRIAL_RESULTS)
        if trial is self.trial:
            self._verify_success(self.result)
        if trial is not self.trial or not trial.agent.channel.ready.is_set():
            return  # superseded, or never ready: whoever started it reports that
        try:  # unmounted (hf-sandbox): Harbor's download at the end beats LegionPi's last one
            trial.agent.keep_sessions()
        except OSError as e:
            log.warning("%s: sessions of Trial %s not kept: %s", self.agent,
                        trial.config.trial_name, e)
        why = _why(trial, task)
        log.info("%s: Trial %s ended: %s", self.agent, trial.config.trial_name, why)
        if why == "closed" or self._closing:
            return
        if why == "trial_hours":
            self._ensure()  # prompts queued as it ended go to the next Trial
            return
        self._resume_failed = trial.agent.channel.restore_error
        self._fail(f"Trial {trial.config.trial_name} ended: {why}")


def _result(task: asyncio.Task) -> Any:
    return None if task.cancelled() or task.exception() else task.result()


def _why(trial: Any, task: asyncio.Task) -> str:
    """Why a Trial's run() ended: LegionPi's reason, else Harbor's error."""
    if task.cancelled():
        return "cancelled"
    if task.exception() is not None:
        return f"{type(task.exception()).__name__}: {task.exception()}"
    info = task.result().exception_info
    return (getattr(trial.agent, "why", None)
            or (info.exception_message if info is not None else None) or "ended")


def make_adapter(workdir: Path, name: str, spec: TypeSpec, *, host: Any = None) -> HarborAdapter:
    """The worker's factory for harness `harbor` (legion.registry); `host` is the worker's
    HarborHost (its Resources, tool bundle and task download)."""
    return HarborAdapter(workdir, name, spec, host=host)
