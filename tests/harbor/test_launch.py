import asyncio
import json
import signal
from pathlib import Path

import pytest
from harbor.models.task.id import GitTaskId
from harbor.models.trial.result import AgentInfo, TrialResult
from msgboard.client import Client

from legion.harbor import launch
from legion.harbor.config import AgentSettings, Config
from legion.harbor.sources import ResolvedTask

EXAMPLES = Path(__file__).parent / "examples"


@pytest.mark.parametrize("effort", [None, "off", "low"])
def test_reasoning_effort_reaches_native_pi(monkeypatch, tmp_path, effort):
    from harbor.agents.factory import AgentFactory

    spec = Config.load(EXAMPLES / "pi-hf.yaml")
    spec.inference.reasoning_effort = effort
    spec.limits.agent_timeout_seconds = 3480
    monkeypatch.setenv("HF_TOKEN", "test-inference-secret")
    resolved = ResolvedTask(
        GitTaskId(
            git_url=spec.task.repo,
            git_commit_id=spec.task.revision,
            path=Path("tasks") / spec.task.selector,
        ),
        spec.task.revision,
        "test",
    )
    config = launch.build_trial_config(
        spec, resolved, tmp_path, "agent-001", "instructions", spec.api_key()
    )
    assert ("reasoning_effort" in config.agent.kwargs) == (effort is not None)
    assert config.agent.override_timeout_sec == 3480
    agent = AgentFactory.create_agent_from_config(
        config.agent, logs_dir=tmp_path / "logs"
    )
    assert agent.options.reasoning_effort == effort
    assert agent.options.max_turns == spec.limits.agent_max_turns


@pytest.fixture
def prepared(monkeypatch, tmp_path, message_board_provider):
    spec = Config.load(EXAMPLES / "pi-hf.yaml")
    spec.output_dir = tmp_path / "results"
    monkeypatch.setenv("HF_TOKEN", "test-inference-secret")
    source = tmp_path / "source"
    source.mkdir()
    (source / "instruction.md").write_text("Solve the task using a peer's contribution")

    async def resolve(settings):
        return ResolvedTask(
            GitTaskId(
                git_url=settings.repo, git_commit_id="c" * 40, path=Path("tasks/test")
            ),
            "d" * 40,
            settings.dataset,
        )

    async def download(task, directory):
        task.local_path = source
        return source

    monkeypatch.setattr(launch, "resolve_task", resolve)
    monkeypatch.setattr(launch, "download_task", download)

    class Networks:
        def __init__(self, run_id, engine="docker"):
            self.allocations = {}

        async def start(self):
            pass

        async def create(self, name, path):
            path.write_text("{}")
            return path

        async def stop(self):
            pass

    monkeypatch.setattr(launch, "TrialNetworks", Networks)
    return spec


class FakeTrial:
    def __init__(self, config, events, mode="cooperate", expected_count=2):
        self.config, self.events, self.mode = config, events, mode
        self.expected_count = expected_count
        self.hooks = {}
        self.result = TrialResult(
            task_name="test",
            trial_name=config.trial_name,
            config=config,
            trial_uri=(config.trials_dir / config.trial_name).as_uri(),
            task_id=config.task.get_task_id(),
            task_checksum="test-fixture",
            agent_info=AgentInfo(name="fake", version="1"),
        )

    def add_hook(self, event, callback):
        self.hooks[event] = callback

    async def run(self):
        from harbor.trial.hooks import TrialEvent

        assert TrialEvent.AGENT_START in self.hooks
        assert self.config.agent.name == "pi"
        assert not self.config.agent.import_path
        assert "board_bundle" not in self.config.agent.kwargs
        env = self.config.agent.env
        assert env["HF_TOKEN"] == "test-inference-secret"
        assert "test-inference-secret" not in self.config.model_dump_json()
        assert env["MSGBOARD_TOKEN"] not in self.config.model_dump_json()
        self.events.append(self.config.trial_name)
        if self.mode == "wait":
            try:
                await asyncio.sleep(60)
            finally:
                self.events.append("cancelled")
            return self.result
        # Deterministic cooperative test: exchange information through the real board.
        with Client(env["MSGBOARD_URL"], env["MSGBOARD_TOKEN"]) as client:
            general = client.channel_id("general")
            client.post(general, self.config.trial_name)
            for _ in range(200):
                if len(client.read(general)["messages"]) == self.expected_count:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Agents did not run concurrently")
        from harbor.models.verifier.result import VerifierResult

        self.result.verifier_result = VerifierResult(rewards={"reward": 1.0})
        target = self.config.trials_dir / self.config.trial_name
        target.mkdir(parents=True)
        (target / "result.json").write_text(self.result.model_dump_json())
        return self.result


@pytest.mark.parametrize(
    "instructions",
    [None, "Review independently, then share a résumé. Keep {braces} literal.\n"],
)
async def test_concurrent_participants_and_separate_artifacts(
    prepared, monkeypatch, tmp_path, instructions
):
    events = []
    participant_instructions = []
    prepared.team_file = None
    if instructions is not None:
        prepared.team_file = tmp_path / "team.md"
        prepared.team_file.write_text(instructions, encoding="utf-8")

    async def factory(config):
        participant_instructions.append(config.extra_instructions[0])
        if prepared.team_file is not None:
            prepared.team_file.write_text("Changed after startup")
        return FakeTrial(config, events)

    monkeypatch.setattr(launch.Trial, "create", factory)

    directory, code = await launch.run_experiment(prepared)
    data = json.loads((directory / "run.json").read_text())
    assert code == 0
    assert len(set(events)) == 2
    assert all(p["rewards"] == {"reward": 1.0} for p in data["participants"])
    assert len({p["result_path"] for p in data["participants"]}) == 2
    assert "test-inference-secret" not in (directory / "run.json").read_text()
    assert (
        len(
            (directory / "resources/message_board/board.jsonl").read_text().splitlines()
        )
        == 2
    )
    assert data["tools"]["sha256"]
    assert data["configuration"]["tools"] == prepared.tools == ["message_board.*"]
    assert "message_board.*" not in data["tools"]["tools"]
    assert {
        "message_board.read",
        "message_board.post",
        "message_board.create_channel",
    } <= data["tools"]["tools"].keys()
    assert (directory / "inputs/task.md").read_text() == (
        "Solve the task using a peer's contribution"
    )
    assert len(participant_instructions) == 2
    for participant, agent_instructions in zip(
        data["participants"], participant_instructions
    ):
        assert participant["role"] is None
        assert "Your role:" not in agent_instructions
        assert (
            directory / participant["instructions_path"]
        ).read_text() == agent_instructions
        assert participant["name"] in agent_instructions
        assert "message_board.post" in agent_instructions
        assert "Changed after startup" not in agent_instructions
        assert "Share useful findings" not in agent_instructions
        if instructions:
            assert instructions.strip() in agent_instructions
    if instructions is None:
        assert "instructions" not in data["artifacts"]
    else:
        assert (directory / data["artifacts"]["instructions"]).read_text(
            encoding="utf-8"
        ) == instructions


@pytest.mark.parametrize("kind", ["shared", "role"])
async def test_missing_instructions_fail_before_resources_start(
    prepared, tmp_path, kind
):
    if kind == "shared":
        prepared.team_file = tmp_path / "missing.md"
    else:
        prepared.agents = AgentSettings(
            roles={"explorer": {"role_file": tmp_path / "missing.md"}}
        )
    with pytest.raises(FileNotFoundError, match="missing.md"):
        await launch.run_experiment(prepared)
    assert not prepared.output_dir.exists()


async def test_roles_run_concurrently_with_separate_frozen_prompts(
    prepared, monkeypatch, tmp_path
):
    texts = {
        "explorer": "Explore independently. Preserve {literal} and café.\n",
        "reviewer": "Check peer claims against your own task.\n",
    }
    paths = {}
    for role, text in texts.items():
        paths[role] = tmp_path / f"{role}.md"
        paths[role].write_text(text, encoding="utf-8")
    prepared.agents = AgentSettings(
        roles={
            "explorer": {"count": 2, "role_file": paths["explorer"]},
            "reviewer": {"count": 1, "role_file": paths["reviewer"]},
        }
    )
    shared = prepared.team_file.read_text()
    participant_instructions, events, tokens = {}, [], set()

    async def factory(config):
        name = config.trial_name.rsplit("-", 2)[-2:]
        participant_instructions["-".join(name)] = config.extra_instructions[0]
        assert config.agent.model_name == prepared.model
        assert config.agent.kwargs["max_turns"] == prepared.limits.agent_max_turns
        tokens.add(config.agent.env["MSGBOARD_TOKEN"])
        for path in paths.values():
            path.write_text("Changed after startup")
        return FakeTrial(config, events, expected_count=3)

    monkeypatch.setattr(launch.Trial, "create", factory)

    directory, code = await launch.run_experiment(prepared)
    data = json.loads((directory / "run.json").read_text())
    assert code == 0
    assert len(set(events)) == len(tokens) == 3
    assert [p["role"] for p in data["participants"]] == [
        "explorer",
        "explorer",
        "reviewer",
    ]
    assert all(p["rewards"] == {"reward": 1.0} for p in data["participants"])
    assert (
        directory / "inputs/task.md"
    ).read_text() == "Solve the task using a peer's contribution"
    for participant in data["participants"]:
        name, role = participant["name"], participant["role"]
        agent_instructions = participant_instructions[name]
        assert (
            directory / participant["instructions_path"]
        ).read_text() == agent_instructions
        assert f"Your role: {role}." in agent_instructions
        assert (
            agent_instructions.index(shared.strip())
            < agent_instructions.index(texts[role].strip())
            < agent_instructions.index("## Tools")
        )
        assert "Changed after startup" not in agent_instructions
        for other in texts.keys() - {role}:
            assert texts[other].strip() not in agent_instructions
        for peer in data["participants"]:
            if peer["name"] != name:
                assert f"{peer['name']} ({peer['role']})" in agent_instructions
    for role, text in texts.items():
        snapshot = data["artifacts"]["role_instructions"][role]
        assert snapshot == f"inputs/roles/{role}.md"
        assert (directory / snapshot).read_text(encoding="utf-8") == text
    channels = json.loads(
        (directory / "resources/message_board/channels.json").read_text()
    )
    assert {m["external_id"] for m in channels[0]["members"]} >= set(
        participant_instructions
    )


async def test_role_without_extra_text_or_board(prepared, monkeypatch):
    prepared.resources = {}
    prepared.tools = []
    prepared.team_file = None
    prepared.agents = AgentSettings(roles={"explorer": {}})

    class IndependentTrial(FakeTrial):
        async def run(self):
            assert "Your role: explorer." in self.config.extra_instructions[0]
            assert "## Tools" not in self.config.extra_instructions[0]
            assert not self.hooks
            from harbor.models.verifier.result import VerifierResult

            self.result.verifier_result = VerifierResult(rewards={"reward": 0})
            return self.result

    async def factory(config):
        return IndependentTrial(config, [])

    monkeypatch.setattr(launch.Trial, "create", factory)

    directory, code = await launch.run_experiment(prepared)
    data = json.loads((directory / "run.json").read_text())
    assert code == 0 and len(data["participants"]) == 1
    assert "role_instructions" not in data["artifacts"]


async def test_setup_failure_keeps_all_participants_without_fake_rewards(
    prepared, monkeypatch
):
    async def factory(config):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(launch.Trial, "create", factory)

    directory, code = await launch.run_experiment(prepared)
    data = json.loads((directory / "run.json").read_text())
    assert code == 1
    assert data["status"] == "failed"
    assert len(data["participants"]) == 2
    assert all(
        p["status"] == "failed" and p["rewards"] is None for p in data["participants"]
    )


async def test_deadline_cancels_trials_and_exports_partial_board(prepared, monkeypatch):
    from types import SimpleNamespace

    events = []
    deadline = asyncio.timeout(None)
    # Exercise the real timeout, but only expire it once both agents are active.
    launch_asyncio = SimpleNamespace(**vars(asyncio))
    launch_asyncio.timeout = lambda _: deadline
    monkeypatch.setattr(launch, "asyncio", launch_asyncio)

    async def factory(config):
        return FakeTrial(config, events, mode="wait")

    monkeypatch.setattr(launch.Trial, "create", factory)
    task = asyncio.create_task(launch.run_experiment(prepared))
    try:
        async with asyncio.timeout(15):
            while len(events) < 2:
                await asyncio.sleep(0.01)
        deadline.reschedule(asyncio.get_running_loop().time())
        directory, code = await asyncio.wait_for(task, 15)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    data = json.loads((directory / "run.json").read_text())
    assert code == 1 and data["status"] == "timed_out"
    assert events.count("cancelled") == 2
    assert all(
        p["rewards"] is None and p["status"] == "timed_out"
        for p in data["participants"]
    )
    assert (directory / "resources/message_board/board.jsonl").exists()


@pytest.mark.parametrize("stop", ["interrupt", "sigterm", "board_exit"])
async def test_resource_failure_cancels_peers_and_retains_state(
    prepared, monkeypatch, message_board_provider, stop
):
    events = []
    runtimes = []

    original_start = message_board_provider.start

    async def tracked_start(self):
        runtimes.append(self)
        await original_start(self)

    monkeypatch.setattr(message_board_provider, "start", tracked_start)

    async def factory(config):
        return FakeTrial(config, events, mode="wait")

    previous_handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(launch.Trial, "create", factory)
    task = asyncio.create_task(launch.run_experiment(prepared))
    try:
        async with asyncio.timeout(15):
            while len(events) < 2:
                await asyncio.sleep(0.01)
        if stop == "interrupt":
            task.cancel()
        elif stop == "sigterm":
            assert previous_handler == signal.SIG_DFL
            assert signal.getsignal(signal.SIGTERM) != signal.SIG_DFL
            signal.raise_signal(signal.SIGTERM)
        else:
            runtimes[0].process.kill()
        directory, code = await asyncio.wait_for(task, 15)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    data = json.loads((directory / "run.json").read_text())
    interrupted = stop in ("interrupt", "sigterm")
    assert code == (130 if interrupted else 1)
    assert data["exit_code"] == code
    assert data["status"] == ("interrupted" if interrupted else "failed")
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert events.count("cancelled") == 2
    assert all(p["status"] == "cancelled" for p in data["participants"])
    assert runtimes[0].process.returncode is not None
    assert runtimes[0].data_dir.is_dir()
    if interrupted:
        assert (directory / "resources/message_board/board.jsonl").exists()
    else:
        assert "exited unexpectedly" in data["error"]
        assert data["resources"]["message_board"]["export_error"]


@pytest.mark.parametrize("handler_kind", ["default", "custom", "asyncio"])
async def test_preflight_failure_preserves_sigterm_handler(monkeypatch, handler_kind):
    spec = Config.load(EXAMPLES / "pi-hf.yaml")
    # HF_TOKEN would fall back to a logged-in token, so use a key with no fallback.
    spec.inference.api_key_env = "LEGION_TEST_MISSING_KEY"
    monkeypatch.delenv("LEGION_TEST_MISSING_KEY", raising=False)
    previous_handler = signal.getsignal(signal.SIGTERM)
    loop = asyncio.get_running_loop()
    called = asyncio.Event()
    if handler_kind == "custom":
        signal.signal(signal.SIGTERM, lambda *_: called.set())
    elif handler_kind == "asyncio":
        loop.add_signal_handler(signal.SIGTERM, called.set)
    expected_handler = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(ValueError, match="Set LEGION_TEST_MISSING_KEY"):
            await launch.run_experiment(spec)
        assert signal.getsignal(signal.SIGTERM) == expected_handler
        if handler_kind != "default":
            signal.raise_signal(signal.SIGTERM)
            await asyncio.wait_for(called.wait(), timeout=1)
    finally:
        if handler_kind == "asyncio":
            loop.remove_signal_handler(signal.SIGTERM)
        signal.signal(signal.SIGTERM, previous_handler)


async def test_export_failure_returns_nonzero(
    prepared, monkeypatch, message_board_provider
):
    def fail_export(self, directory):
        raise OSError("disk full")

    monkeypatch.setattr(message_board_provider, "_export", fail_export)

    async def factory(config):
        return FakeTrial(config, [])

    monkeypatch.setattr(launch.Trial, "create", factory)

    directory, code = await launch.run_experiment(prepared)
    data = json.loads((directory / "run.json").read_text())
    assert code == 1 and data["status"] == "failed"
    assert "disk full" in data["resources"]["message_board"]["export_error"]
    assert all(p["rewards"] == {"reward": 1.0} for p in data["participants"])


async def test_multistep_outcomes_remain_visible(prepared, monkeypatch):
    from harbor.models.trial.result import ExceptionInfo, StepResult
    from harbor.models.verifier.result import VerifierResult

    class StepTrial(FakeTrial):
        async def run(self):
            self.result.step_results = [
                StepResult(
                    step_name="first",
                    verifier_result=VerifierResult(rewards={"reward": 1}),
                ),
                StepResult(
                    step_name="second",
                    exception_info=ExceptionInfo.from_exception(
                        RuntimeError("step failed")
                    ),
                ),
            ]
            return self.result

    async def factory(config):
        return StepTrial(config, [])

    monkeypatch.setattr(launch.Trial, "create", factory)

    directory, code = await launch.run_experiment(prepared)
    data = json.loads((directory / "run.json").read_text())
    assert code == 1
    assert data["status"] == "failed"
    for participant in data["participants"]:
        assert participant["status"] == "failed"
        assert participant["steps"][0]["rewards"] == {"reward": 1}
        assert "step failed" in participant["steps"][1]["error"]


async def test_partial_network_setup_waits_for_trial_cleanup(prepared, monkeypatch):
    events = []

    class FailingNetworks:
        allocations = {}

        def __init__(self, run_id, engine="docker"):
            pass

        async def start(self):
            pass

        async def create(self, name, path):
            if name == "agent-002":
                async with asyncio.timeout(5):
                    while not events:
                        await asyncio.sleep(0.01)
                raise RuntimeError("network creation failed")
            path.write_text("{}")
            return path

        async def stop(self):
            assert events[-1] == "cancelled"
            events.append("networks removed")

    monkeypatch.setattr(launch, "TrialNetworks", FailingNetworks)

    async def factory(config):
        assert config.agent.env["MSGBOARD_SOCKET"] == "/run/legion-board.sock"
        assert config.environment.mounts[0]["read_only"]
        return FakeTrial(config, events, mode="wait")

    monkeypatch.setattr(launch.Trial, "create", factory)

    directory, code = await launch.run_experiment(prepared)
    data = json.loads((directory / "run.json").read_text())
    assert code == 1 and data["status"] == "failed"
    assert "network creation failed" in data["error"]
    assert events[-1] == "networks removed"
    assert all(p["status"] == "cancelled" for p in data["participants"])


@pytest.mark.parametrize("reasoning", [None, "low"])
async def test_no_resources_or_tools_runs_independently(
    prepared, monkeypatch, reasoning
):
    from harbor.trial.hooks import TrialEvent

    prepared.resources = {}
    prepared.tools = []
    prepared.team_file = None
    prepared.inference.reasoning_effort = reasoning

    class IndependentTrial(FakeTrial):
        async def run(self):
            assert set(self.hooks) == ({TrialEvent.AGENT_START} if reasoning else set())
            assert set(self.config.agent.env) == {"HF_TOKEN", "LEGION_TOOLS_MANIFEST"}
            assert not self.config.environment.mounts
            from harbor.models.verifier.result import VerifierResult

            self.result.verifier_result = VerifierResult(rewards={"reward": 0})
            return self.result

    async def factory(config):
        return IndependentTrial(config, [])

    monkeypatch.setattr(launch.Trial, "create", factory)

    directory, code = await launch.run_experiment(prepared)
    data = json.loads((directory / "run.json").read_text())
    assert code == 0
    assert all(p["status"] == "completed" for p in data["participants"])
    assert data["resources"] == {}
    assert "msgboard" not in data["versions"]
    assert not (directory / "state/resources/message_board").exists()
    assert all(p["resources"] == {} for p in data["participants"])


async def test_board_tools_require_configured_resource(prepared):
    prepared.resources = {}
    with pytest.raises(ValueError, match="require resources: message_board"):
        await launch.run_experiment(prepared)


@pytest.mark.parametrize("backend,delete", [("docker", True), ("podman", False)])
def test_podman_retains_images_shared_by_trials(prepared, tmp_path, backend, delete):
    prepared.environment.backend = backend
    task = ResolvedTask(
        GitTaskId(
            git_url=prepared.task.repo, git_commit_id="c" * 40, path=Path("tasks/test")
        ),
        "c" * 40,
        "test",
    )
    trial = launch.build_trial_config(
        prepared, task, tmp_path, "agent-001", "", "test-inference-secret"
    )
    assert trial.environment.delete is delete


def test_hf_sandbox_outlives_the_run(tmp_path):
    spec = Config.load(EXAMPLES / "mini-swarm.yaml")
    task = ResolvedTask(
        GitTaskId(
            git_url=spec.task.repo, git_commit_id="c" * 40, path=Path("tasks/test")
        ),
        "c" * 40,
        "test",
    )
    trial = launch.build_trial_config(
        spec, task, tmp_path, "agent-001", "", "test-inference-secret"
    )
    # Pi's run is one command, which the sandbox's idle timeout doesn't count.
    assert trial.environment.type == "hf-sandbox" and trial.environment.delete
    assert trial.environment.kwargs["flavor"] == "cpu-basic"
    assert trial.environment.kwargs["job_timeout"] > spec.limits.run_timeout_seconds


@pytest.mark.parametrize("failure", [None, "pi", "missing_reward"])
async def test_participant_outcomes_determine_run_status(
    prepared, monkeypatch, failure
):
    from harbor.models.verifier.result import VerifierResult

    class OutcomeTrial(FakeTrial):
        async def run(self):
            self.result.verifier_result = VerifierResult(rewards={"reward": 0})
            if self.config.trial_name.endswith("agent-001"):
                if failure == "missing_reward":
                    self.result.verifier_result = None
                if failure == "pi":
                    folder = self.config.trials_dir / self.config.trial_name / "agent"
                    folder.mkdir(parents=True)
                    (folder / "pi.txt").write_text(
                        json.dumps(
                            {
                                "type": "message_end",
                                "message": {
                                    "role": "assistant",
                                    "stopReason": "error",
                                    "errorMessage": "Connection error. test-inference-secret",
                                },
                            }
                        )
                    )
            return self.result

    async def factory(config):
        return OutcomeTrial(config, [])

    monkeypatch.setattr(launch.Trial, "create", factory)
    directory, code = await launch.run_experiment(prepared)
    data = json.loads((directory / "run.json").read_text())
    assert code == int(failure is not None)
    assert data["status"] == ("failed" if failure else "completed")
    assert data["participants"][1]["status"] == "completed"
    assert data["participants"][1]["rewards"] == {"reward": 0}
    assert "test-inference-secret" not in (directory / "run.json").read_text()


@pytest.mark.parametrize("backend,platform,expected", [
    ("docker", "darwin", "http://host.docker.internal:8765"),  # Docker Desktop
    ("docker", "linux", "http://127.0.0.1:8765"),  # no such name by default
    ("podman", "linux", "http://host.containers.internal:8765"),
])
def test_this_machine_as_agent_containers_reach_it(monkeypatch, backend, platform, expected):
    monkeypatch.setattr(launch.sys, "platform", platform)
    assert launch.agent_url("http://127.0.0.1:8765", backend) == expected
    assert launch.agent_url("https://example.com/v1", backend) == "https://example.com/v1"
