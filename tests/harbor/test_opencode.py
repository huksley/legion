"""The OpenCode harness (legion.harbor.harnesses.opencode): the spec reaches the worker and its
Trial as OpenCode on the run's endpoint, and a turn's `opencode run --format=json` output
becomes legion messages and usage."""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from harbor.agents.factory import AgentFactory
from harbor.models.task.id import GitTaskId

from legion.harbor import launch
from legion.harbor.agent import Channel
from legion.harbor.harnesses import opencode
from legion.harbor.sources import ResolvedTask
from legion.harbor.spec import Spec, compile_spec
from legion.models import TokenUsage
from legion.worker.adapters.harbor import AGENTS

URL = "https://inference.example.com/v1"
TASK = {"registry_commit": "d" * 40, "dataset": "tasks@1", "task": {"git_url": "x", "path": "t"}}
SPEC = f"""
name: opencode-team
harness: opencode
model: deepseek-4-1-flash-1017
inference:
  urls: [{URL}]
  api_key_env: EXAMPLE_API_KEY
  context_window: 65536
task:
  repo: https://huggingface.co/datasets/org/tasks
  selector: t
agents:
  count: 3
environment:
  backend: docker
"""


def load(tmp_path, harness="opencode"):
    (tmp_path / "spec.yaml").write_text(SPEC.replace("harness: opencode", f"harness: {harness}"))
    return Spec.load(tmp_path / "spec.yaml", env={})


def trial_config(tmp_path, spec):
    task = ResolvedTask(GitTaskId(git_url=spec.task.repo, git_commit_id="c" * 40,
                                  path=Path("tasks/t")), "c" * 40, "test")
    return launch.build_trial_config(spec, task, tmp_path, "agent-001", "instructions",
                                     "secret-key")


@pytest.mark.parametrize("harness", ["opencode", "pi"])
def test_the_spec_reaches_the_worker_and_the_trial(tmp_path, harness):
    spec = load(tmp_path, harness)
    (agent,) = compile_spec(spec, TASK).types.values()
    tc = trial_config(tmp_path, spec)
    if harness == "pi":  # compiles and runs as before
        assert "harness" not in agent.harbor.config
        assert (tc.agent.name, tc.agent.model_name) == ("pi", "deepseek-4-1-flash-1017")
        return
    assert AGENTS[agent.harbor.config["harness"]].endswith(":LegionOpenCode")
    assert (tc.agent.name, tc.agent.model_name) == ("opencode", "legion/deepseek-4-1-flash-1017")
    assert tc.agent.base_url == URL and tc.agent.env["EXAMPLE_API_KEY"] == "secret-key"
    assert tc.agent.kwargs == {"version": opencode.OPENCODE_VERSION}
    spec.check_model_connection("secret-key")


def test_opencode_json_registers_the_model_on_the_channels_endpoint(tmp_path):
    """The config `opencode run` reads: the `legion` provider at the channel's base URL (the
    live spec's) with the key's variable, the model's context window, no autoupdate,
    sharing or snapshots."""
    tc = trial_config(tmp_path, load(tmp_path))
    tc.agent.name, tc.agent.import_path = None, AGENTS["opencode"]
    agent = AgentFactory.create_agent_from_config(tc.agent, logs_dir=tmp_path / "logs")
    agent.channel = Channel(emit=lambda event: None, prompts=asyncio.Queue(),
                            home=tmp_path / "home", model="m", trial_seconds=60,
                            base_url="http://10.0.0.2:8000/v1", context_window=65536)
    command = agent._build_register_config_command()  # a pure builder; no container needed
    quoted = command.split("&& echo ", 1)[1].rsplit(" > ", 1)[0]
    written = json.loads(subprocess.run(["sh", "-c", f"printf %s {quoted}"],
                                        capture_output=True, text=True, check=True).stdout)
    provider = written["provider"]["legion"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://10.0.0.2:8000/v1"
    key = provider["options"]["apiKey"]
    assert key.startswith("{env:") and agent.model_connection.env[key[5:-1]] == "secret-key"
    assert provider["models"]["deepseek-4-1-flash-1017"] == {
        "tool_call": True, "limit": {"context": 65536, "output": 16384}}
    assert (written["autoupdate"], written["share"], written["snapshot"]) == (
        False, "disabled", False)


TURN = [
    {"type": "step_start", "part": {"type": "step-start"}},
    {"type": "reasoning", "part": {"type": "reasoning", "text": "Post first."}},
    {"type": "tool_use", "part": {"type": "tool", "tool": "bash", "callID": "c1", "state": {
        "status": "completed", "input": {"command": "legion-tool message_board.post"},
        "output": "{\"ok\": true}"}}},
    {"type": "step_finish", "part": {"type": "step-finish", "tokens": {
        "input": 1000, "output": 50, "reasoning": 20, "cache": {"read": 300, "write": 0}}}},
    {"type": "step_start", "part": {"type": "step-start"}},
    {"type": "text", "part": {"type": "text", "text": "Posted my hello."}},
    {"type": "step_finish", "part": {"type": "step-finish", "tokens": {
        "input": 200, "output": 10, "reasoning": 0, "cache": {"read": 1300, "write": 5}}}},
]


def test_a_turn_becomes_pi_shaped_messages_and_usage(tmp_path):
    out = tmp_path / "opencode.txt"
    out.write_text("".join(json.dumps(e) + "\n" for e in TURN) + "not json\n")
    messages, usage = opencode.read_turn(out, "m")
    # reasoning is billed as output, as Pi counts it
    assert usage == TokenUsage(input=1200, output=80, cache_read=1600, cache_write=5,
                               reasoning=20)
    first, result, last = messages
    assert [b["type"] for b in first["content"]] == ["thinking", "toolCall"]
    assert first["content"][1]["name"] == "bash" and first["usage"]["output"] == 70
    assert (result["role"], result["toolName"], result["isError"]) == ("toolResult", "bash", False)
    assert last["content"] == [{"type": "text", "text": "Posted my hello."}]
    assert opencode.terminal_opencode_error(out) is None

    out.write_text(json.dumps(TURN[0]) + "\n" + json.dumps({"type": "error", "error": {
        "name": "APIError", "data": {"message": "429 Too Many Requests"}}}) + "\n")
    assert opencode.terminal_opencode_error(out) == "429 Too Many Requests"
    assert opencode.read_turn(out) == ([], TokenUsage())
