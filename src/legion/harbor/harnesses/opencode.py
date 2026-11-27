"""LegionOpenCode: Harbor's OpenCode as a legion turn server (legion.harbor.agent.TurnServer).

Each legion prompt is one `opencode run --format=json` in the Trial's container; turns after
the first add `--continue`, which takes up the newest session of the agent's working
directory. OpenCode keeps its sessions under XDG_DATA_HOME, which Harbor points at the agent's
logs dir (`opencode/xdg-data`): mounted on docker and podman, downloaded on hf-sandbox, and
copied to the channel's home after each turn like Pi's session files.

The model is reached through one generated OpenAI-compatible provider, `legion`
(PROVIDER): the run's `inference` block gives its base URL and key, and the channel its
context window (`limit.context`, so OpenCode compacts before the server refuses a request).

Not covered: `limits.agent_max_turns` (OpenCode's `run` has no step limit flag; the type's
`turn_timeout` bounds a turn) and `inference.reasoning_effort`.
"""

import copy
import json
import shlex
from pathlib import Path

from harbor.agents.installed.opencode import OpenCode

from legion.models import TokenUsage

from ..agent import TurnServer, kill_command

PROVIDER = "legion"  # the provider id the generated OpenCode config registers the model under
OPENCODE_VERSION = "1.19.6"  # the opencode-ai npm release Harbor 0.25's flags were checked with
OUTPUT_LIMIT = 32768  # `limit.output` in the model entry (OpenCode requires one with `context`)
# Top-level directories of OpenCode's data dir that hold no session state (downloaded tools,
# logs): not copied to the channel's home.
SKIP = {"bin", "log"}
TOOL_OUTPUT = 4000  # characters of a tool's output kept in its toolResult message


def opencode_tokens(tokens):
    """A step_finish `tokens` object as TokenUsage: OpenCode counts reasoning apart from
    `output`; legion bills it as output (`reasoning` is inside `output`, as for Pi)."""
    tokens = tokens if isinstance(tokens, dict) else {}
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    reasoning = int(tokens.get("reasoning") or 0)
    return TokenUsage(input=int(tokens.get("input") or 0),
                      output=int(tokens.get("output") or 0) + reasoning,
                      cache_read=int(cache.get("read") or 0),
                      cache_write=int(cache.get("write") or 0),
                      reasoning=reasoning)


def _events(path):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def read_turn(path, model=None):
    """(messages, usage) of one turn's opencode.txt, the messages in Pi's shape (what legion's
    events, status and loop summaries read): per OpenCode step one assistant message with
    `text`, `thinking` and `toolCall` blocks and the step's usage, then a `toolResult` per
    finished tool call."""
    messages, usage = [], TokenUsage()
    parts = None
    for event in _events(path):
        kind = event.get("type")
        if kind == "step_start":
            parts = []
        elif kind in ("text", "reasoning", "tool_use") and parts is not None:
            part = event.get("part")
            if isinstance(part, dict):
                parts.append(part)
        elif kind == "step_finish" and parts is not None:
            finish = event.get("part") if isinstance(event.get("part"), dict) else {}
            step = opencode_tokens(finish.get("tokens"))
            usage += step
            messages.extend(_step_messages(parts, step, model))
            parts = None
    if parts:  # a step cut short (an error, a kill): its parts, without usage
        messages.extend(_step_messages(parts, None, model))
    return messages, usage


def _step_messages(parts, step, model):
    content, results = [], []
    for part in parts:
        kind = part.get("type")
        if kind == "text" and part.get("text"):
            content.append({"type": "text", "text": part["text"]})
        elif kind == "reasoning" and part.get("text"):
            content.append({"type": "thinking", "thinking": part["text"]})
        elif kind == "tool":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            call_id = part.get("callID") or part.get("id") or ""
            arguments = state.get("input")
            content.append({"type": "toolCall", "id": call_id, "name": part.get("tool") or "?",
                            "arguments": arguments if isinstance(arguments, dict) else {}})
            if state.get("status") in ("completed", "error"):
                output = state.get("output") if state.get("status") == "completed" else (
                    state.get("error"))
                results.append({"role": "toolResult", "toolCallId": call_id,
                                "toolName": part.get("tool") or "?",
                                "content": [{"type": "text",
                                             "text": str(output or "")[:TOOL_OUTPUT]}],
                                "isError": state.get("status") == "error"})
    message = {"role": "assistant", "content": content}
    if model:
        message["model"] = model
    if step is not None:
        message["usage"] = {"input": step.input, "output": step.output,
                            "cacheRead": step.cache_read, "cacheWrite": step.cache_write,
                            "reasoning": step.reasoning}
    return [message, *results] if content or step is not None else results


def terminal_opencode_error(path):
    """The first `error` event of one turn's opencode.txt, as text, or None."""
    for event in _events(path):
        if event.get("type") != "error":
            continue
        error = event.get("error")
        if isinstance(error, dict):
            data = error.get("data")
            message = data.get("message") if isinstance(data, dict) else None
            return str(message or error.get("name") or error)[:500]
        return str(error)[:500]
    return None


class LegionOpenCode(TurnServer, OpenCode):
    _OUTPUT_FILENAME = "opencode.txt"
    _SESSIONS_DIRECTORY = Path("opencode/xdg-data")
    # `opencode --model=<id> run ...`: the CLI and the shell that started it; the bracket keeps
    # the kill script itself from matching
    _KILL = kill_command("*[o]'pencode --model='*' run '*")
    _LABEL = "opencode"

    def _read_turn(self, output):
        return read_turn(output, getattr(getattr(self, "channel", None), "model", None))

    def _turn_error(self, output, max_turns):
        return terminal_opencode_error(output)

    def _session_files(self, root):
        """Every file of OpenCode's data dir (`<root>/opencode/...`: its database, session
        storage and snapshots) except its downloaded tools and logs."""
        if not root.is_dir():
            return []
        kept = []
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            parts = path.relative_to(root).parts
            if parts[0] == "opencode" and len(parts) > 2 and parts[1] in SKIP:
                continue
            kept.append(path)
        return kept

    def _build_register_config_command(self):
        """Harbor's opencode.json with the channel's endpoint in the `legion` provider: its
        base URL and headers, and the model's context window; no autoupdate or sharing."""
        ch = getattr(self, "channel", None)
        base = getattr(self, "_legion_base_config", None)
        if base is None:
            base = self._legion_base_config = copy.deepcopy(self._opencode_config)
        model_id = self._execution_model_name().partition("/")[2]
        entry = {"tool_call": True}
        options = {}
        if ch is not None:
            if ch.context_window:
                entry["limit"] = {"context": ch.context_window,
                                  "output": min(OUTPUT_LIMIT, ch.context_window // 4)}
            if ch.base_url:
                options["baseURL"] = ch.base_url
            if ch.headers:
                options["headers"] = dict(ch.headers)
        # no undo snapshots: a task's working directory is often `/`, which OpenCode would
        # git-add in full at every step; legion keeps the session, not file history
        overlay = {"autoupdate": False, "share": "disabled", "snapshot": False,
                   "provider": {PROVIDER: {"name": "legion", "models": {model_id: entry}}}}
        self._opencode_config = self._deep_merge(copy.deepcopy(base), overlay)
        command = super()._build_register_config_command()
        if command is None or not options:
            return command
        # Harbor sets the connection's base URL last; the channel's (the live spec's, as the
        # container reaches it) wins, as in LegionPi's models.json
        words = shlex.split(command)
        config = json.loads(words[words.index("echo") + 1])
        config = self._deep_merge(config, {"provider": {PROVIDER: {"options": options}}})
        return (f"mkdir -p ~/.config/opencode && echo {shlex.quote(json.dumps(config, indent=2))}"
                " > ~/.config/opencode/opencode.json")
