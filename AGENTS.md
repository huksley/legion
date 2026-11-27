# legion — notes for agents working in this repo

Read `docs/design.md` first. It is the current design; its "Contracts" section is the
contract and `src/legion/models.py` is that contract in code. Do not change models.py
casually: controller, worker and CLI all import it.

## Layout
- `src/legion/models.py` — shared pydantic contracts (spec, heartbeat, commands, API shapes).
- `src/legion/controller/` — FastAPI app, SQLite store, reconcile loop, HF Jobs client, status page.
- `src/legion/worker/` — worker loop, the `harbor` harness adapter (one Harbor Trial per agent), bucket sync.
- `src/legion/harbor/` — the Harbor task path: spec compile, Trials, Pi and OpenCode (harnesses/), and the `message_board`, `world`, `spawn` resources and tools (builtin/).
- `src/legion/comms/` — the controller's msgboard client and the board rules the board resource applies (docs/comms.md).
- `src/legion/cli.py` — `legion` CLI (typer) over the controller API.
- `experiments/` — Harbor experiments (`mini-swarm`, `escape-room`, `pybot`).
- `tests/` — pytest; no network, no real jobs. Fake the Hub client and Harbor; tiny real runs use tests/tinyrun.py.

## Conventions
- Python 3.12+, pydantic v2, httpx, typer, FastAPI. Keep it small; no new frameworks.
- Keep imports at module scope by default. Use function-local imports only for a concrete reason,
  such as avoiding a circular dependency, keeping optional dependencies optional, or deferring
  expensive initialization. Add a short comment explaining the reason.
- Install with `uv sync --all-extras` (Harbor and controller dependencies are included by default; the dev extra adds tests and linting).
- `ruff check` and `pytest` must pass before pushing.
- Always pass `namespace=` to every `huggingface_hub` jobs call (saves a `whoami` per call).
- Never launch an HF Job from tests. Experiments launch jobs only on `cpu-basic` unless the operator approved otherwise.
- Commit small; push your branch at the end of every turn that produced work.
- YAML (spec files, and YAML snippets in docs and READMEs): write mappings in block style, never
  inline `{a: 1}`; inline lists like `[a, b]` and an empty mapping `{}` are fine.

## Tests
Be conservative about adding tests.
- **Do not add tests by default.** Add one only when the user asks for it, or when a change to **core** legion
  (spec/models, controller, worker, CLI lifecycle, harness adapters, comms, checkpoint/resume) carries a clear
  regression risk that a test covers best. Do not add tests for specific worlds or experiments.
- **Prefer one tiny real run over many isolated tests.** Use the smallest real setup that shows the behaviour:
  a real controller and worker, a real spec, local compute and a fake Harbor environment (tests/tinyrun.py).
- **Use real objects, not stand-ins.** Do not use `MagicMock`, `SimpleNamespace`, `object.__new__` (building an
  instance without its `__init__`) or monkeypatched internals in place of legion's own objects: they only have
  what the test author thought of, so the test keeps passing when the real code changes. Fake only what cannot
  run in the test environment: the Hub, HF Jobs, Slurm, model servers, containers.
- **Test through the public surface** (the CLI, the controller API, the spec, public functions) rather than
  private methods: a test that calls a private method usually has to build the object around it by hand and
  breaks on any refactor. Call a private function directly only when its behaviour cannot be reached
  deterministically otherwise, such as a pure helper.
- **No `slow` tests.** Keep runs tiny. Tests that need an optional dependency or tool (an SDK,
  Linux-only features) are gated with `pytest.importorskip` or `pytest.mark.skipif`, so they skip where they
  cannot run.
- **Scratch tests are fine while developing:** keep them in `scratch/` or `tmp/` and do not commit them.
- **Run what you touched:** `pytest tests/test_<area>.py` (or `-k`) while iterating, and the full suite once
  before pushing.
- **Full suite in parallel:** `pytest -n auto` (pytest-xdist, as CI runs it); tests must not share fixed
  ports, files or global state.
- **Documentation-only changes need no tests:** run `git diff --check` instead.
