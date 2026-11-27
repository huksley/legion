# Local mode: `legion launch`

One spec from a laptop to 1k agents. `compute: local` runs the swarm on your machine:
`legion launch spec.yaml` starts the controller on 127.0.0.1 and launches each worker as a
subprocess, both in the background (detached; output in `.legion/launch.log`, the run directory
under `.legion/<swarm>-<UTC time>/`), then returns. `legion status`, `legion prompt`,
`legion update` and `legion stop` find the run through `.legion/current` in the same
directory. The controller writes a run bundle when it ends. The
controller is the same code as on HF Jobs (reconcile loop, placement, heartbeats, admission,
prompts, events); only the backend that starts workers differs. `compute: local` is
the default for Harbor specs; see
[Switching to HF Jobs](#switching-to-hf-jobs) for remote execution.

No HF token, bucket or network is needed when the model endpoint is local and no
`bucket_id` or remote resource is configured. A configured bucket is resolved and created
at launch on every backend, including local compute; it requires HF bucket write access.

## Quickstart

Install (Python 3.12+; Harbor comes with legion):

```sh
git clone https://github.com/huggingface/legion && cd legion
uv sync
```

Write a Harbor spec (`compute: local` is the default). This one runs 2 agents on one
terminal-bench task in docker, with the HF router as the model server (tests/harbor/examples/
has more; experiments/mini-swarm/spec.yaml adds a Space board and a loop, on hf-sandbox):

```yaml
name: hello
model: moonshotai/Kimi-K2.5             # on the HF router (the default inference)
inference:
  api_key_env: HF_INFERENCE_TOKEN
task:
  repo: https://huggingface.co/datasets/harborframework/terminal-bench-2.1
  dataset: terminal-bench-2.1@2.1.0
  selector: regex-log
  revision: 3e235dff6880252a587fa479c09bcb1e16edf2eb
agents:
  count: 2
environment:
  backend: docker
limits:
  run_timeout_seconds: 900
  agent_max_turns: 30
```

`pricing: [in, out, cached]` (USD per 1M tokens) gives the run a cost; Harbor reports none for
router models ([docs/open-models.md](open-models.md)).
This example omits `namespace`: router requests use `HF_INFERENCE_TOKEN`'s default personal
billing account with no bill-to header. Enable Inference Providers calls on that key.
Private Hub tasks additionally need a separate `HF_TOKEN` with read access; the public task
above needs no Jobs, Space or bucket permissions. See [billing and storage](billing.md).

Run it:

```sh
export HF_INFERENCE_TOKEN=...
uv run legion launch spec.yaml
uv run legion prompt --swarm hello --type agent "Start on the task."
uv run legion status hello
uv run legion stop hello --release      # drain the workers; `legion launch hello` resumes
uv run legion stop hello --delete --yes # end the controller, write the bundle
```

For tests and dry runs, `python -m legion.controller.localrun spec.yaml --prompt "..." --until idle`
runs the same thing in the foreground and exits when done.
Its launch path also prepares `bucket_id` before starting the controller. `--env` files
supply credentials as well as spec variables. `--no-apply` is the background serving mode
used internally by `legion launch`, which supplies its already prepared checkpoint URL.

- `legion prompt --swarm <swarm> --type <type> "text"` sends a prompt to every agent of a type
  (optional; without it the agents wait for prompts).
- `legion stop <swarm>` holds the swarm; `legion stop <swarm> --release` drains the workers and keeps
  the controller (`legion launch <swarm>` resumes); `legion stop <swarm> --delete` ends the
  controller and writes the bundle. A run also ends at the spec's `limits.budget_usd`, and a run
  whose every type has a `loop:` ends by itself once every loop is over and every agent is idle (or stuck after a turn timeout with no resume left). The test harness (`python -m legion.controller.localrun`) additionally has
  `--until idle|10m` and `--grace S`.
- The spec's `tools_dir` and `resources_dir` are packed, uploaded and named in the spec by
  their hash before the run starts, as `legion launch` does.
- After editing the experiment folder (spec, prompt files, tools), run `legion update <folder>` to
  apply the change to the running swarm; a broken save is shown and not applied
  ([live.md](live.md)).

## Reaching the run

While the run is up, `.legion/current` (mode 0600) holds the controller URL
and the generated token. Every other command uses it when `LEGION_URL` is not set, so from the
same directory:

```sh
uv run legion status
uv run legion status hello --type agent
uv run legion prompt amber-otter "Report your progress."
uv run legion prompt --swarm hello --type agent --state idle "Next round."
uv run legion status amber-otter --events --follow
```

From elsewhere: `LEGION_CURRENT=/path/to/.legion/current legion status`, or set `LEGION_URL` and
`LEGION_TOKEN` from that file.

**At the end** the run halts the reconcile loop (nothing relaunches) and drains every worker:
idle agents stop at once, in-flight turns get 10 s (`--grace` of the foreground harness). Then each remaining worker gets
SIGTERM: it interrupts its turns (a `turn_interrupted` event) and its harness kills the tool
processes it started. Up to 5 s later (90 s for a `harbor` worker, which first cancels its Trials
and waits for their containers to stop), anything still alive is killed, including tool
processes in their own session. Finally the run waits for the last events and writes the bundle. The
budget check, the view and the bundle use the spec as currently applied, so a
`legion update` during the run counts: `limits.budget_usd`, for example.

```
.legion/hello-20261021-161443/
  run.json             spec (env values with TOKEN/KEY/SECRET/... in the name redacted), stop reason,
                       duration, totals (agents, turns, events, cost, tokens), per agent and per worker,
                       `loops` (per type: state, stop reason, rounds, final round) and `budget`
                       (budget_usd, final reserve, cost, which budget stop fired)
  events.jsonl         every agent's events and the swarm-level ones (loop_*, final_done,
                       budget_stop; agent ""), merged by time (pi session format v3)
  events/<agent>.jsonl one file per agent
  agents/<agent>/      agent workdirs
  logs/                controller.log, <worker>.log      controller/  the controller's SQLite state
```

## Switching to HF Jobs

Change `compute: local` to `compute: hf-jobs` and set `bucket_id: legion-runs` (created private
if missing, under `HF_TOKEN`'s personal account). The omitted `namespace` keeps infrastructure
in that account and router calls on the inference key's default billing account. See
[billing and storage](billing.md), [deploy a controller](deploy-job.md), and run
`legion launch spec.yaml`. `legion launch` of a
`hf-jobs` spec goes to the jobs controller, and a jobs controller refuses a `local` spec (HTTP 400), so a spec never runs
on a backend it did not ask for.

A long-lived local controller can also be started by hand:
`LEGION_TOKEN=... LEGION_COMPUTE=local uv run legion-controller`, then `legion launch` a `local` spec.

## What differs locally

- Agent names follow the spec's `naming:` as on HF Jobs: a Harbor spec's random words by default,
  consecutive `agent-001`, ... with `naming: consecutive-id`, `<type>-NNNN` for a swarm spec
  (docs/spawn.md "Names"); the run bundle's
  `events/<name>.jsonl` and `agents/<name>/` use the same names.

Same code as on jobs: the spec, the API and CLI, the reconcile loop and placement
(`agents_per_worker`, `workers:`), heartbeats and agent states, prompt admission
(`max_working`), bulk and children prompts, spawn and its limits, events, cost from the pinned
prices, worker lifetime and planned restarts, crash backoff and `crashlooping`. The local tests
exercise placement over two workers, prompts, events, the CLI and stop; spawn is not yet run
locally.

Different:

1. **Workers are processes.** One `python -m legion.worker.main` subprocess per worker slot, with
   the job env contract (`LEGION_URL`, `LEGION_TOKEN`, `WORKER_ID`, `WORKER_INCARNATION`, `SWARM`,
   `TYPE`, `DEADLINE`, `DRAIN_WINDOW`, the type's `env`, its secrets). Job ids are `local-NNNN`.
   Exit 0 is `COMPLETED`, any other exit `ERROR` ("exit N"), a process past its lifetime is
   killed and reported as `Job timeout` (a planned restart, as on HF Jobs).
2. **Local worker placement:** `image`, `flavor`, the network group, the wheel and bootstrap
   do not select a local worker process's hardware. `namespace` still controls billing for
   HF sandboxes, model Jobs and router requests. A configured `bucket_id` is prepared at
   launch and can hold the controller checkpoint. Local agents' workdirs stay in
   `<run>/agents/` (shared by the run's workers, so a restarted worker slot finds them on disk).
3. **Secrets come from your shell.** The names a spec lists (`secrets:`, `api_key_env`) are read from the environment of `legion launch` and passed to the workers as
   env, as HF Jobs passes job secrets. Workers inherit your environment minus your own legion
   connection settings (`LEGION_*`, except `LEGION_PROMPTS_DIR`); the harness still gets only the allow-listed variables plus what the
   worker passes.
4. **Faster loop.** Reconcile every 1 s (jobs: 5 s; `LEGION_RECONCILE_INTERVAL` overrides both)
   and worker heartbeats every 5 s (jobs: 30 s; a type's `env: {HEARTBEAT_INTERVAL: ...}`
   overrides it). The foreground run checks its stop conditions every 2 s (`LEGION_LOCAL_REFRESH`).
5. **The controller runs next to you.** 127.0.0.1 on a free port, a new random token per run, no
   status port, no jobs proxy auth. The controller state syncs to `LEGION_STATE_URL`, by
   default `hf://buckets/<bucket_id>/controller/<swarm>` when the spec has a bucket.
   A configured bucket requires `HF_TOKEN` (or a saved login) with create/write access;
   missing credentials or access fail launch. `LEGION_STATE_URL=` disables checkpointing
   but still prepares any configured bucket. Omit `bucket_id` for a Hub-free local run.
   Each `legion launch` is a new run
   directory: a run does not resume a previous one.
6. **Workers die with the controller.** A worker exits when the controller process is gone
   (`LEGION_LOCAL_PARENT_PID`), so a killed run leaves no orphan workers. `legion stop --delete` makes the
   controller drain and stop the workers itself. A local worker handles SIGTERM by
   interrupting its turns (a job worker keeps the default behaviour).
7. **Logs are files.** `<run>/logs/<worker>.log` instead of `hf jobs logs`; `legion status`
   says so.
8. **Task environments still apply.** Local workers run Harbor Trials using the spec's
   `environment.backend`: Docker, Podman, an HF Sandbox, or E2B. Local worker placement
   does not make an HF Sandbox run locally or remove its billing namespace.

A worker process that exits is noticed on the next reconcile pass and relaunched after
`compute.crash_backoff` (2 s locally; docs/live.md "Worker liveness").

Known gaps: `limits.budget_usd` is the only spec limit that stops a run (`alive` and
`spawned_total` limit spawning, as on jobs); tested on Linux; on macOS with Docker Desktop (container networks avoid the Mac's routes, and `localhost` URLs reach agents as `host.docker.internal`); Windows is not supported (process groups).
