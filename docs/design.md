# legion: design

How legion works today. The contract in code is `src/legion/models.py`; the user guides are the
other files in docs/.

## What legion is

legion hosts agent swarms and keeps them running: it places agents on workers, prompts them,
syncs their sessions and workdirs, restarts what fails and reports health and cost. Each agent
is one long-lived Harbor Trial in which Harbor's Pi (or OpenCode, with `harness: opencode`)
runs one turn per prompt. legion never
decides what an agent should see: whoever calls `prompt` has decided that. Messaging belongs to
[msgboard](https://github.com/huggingface/msgboard) (the `message_board` resource), shared
state to a world (the `world` resource, docs/world.md).

## Architecture

[View the architecture diagram](architecture.svg).

```
   CLI / library ── apply, prompt, status, events ──┐
                                                    ▼
                    ┌──────────────────────────────────────────┐
                    │ controller                               │
                    │  FastAPI + SQLite + reconcile loop       │
                    │  specs · agents · workers · pending      │
                    │  prompts · events · loops · health       │
                    └──────┬───────────────▲───────────────────┘
          commands         │ long-poll     │ heartbeat, events
      ┌─────────────── worker (one job: HF Jobs, Slurm or a local process) ──┐
      │  resources (message_board, world, spawn), shared by the agents       │
      │  agent A: Harbor Trial (Pi)   agent B: Harbor Trial (Pi)   … up to K │
      │  sync workdirs + Pi sessions → bucket                                │
      └──────────────────────────────────────────────────────────────────────┘
```

### Controller

One process (an HF Job in its workers' network group, a Slurm job, or a local background
process: docs/deploy-job.md, docs/slurm.md, docs/local.md) with SQLite, the reconcile loop, the
loop engine and the health prober as background tasks. One bearer token is shared by the CLI and
workers; each agent gets its own scoped token (Spawn below).

It stores specs and their history, places agents on workers (fill in order, no bin packing, no
moving agents between live workers), launches and cancels worker jobs through the compute
backend, derives agent and worker states from heartbeats, holds pending prompts for agents
whose worker is down, indexes events, derives cost from tokens and serves the JSON API and an
HTML status page.

### Worker

One job per worker slot of a type. It hosts up to `agents_per_worker` agents, at most
`max_working_per_worker` of them in a turn. It holds one long-poll on the controller for
commands, posts one heartbeat every 30 s (random phase) and batches events every 3 s. It starts
the run's resources once (docs/comms.md, docs/world.md), runs the agents' Trials, syncs each
agent's workdir and Pi sessions to the bucket, and drains `drain_window` before its deadline
(no new turns, in-flight turns finish, final sync, exit).

### Agent

A name, a workdir and a Pi session, on whichever worker holds it. Its Trial (an environment on
docker, podman, hf-sandbox or e2b: `legion.worker.adapters.harbor`) starts on the first prompt and
stays up; each prompt is one Pi turn in it, and prompts that arrive during a turn join the next
one. A Trial that fails is reported and the next prompt starts a new one on the same session.

### CLI and library

One package with three entry points: `legion` (CLI and library: `init`, `launch`, `update`,
`stop`, `prompt`, `status`), `legion-controller` and `legion-worker`. `legion launch` compiles a
Harbor spec (`legion.harbor.spec`) into a swarm spec with one type per role, packs the run's
tools and resources folders and applies it.

### Adding a harness, a compute backend or a server starter

Each is one module plus one entry in `src/legion/registry.py`; the spec loader, controller,
worker and CLI read nothing else, and `tests/test_registry.py` checks every entry.

- **Harness**: write `src/legion/worker/adapters/<name>.py` with a class implementing
  `HarnessAdapter` (`adapters/base.py`) and a `make_adapter(workdir, name, spec)` function, and
  add a `HARNESSES` entry (its import path and the wire API preflight probes).
- **Compute backend**: write `src/legion/controller/<name>.py` with a `JobsClient`
  (`controller/jobs.py`) and a `<name>_jobs_client(env, dep, data_dir)` function, and add a
  `COMPUTE` entry (`compute.kind` / `LEGION_COMPUTE` name -> that function). Every kind but
  `local` needs `bucket_id` and `world.url` in the spec.
- **Server starter** (inference servers legion starts itself, like `hf_jobs`): add the block as a
  field of `Inference`, write `src/legion/controller/<name>.py` with a `Starter`
  (`controller/servers.py`: start or reattach, list, check readiness, logs, cancel, set a
  job's deadline) and a
  `<name>(env)` function, and add a `SERVERS` entry (the block's key -> that function).

## Agent lifecycle

### Resume across workers

`persist_workdir` (default true): the worker syncs each agent's workdir and Pi sessions to
`<bucket>/<swarm>/<type>/<agent>/` while it works and once more when it leaves, and a new worker
restores both before the agent's first turn there. File modes travel in
`<workdir>/.legion-modes.json`, since buckets keep only bytes. When a session cannot be resumed
the worker starts a fresh one under the same name, writes `session_restarted` and status shows
`resumed: false`.

### Workdir

The workdir's size is restore time. Regenerable trees (`node_modules/`, `.venv/`, `.cache/`,
`__pycache__/`, ...) are never synced; `workdir.ephemeral` directories are neither synced nor
restored. Above `workdir.soft_files` / `soft_mb` the agent's next prompt starts with a size
note (at most hourly); above `max_files` / `max_mb` a `workdir_over_quota` event follows too.
With `workdir.base: true` a restore downloads one tar plus the files changed since it instead of
every file. Details: `legion.worker.sync`.

### Worker lifetime

Workers live `worker_lifetime` (24 h) plus up to `lifetime_jitter` (4 h), so expiries spread
out; `drain_window` (10 min) before the deadline a worker drains, and the controller relaunches
the slot at once with its agents restored. Launches are spaced to stay inside the backend's rate
limits.

### Interrupted turns

Heartbeats carry in-flight turn ids. After a crash or a preemption the controller writes
`turn_interrupted` into each affected agent's stream and, when the type has a `resume_prompt`,
sends it after the restore.

## Resources

A run's `resources:` are started by each worker and shared by its agents' Trials; their tools
reach the agents through `legion-tool` inside the Trial (`legion.harbor.builtin`).

- `message_board`: a msgboard board (local, a Space, or a `url`), its `message_board.*` tools
  with the posting rules checked before the board is called, and its mentions as `resource`
  events that wake loops (docs/comms.md).
- `world`: an HTTP service holding the swarm's shared state (World API v1); `world.*` tools call
  it with the agent's name (docs/world.md).
- `spawn`: the `spawn.*` tools, which call the controller with the agent's own token (Spawn).

## Spawn

A leader agent creates, prompts and removes child agents of other types (docs/spawn.md). A child
is an ordinary agent with `parent` set and no replica slot, placed on its own type's workers.
`spawn.<type>` lists the types an agent may create, with caps and a `depth`; `apply` rejects
cycles. Each spawnable type declares `workers: {initial, max_extra}`; `limits` caps the swarm
(`alive`, `spawned_total`, `budget_usd`). A refusal is a 409 with a `reason`.

Each agent gets `LEGION_AGENT_TOKEN`, an HMAC of its name and a nonce under a key derived from the
controller token; the controller stores only its hash. It may act on the agent, its subtree, and
prompt its parent. The worker holds it and makes the agent's controller calls with it (the
`spawn` resource).

## Loops and the budget stop

A type's `loop:` (docs/loop.md) keeps its agents prompted: the loop prompt on `every`
(`turn_end` or a duration) and on its `on:` wakes while `while` holds, then one `final` prompt
per live agent. The loop engine (`controller/loop.py`) ticks once a second and sends prompts
through the bulk-prompt path (`state: idle`, `skip_pending`), so loops never stack prompts and
respect `max_working`. `limits.budget_usd` is a stop: at the budget minus the final reserves
loops stop and final rounds run; at the budget every agent is stopped and prompts are refused.

## Health

The controller samples each swarm every 30 s (docs/health.md): agent and worker states,
provider and world probes, usage, rollups and alerts, read by `legion status` and the status
page.

## Checkpoint and resume

The controller's SQLite and the agents' event files are its state. Where its disk does not
survive a restart (job mode), it syncs them to `LEGION_STATE_URL` in a bucket every 60 s and once
more on SIGTERM, and a new controller restores them before it opens the store
(`legion.controller.state`). Agents resume from their own bucket prefix (Resume across workers).

## Contracts

### Spec

A swarm spec (`SwarmSpec`) names the swarm, its compute, optional billing `namespace` and `bucket_id`, its `inference:`
block and its `types:`; `legion launch` writes it from a Harbor spec (README, docs/local.md).
Each type sets `harness: harbor`, its `harbor:` block (written by the compile step), its model
(and optionally its own `inference` and its `pricing`), `replicas`, packing (`agents_per_worker` 50, `max_working_per_worker` 20), lifetime, `workdir`,
`worker` knobs, `loop`, `spawn` and `resume_prompt`. The blocks inside the spec are `extra: forbid`
(an unknown key is refused); `SwarmSpec` and `TypeSpec` themselves drop unknown keys.

`namespace` is the optional billing account for worker/controller Jobs, hf-sandbox and
`hf_jobs` model servers. Unset, Jobs and sandboxes use `HF_TOKEN`'s account. Router requests
send `X-HF-Bill-To` only for an explicit namespace; otherwise the inference key's owner pays.
E2B sandboxes are billed to `E2B_API_KEY`'s account, not `namespace`.
The stored namespace stays empty when omitted. There are no per-inference billing overrides.
In job mode, a spec's explicit namespace or HF_TOKEN's resolved account must match the
controller job's namespace to join its network group. A manually org-deployed controller
refuses a blank spec namespace if the accounts differ and asks for an explicit namespace.
Matching-account runs keep the stored namespace empty and send no router bill-to header.

`bucket_id` is `[namespace/]name`, with no URL or path prefix. `legion launch` qualifies a bare
name using the explicit namespace, else `HF_TOKEN`'s account, and creates the bucket with
`private=True, exist_ok=True` before starting jobs. Existing visibility is retained. Only the
qualified ID is stored and sent to workers (`BUCKET_ID`); controllers and workers never
resolve it again. Legion still adds its checkpoint and per-agent prefixes internally. Init
reports a missing bucket as one launch will create. See [billing and storage](billing.md).
Bucket preparation also applies to local compute and the foreground `localrun` launch
entry point. Disabling controller checkpointing does not skip an explicitly configured
bucket's initialization or its credentials check.

### Inference

`types.<t>.model` is a Hub id (with `urls`, the name the servers serve) and one `inference` block
(`Inference`) says where it runs: the HF router (`hf_router`, the default; legion sends
`<model>:<provider>`), OpenAI-compatible servers you run (`urls`, each agent calling one by
rendezvous hashing on its swarm and name) or vLLM servers legion starts on HF Jobs (`hf_jobs`).
The swarm's block serves every type that does not set its own. For `hf_jobs` the controller
starts one group of jobs per model before preflight (`POST /swarms/{swarm}/servers`,
`controller/servers.py`), waits until each answers, pins their URLs into the resolved block as
`urls`, reattaches to them by label after a restart and cancels them when the run ends; from
there on it is the `urls` path. One rule cleans up: a server job the applied spec does not use
goes once the deadline in its `retire` label has passed. Validation
pins each type's block and `[input, output, cached]` price into the spec
(`types.<t>.resolved_model`, under the block's name: `inference` or `types.<t>.inference`), and
the apply checks the model against each endpoint's `GET /models` and pins the context window it
is served with (`controller/inference.py`), which Pi's model config and `legion status` get. The
block's key goes into the type's job secrets, preflight probes each model with a tool call, and
`max_working` caps the agents of the swarm in a turn on the block, enforced at prompt admission.
`legion update` may change a running Harbor type's block: anything but `max_working` restarts the
type's workers (they hold the key, their Trials the endpoints and allowlist). How-to and
fields: docs/open-models.md.

### States

| State | Meaning |
|---|---|
| `working` | the agent is in a turn |
| `idle` | no turn; `idle_since` is the end of the last one |
| `stuck` | a turn exceeded its timeout and was killed; cleared on the next turn |
| `unknown` | worker not seen for `worker_unknown_after`, or the agent's own report is that old |
| `dead` | worker not seen for `worker_dead_after`, or the job is terminal |

"Seen" is the worker's last heartbeat, command poll or event post from its current job. Workers
retry 429 and 5xx with backoff and jitter, never sooner than `Retry-After`. A type is
`crashlooping` after consecutive worker crashes and `launch_refused` when the backend refuses a
launch (a 4xx); either pauses launching until the next apply.

### API (controller)

Direct calls use `Authorization: Bearer $LEGION_TOKEN`. Through the HF Jobs proxy,
use the [deployment guide's authentication headers](deploy-job.md#two-urls-two-auth-paths).
The status page and `/healthz` do not require controller authentication; a hosting
proxy may still require its own authentication.

| Method | Path | Purpose |
|---|---|---|
| PUT | `/swarms/{swarm}` | apply a spec |
| GET | `/swarms` | list swarms |
| POST, GET | `/swarms/{swarm}/tools[/{version}]` | upload / fetch the run's packed tools and resources folders |
| POST | `/swarms/{swarm}/preflight` | probe each model on its inference block |
| POST, GET, DELETE | `/swarms/{swarm}/servers` | start (or reattach to) the spec's `hf_jobs` servers; their status; cancel them |
| GET, DELETE | `/swarms/{swarm}` | spec and aggregates; drain and remove everything |
| GET | `/swarms/{swarm}/agents` | paginated agent list |
| GET | `/agents/{name}` | one agent's status |
| POST | `/agents/{name}/prompt` | prompt one agent; acks with `turn_id` when a turn started |
| POST | `/swarms/{swarm}/prompts` | one pending prompt per matching agent (`type`, `state`, `skip_pending`) |
| GET | `/agents/{name}/events?after=&wait=` | the agent's event stream, long-poll |
| GET | `/workers`, `/workers/{id}` | workers |
| GET | `/workers/{id}/commands?wait=` | worker long-poll (worker token) |
| POST | `/workers/{id}/heartbeat`, `/workers/{id}/events` | heartbeat and event batches (worker token) |
| POST | `/agents/{parent}/spawn` | create a child agent ([spawning](spawn.md)) |
| DELETE | `/agents/{name}` | remove a child agent |
| GET | `/whoami` | authenticated identity |
| GET, POST | `/swarms/{swarm}/loop` | inspect or control loops ([loops](loop.md)) |
| GET | `/swarms/{swarm}/events` | swarm-level events |
| GET | `/swarms/{swarm}/health`, `/swarms/{swarm}/health/history?since=&res=` | health snapshots ([health](health.md)) |
| GET | `/` | HTML status page |
| GET | `/healthz` | health and request counters |

Single-agent prompts accept `{text, timeout?, reply_to?}`. Bulk prompts accept
`{type, state?, text, timeout?, reply_to?}` and return counts. With `state: idle`,
agents that already have a pending prompt are skipped, so repeated bulk calls do
not stack prompts. Agent lists accept `type`, `state`, and `cursor` filters.

Prompt results are read from the event stream. A prompt for an agent whose worker is down is
held in SQLite and delivered on the replacement's first poll (at least once). Each launch gets an
incarnation token; requests from a replaced job get a 409.

### Heartbeat

One POST per worker every 30 s: its deadline, draining flag, `spec_rev` and one record per agent
(state, turn id, idle since, session id, cumulative tokens by model, last sync). The response
carries the new type spec when `spec_rev` differs; the worker applies it at each agent's next
turn boundary (docs/live.md). Optional telemetry: `mem` (the worker's cgroup), `restore`
(restores in flight), `knobs_rev`, `wants_rotation`, and per agent `rev` / `start_rev`.

### Events and sessions

One stream per agent in pi session format v3, from Pi's own events plus the worker's
`session_started`, `session_resumed`, `session_restarted`, `turn_interrupted` and `sync`. The
bucket holds the same stream that `/agents/{name}/events` serves.

### Usage and cost

Tokens per model per turn come from the events; the controller prices each heartbeat's new
tokens with the pinned price (or its price table), so a price change applies from then on and
does not reprice history.

## Failure handling

| Event | Response |
|---|---|
| Worker reaches its deadline | drain, then relaunch at once with its agents restored |
| Worker crashes | relaunch with backoff; `turn_interrupted`; `resume_prompt` if set |
| Consecutive crashes of one type | `crashlooping`, launching paused |
| Launch refused (4xx) | `launch_refused` with the backend's message, no retries until re-apply |
| Launch fails transiently | retry, 30 s doubling to 5 min; last error in status |
| Turn exceeds its timeout | turn killed, agent `stuck` once, then idle |
| Session cannot resume | fresh session, `session_restarted`, `resumed: false` |
| Controller restarts | restore its state (Checkpoint and resume), check workers against the backend's jobs, reattach to `hf_jobs` servers by label, wait for heartbeats |
| An `hf_jobs` server fails to start | the launch stops with the reason and the job's last log lines; its jobs are cancelled |
| Rate limits (429) | back off, honour `Retry-After`, keep launch spacing |
