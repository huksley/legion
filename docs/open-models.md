# Open models in legion

A spec names its model with `model`, a Hub id, and says where it runs with one `inference` block:

- **`hf_router`**: the [HF router](https://huggingface.co/docs/inference-providers/index)
  (Inference Providers). This is the default when the spec has no `inference` block.
- **`urls`**: OpenAI-compatible servers you run, on a Slurm cluster, a cloud box or your own
  machine.
- **`hf_jobs`**: vLLM servers legion starts on [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
  for the run, and cancels when it ends.

The block has the same shape in a Harbor spec (the run's, or a role's own) and in a swarm spec
(the swarm's, or a type's own). This page is the how-to; the contract is in
[design.md](design.md) "Inference".

## The spec

```yaml
model: Qwen/Qwen3.8-27B            # a Hub id; with urls, the name the servers serve

# The HF router (the default)
# namespace: my-org              # optional org billing; omit for personal-account defaults
inference:
  hf_router:
    provider: deepinfra            # a provider, or a policy: fastest (the router's default), cheapest, preferred
  api_key_env: HF_INFERENCE_TOKEN  # default: HF_TOKEN

# Servers you run
inference:
  urls: [http://10.53.196.166:8000/v1, http://10.53.196.167:8000/v1]
  api_key_env: VLLM_API_KEY        # optional: no key by default

# Servers legion starts on HF Jobs
inference:
  hf_jobs:
    flavor: a100-large             # a GPU flavor (`hf jobs hardware`)
    tool_call_parser: qwen3_coder  # Pi needs tool calls
    reasoning_parser: qwen3
    vllm_args: [max_model_len=131072]
    startup_timeout: 30m           # default 20m
  api_key_env: HF_INFERENCE_TOKEN  # default: HF_TOKEN

# Settings for any of them
inference:
  max_working: 3                   # agents of the swarm in a turn at once on these endpoints
  context_window: 131072           # used only when the servers report none
  reasoning_effort: low            # Pi's --thinking level
  metrics: true                    # urls only: scrape each server's vLLM /metrics (docs/health.md)
```

| Field | Meaning |
|---|---|
| `hf_router.provider` | A router provider (`deepinfra`, `cerebras`, ...), or a selection policy: `fastest`, `cheapest`, `preferred`. Unset: the router's default (`fastest`). legion sends `<model>:<provider>`. |
| `urls` | One or more base URLs ending in `/v1`, all serving `model`. A comma-separated string works too (`urls: ${VLLM_URLS}`). |
| `hf_jobs.flavor` | Required. The GPU flavor of each server's job, e.g. `l4x1`, `a100-large`, `h200`. |
| `hf_jobs.tool_call_parser` | Required. vLLM's `--tool-call-parser` (`qwen3_coder` for Qwen3.5 and Qwen3.8); legion always adds `--enable-auto-tool-choice`. |
| `hf_jobs.reasoning_parser` | vLLM's `--reasoning-parser` (`qwen3`). |
| `hf_jobs.vllm_args` | Every other `vllm serve` flag, as `KEY=VALUE` strings ([below](#servers-legion-starts-hf_jobs)). |
| `hf_jobs.timeout` | The job's lifetime, the backstop if the run never cancels it. Default: the run's `limits.run_timeout_seconds` plus `startup_timeout` plus 15 minutes. |
| `hf_jobs.startup_timeout` | How long legion waits for every replica to answer (default `20m`). |
| `hf_jobs.replicas` | Servers of the model (default 1); each agent calls one, as with several `urls`. |
| `hf_jobs.image` | The vLLM image (default `vllm/vllm-openai:v0.33.0`). |
| `api_key_env` | The env var that holds the key (never the key itself). Default: `HF_TOKEN` for `hf_router` and `hf_jobs`, none for `urls`. |
| `max_working` | See [`max_working`](#max_working). |
| `context_window` | The model's context window in tokens, for servers that do not report one. |
| `reasoning_effort` | Pi's `--thinking` level (`off` ... `max`). |
| `metrics` | `urls` only: scrape each server's vLLM `/metrics` into the health snapshots. |

A spec sets one of `hf_router`, `urls` and `hf_jobs`. A role (Harbor spec) or a type (swarm spec) may set its own `model`, on the run's endpoints, and
its own `inference`, which replaces the run's block as a whole. `pricing` stays where it was: the
run's and a role's `[input, output, cached]` in USD per 1M tokens (see [Cost](#cost)).

The top-level `namespace` selects the billing account for all HF Jobs (including vLLM),
hf-sandbox and router calls. With it unset, Jobs use `HF_TOKEN`'s account and the router bills
the inference key's owner; legion sends no `X-HF-Bill-To` header. With an org set, the inference
key needs Inference Providers permission on that org (`inference.serverless.write`); for
`hf_jobs` it also needs read access to the org to pass the jobs proxy. `HF_TOKEN` needs Jobs
and bucket write access there. See [billing and storage](billing.md) for token details.

`legion launch` compiles a Harbor spec's block into the swarm spec as it is. Each type's
`resolved_model` then pins the block it runs on, its price and, at apply, the context window
(below). Harbor's Pi speaks OpenAI **Chat Completions** (`POST {url}/chat/completions`).

## The HF router (`hf_router`)

`model` is the Hub id the router lists (`Qwen/Qwen3.8-27B`, not `huggingface/...` and no
`:provider` suffix: the provider goes in `hf_router.provider`). Pin a provider that serves the
model with tool calls: some router providers fail every turn after a tool call, and preflight
catches that, since it sends a tool result back. `GET https://router.huggingface.co/v1/models`
lists each model's providers with their context length, tool support and prices.

The key defaults to `HF_TOKEN`. Every agent's container gets it, so with HF Sandboxes, HF Jobs or
Slurm workers the spec compiler refuses `HF_TOKEN` (the workers' own Hub token): create a
[fine-grained token](https://huggingface.co/settings/tokens/new?tokenType=fineGrained) with only
**Make calls to Inference Providers** and name it, e.g. `api_key_env: HF_INFERENCE_TOKEN`
(experiments/mini-swarm/README.md).

## Servers you run (`urls`)

Any OpenAI-compatible server with tool calling works: vLLM or SGLang on a Slurm cluster,
llama.cpp's `llama-server`, Ollama, a box on the LAN, or another hosted router. Turn tool calling
on, e.g. vLLM's `--enable-auto-tool-choice --tool-call-parser <parser>` (the parser depends on the
model family; `qwen3_coder` for Qwen3.5 and Qwen3.8).

- **Several servers.** Each agent calls one of the `urls`, picked by rendezvous hashing on its
  swarm's name and its own: the agents spread over the servers, as do the same names in
  concurrent swarms (each run's `agent-001`); an agent keeps its server, and so its prefix cache,
  across Trials and restarts while the list is the same; and a server added or removed moves only
  the agents that must move. Preflight probes every server, and with `metrics: true` the health
  snapshots cover each one (docs/health.md).
- **Reachability.** The agents must reach every URL; the controller's checks use the URLs as
  written.
  - A private cluster address (`10.x`) needs the swarm on the same cluster (`compute: slurm`,
    docs/slurm.md).
  - `localhost` is the machine that runs legion. With `environment.backend: podman` legion gives
    the agents `host.containers.internal` instead, the name podman puts in every container's
    `/etc/hosts` for its host, and allowlists it. The server must listen on the host's address,
    not only on `127.0.0.1`: vLLM does by default, Ollama with `OLLAMA_HOST=0.0.0.0`.
  - HF Sandboxes and HF Jobs workers cannot reach your machine or a private network: the spec
    compiler refuses `localhost` and private addresses there. Serve the model at a public URL.
    Under `compute: slurm`, `localhost` would be each worker's own node and is refused too.
- **Keys.** A server without a key needs no `api_key_env`; Pi then gets a placeholder key.
- **A server that moves.** When a server restarts on another node, update `urls` and run
  `legion update`. A change to a running type's block (anything but `max_working`) restarts the
  type's workers: they drain (in-flight turns finish), and their agents resume on new workers,
  whose Trials call and allowlist the new servers and get the new key.
- **Cost.** $0 unless the spec sets `pricing`; the real cost is the GPUs' time, which legion does
  not count.

### Your own machine

Serve the model, then run experiments/mini-swarm/urls.yaml: mini-swarm's three agents in
rootless podman containers, on `Qwen/Qwen3.5-0.8B` at `http://localhost:8000/v1`, with a local
board.

```sh
vllm serve Qwen/Qwen3.5-0.8B --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
# or Ollama (MODEL is its own name):
#   OLLAMA_HOST=0.0.0.0 ollama serve & ollama pull qwen3.5:2b
#   MODEL=qwen3.5:2b SERVER_URL=http://localhost:11434/v1 uv run legion launch ...
uv run legion launch experiments/mini-swarm/urls.yaml
uv run legion status mini-swarm-urls
```

### On the HF cluster

The server runs on a GPU node, in Pyxis; the agents run in rootless podman on a CPU node and
reach it by the GPU node's address. The GPU nodes' podman (3.4, with CNI plugins that refuse the
networks it writes) cannot start the containers on the network legion gives each Trial; the CPU
nodes' (4.9) can. `scripts/slurm/podman-node.sh` sets up podman in a shell on a node as
`worker.sbatch` does for Slurm workers (storage under `/tmp`, no user systemd).

```sh
sbatch -p hopper-b200 serve-vllm.sbatch Qwen/Qwen3.5-0.8B
# its log ends with: serving Qwen/Qwen3.5-0.8B at http://<node address>:8000/v1
srun -p hopper-cpu -n 1 -c 8 --mem=16G --pty bash
source .venv/bin/activate && source scripts/slurm/podman-node.sh
SERVER_URL=http://<node address>:8000/v1 legion launch experiments/mini-swarm/urls.yaml
legion status mini-swarm-urls
```

`serve-vllm.sbatch`, one vLLM server on one GPU (the Slurm recipe below uses it too):

```bash
#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.log
set -uo pipefail
MODEL=${1:-Qwen/Qwen3.5-0.8B}
PORT=${PORT:-8000}
IMAGE=${IMAGE:-/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh}
export HF_HOME=/hf
mkdir -p "/scratch/$USER/hf"  # the weights, on the node's own disk
srun --overlap --container-image="$IMAGE" --no-container-mount-home \
  --container-mounts="/scratch/$USER/hf:/hf" --container-env=HF_HOME \
  vllm serve "$MODEL" --port "$PORT" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 &
until curl -sf -o /dev/null "localhost:$PORT/v1/models"; do
  sleep 10
  kill -0 $! 2>/dev/null || { echo "vLLM exited during startup"; exit 1; }
done
echo "serving $MODEL at http://$(hostname -I | awk '{print $1}'):$PORT/v1"
wait
```

### Slurm

Big models are servers you run: start them with your own job script (one per server, e.g.
`serve-vllm.sbatch` above, or a multi-node one behind a router), then list their URLs from a
`compute: slurm` spec (docs/slurm.md). Its workers run on the cluster, so they reach the nodes'
addresses:

```yaml
compute:
  kind: slurm
model: Qwen/Qwen3.5-0.8B
inference:
  urls: ${VLLM_URLS}               # run.env: VLLM_URLS=http://10.53.1.2:8000/v1,http://10.53.1.3:8000/v1
environment:
  backend: podman
```

`legion init` checks that the servers answer a tool call (its `providers` check), and
`legion launch` checks the model and pins its context window. A server that restarts on another
node gets a new address: update `VLLM_URLS` in run.env and run `legion update` (the type's
workers restart on the new URLs).

## Servers legion starts (`hf_jobs`)

With `hf_jobs`, nobody starts a server by hand or copies a URL. `legion launch`:

1. asks the controller to start the servers (`POST /swarms/{swarm}/servers`): one group per
   model, `replicas` jobs each, or the running ones it finds again after a restart;
2. waits until every replica answers `GET /v1/models` through the jobs proxy, for at most
   `startup_timeout`, printing each job's progress (`SCHEDULING (Pulling container image)`,
   `RUNNING`, `ready`);
3. runs preflight against them, and the apply pins their URLs into the spec as `urls` would be
   (`types.<t>.resolved_model.inference.urls`).

From there on everything is the `urls` path: the model check, the context window read from the
servers, preflight and Pi's model config. The run's end cancels the jobs: `legion stop --delete`,
or a local run that ends by itself. `legion update` of a changed `hf_jobs` block or model starts
its new servers first. The apply then pins them, and, as for any block change, the type's
workers restart on them. With `hf_jobs`, `max_working` is also the servers' `max_num_seqs`, so
changing it starts new servers too.

**Cleanup is one rule**, which the controller applies to the swarm's server jobs every 30
seconds: a job the applied spec does not use (its group, or a replica beyond the group's
`replicas`) is cancelled once the deadline in its `retire` label has passed. The deadline sits
on the job, so a restarted controller applies the same rule.

- A new job's deadline is the time its launch or update has to apply it: `startup_timeout` plus
  5 minutes, shortened to 5 minutes once it answers. An update that is refused or interrupted
  thus leaves the applied servers as they were, and its own go shortly after.
- An apply that stops using a job (another block, fewer replicas) gives it the types' longest
  `turn_timeout`, for the turns of the agents draining from it. A reused job's deadline is only
  ever pushed back, and a job the spec uses again keeps serving.
- A failed group's jobs, and every job of a deleted swarm, are cancelled at once. A cancel the
  Hub refuses is tried again at the next sweep.
- Only jobs with an `inference` label are server jobs; the swarm's worker jobs are never
  touched.

```text
servers Qwen/Qwen3.8-27B (hf_jobs a100-large, namespace you, types agent, context 131072):
  replica 0: job 6afb00a85b0bff80b2ac4d54 RUNNING, ready, https://6afb00a85b0bff80b2ac4d54--8000.hf.jobs/v1, up 14m
```

- **One group per model.** A vLLM server serves one model, so a role with its own `model` gets
  its own jobs: with the run's `hf_jobs` settings, or with its own if it sets its own
  `inference` block (another flavor, say). Roles with the same model and settings share them,
  with the most `replicas` (and the longest `startup_timeout`) any of them asks for.
  `legion status`, `GET /swarms/{swarm}/servers` and the status page list each job by model,
  with its URL, stage, state, flavor and uptime.
- **The job.** `vllm serve <model>` on `hf_jobs.image`, with port 8000 exposed through the jobs
  proxy (`expose=[8000]`, as `hf jobs run --expose 8000`) at `https://<job id>--8000.hf.jobs`,
  and `hf_jobs.timeout` as its `--timeout`. The jobs carry labels as workers do (`swarm`,
  `inference`: the group's settings hash, `replica`, `retire`: the deadline above); a restarted
  controller finds them by these instead of starting new ones (it retries until the Hub
  answers). The weights download inside the job, which gets no Hub token, so the model must be
  public.
- **The vLLM arguments** follow trl-internal's evals config. legion sets `--host`, `--port`,
  `--served-model-name` (the Hub id) and the tool-calling flags. `vllm_args` holds every other
  flag as a `KEY=VALUE` string: `max_model_len=131072` becomes `--max-model-len 131072`, and
  `true` and `false` become `--flag` and `--no-flag`. Strings keep YAML from guessing types. A
  key legion owns is refused: `host`, `port`, `served_model_name`, `model`, the tool-calling and
  reasoning flags (named fields), and `api_key` ([below](#keys-and-the-jobs-proxy)). So is a key
  given twice. `max_num_seqs` (the block's `max_working`) and `tensor_parallel_size` (the
  flavor's GPUs, divided by any `data_parallel_size` or `pipeline_parallel_size`) are filled in
  unless `vllm_args` sets them.
- **Failures** stop the launch with the reason and the job's last log lines, show in `legion
  status`, and cancel the group's jobs: an image that does not pull, vLLM running out of memory
  (a `max_model_len` the GPU cannot hold), or no answer within `startup_timeout` (also when
  the Hub does not answer). A start refused part-way (a quota) cancels the jobs it started. An
  apply drops failed groups the spec no longer uses from the status. A job that ends
  during the run marks its server `failed` in `legion status` (the verdict is failing); legion does
  not replace it.
- **Not cancelled:** `legion stop --release` keeps the servers (the run is held, not over), and a
  controller killed outright leaves them to their `timeout`. A `legion launch` of the same spec
  afterwards reuses them.
- **Cost.** The flavor's hourly price while the job runs (`hf jobs hardware`). legion prices the
  tokens as for `urls` ($0 unless the spec sets `pricing`) and does not count GPU time.

### Keys and the jobs proxy

The jobs proxy checks every request for `Authorization: Bearer <HF token>`, a token with read
access to the job's namespace, and answers 401 otherwise. The agents' key (`api_key_env`) is
such a token. The controller's own `HF_TOKEN` starts, lists, probes and cancels the jobs. The key
sits in every agent's container, so with HF Sandboxes, HF Jobs or Slurm workers it must not be
`HF_TOKEN` (the spec compiler refuses it): give the agents a fine-grained token of their own.

What a request to an exposed job gets, as tested with an echo server and with vLLM v0.33:

| Exposure | No token | A bogus token | An HF token that can read the namespace | Headers the container sees |
|---|---|---|---|---|
| `--expose 8000` | 401 | 401 | 200 | no `Authorization`; others (`X-Api-Key`) pass |
| `--expose-public 8000` | 200 | 200 | 200 | no `Authorization` |

- **legion uses `--expose`.** The proxy checks the HF token; vLLM runs without a key behind it.
- **`--expose-public` with vLLM's `--api-key` does not work.** The proxy drops `Authorization` in
  both modes, and vLLM's key check reads only `Authorization: Bearer`, so every request gets
  vLLM's 401. A key in another header gets through the proxy, but vLLM would need a shim in the
  job to check it. legion refuses `api_key` in `vllm_args`.
- **The agents' token.** The Hub docs ask for "an HF token with read access to the job's
  namespace". Tested: a fine-grained token of the namespace's user with `repo.content.read`,
  `inference.serverless.write` and `job.write` passes; so does the user's write token. Not
  tested: a token without `job.write` (whether repository read alone is enough), or an org's
  namespace.

### `legion serve`

`legion serve <spec>` starts a spec's `hf_jobs` servers on their own, with the code a run uses,
waits until they answer, and prints their URLs, the job ids and how to stop them. Several runs
can then share one server, without each paying for its startup, by listing its URL:

```sh
export BOARD_SPACE="mini-swarm-$(uv run python -c 'import uuid; print(uuid.uuid4().hex[:12])')"
uv run legion serve experiments/mini-swarm/hf-jobs.yaml --timeout 4h
# ready:
#   Qwen/Qwen3.8-27B replica 0: https://<job id>--8000.hf.jobs/v1 (job <job id>, a100-large, namespace <your-username>)
# A run uses them with, in its spec:
#   model: Qwen/Qwen3.8-27B
#   inference:
#     urls: [https://<job id>--8000.hf.jobs/v1]
#     api_key_env: HF_INFERENCE_TOKEN  # an HF token that can read the job's personal namespace
uv run legion serve experiments/mini-swarm/hf-jobs.yaml --stop   # or: hf jobs cancel <job id>
```

This swarm config requires `BOARD_SPACE`; `serve` itself starts no board. If you later
launch independent swarms with a Space board, give each one its own name. Bare names resolve
under `HF_TOKEN`'s personal account; qualified IDs explicitly select another permitted account.

It needs no controller: `HF_TOKEN` (or the saved login) starts and probes the jobs. They are
labelled `serve` instead of `swarm`, so no run's controller adopts or cancels them; only
`--stop` (every job `legion serve` labelled for the spec's name) or their `timeout`
(`--timeout`, default the spec's) ends them. With `--run` (or `LEGION_RUN_DIR`) it reads the
run's cluster.env and run.env, as `legion launch` does. legion does not share
servers across runs by itself.

## Model check and context window

At apply the controller reads `GET <url>/models` from each endpoint, as written:

- **The router** must list `model`. A named `provider` must serve it with tool support; the
  apply lists the providers that do when it does not.
- **Each server you run** must serve `model`; the apply lists what it serves when it does not.

It then pins the context window the model is served with into the spec
(`types.<t>.resolved_model.context_window`): the smallest over the servers, or the router
providers a request may reach (the named one, or under a policy every one with tool support).
Each counts with its own window (vLLM's and SGLang's `max_model_len`, the router's
`context_length`), or `inference.context_window` when it reports none; one without either
refuses the apply, naming it. Pi's model config gets it as `contextWindow`, so Pi
compacts before the server refuses a request, and `legion status` shows it per model. A re-apply
with the same block and model keeps the pinned value without asking the servers again.

## Keys

The spec only ever names a key (`api_key_env`). The value lives in three places, each by that
name:

| where | who puts it there | used for |
|---|---|---|
| controller environment (Space secrets, run.env) | you | the model check, preflight; sent to jobs |
| worker job secrets | the controller, for the types on that block | read by the worker |
| the agent's Trial | the worker (Harbor) | Pi's requests |

How the key travels: `Authorization: Bearer <key>`. Controller secrets to set: every
`api_key_env` your blocks name (or `HF_TOKEN` for the router and `hf_jobs` by default), plus the
`secrets` your types list, plus `LEGION_TOKEN`; with `hf_jobs`, also `HF_TOKEN`, which starts the
jobs. A Trial's environment is the task's plus the inference key, the
tools and the resources; `LEGION_TOKEN` and the workers' `HF_TOKEN` stay in the worker.

## Harnesses

A Harbor spec's `harness` picks the agent in each Trial: `pi` (the default) or `opencode`.
Both run one turn per prompt in the agent's long-lived Trial (`legion.harbor.agent.TurnServer`)
and take the same `inference` block:

```yaml
harness: opencode
model: deepseek-4-1-flash-1017
inference:
  urls: [https://inference.example.com/v1]
  api_key_env: EXAMPLE_API_KEY
  context_window: 131072
```

OpenCode (`legion.harbor.harnesses.opencode`) runs `opencode run --format=json` per turn, and
`--continue` after the first. Harbor installs the pinned `opencode-ai` release
(`OPENCODE_VERSION`) in the task's container. The model is registered under one generated
OpenAI-compatible provider, `legion`, with the block's base URL, key and context window
(`limit.context`, so OpenCode compacts before the server refuses a request). Its sessions are
OpenCode's data dir, kept and restored like Pi's. Undo snapshots are off: a task's working
directory is often `/`. Not supported yet with OpenCode: `limits.agent_max_turns` (the type's
`turn_timeout` bounds a turn) and `inference.reasoning_effort`.

## Reasoning

`inference.reasoning_effort` is Pi's `--thinking` level. Whether a server honours it is up to
the server; Qwen3 on vLLM turns thinking off through the chat template, not the effort, so set
such defaults on the server (vLLM can set default chat-template arguments at startup).

## Cost

Tokens are reported under the type's `model` and priced from the `[input, output, cached]` row
pinned at apply: the role's `pricing` for its own model, else the run's `pricing`, else
`[0, 0, 0]`. The router's prices are not fetched for you (`GET /v1/models` lists them per
provider). Cache writes count at the input price. A re-apply with a new price changes the cost
of new tokens only.

Cost is only as good as the usage the server reports. A turn that completes with zero tokens is
flagged: the worker logs a warning (once per agent), the `turn_ended` event carries
`usage_missing: true`, and `legion status` and the status page count such agents per inference
block (`zero_usage_agents`) with a "cost unknown" warning.

## Preflight

`legion update --check` on a running swarm (or `POST /swarms/{swarm}/preflight` with the spec as
the body) makes the controller probe each distinct model and inference block the way a harness
starts. First the model check above, then two non-streamed Chat Completions turns, as Pi calls
it (a tool call written as text is no tool call, since Pi takes only `tool_calls`): turn one asks
for a call to a trivial `get_time` tool; turn two sends the model's output back with the tool's
result and must complete. Per model it reports `ok`, `tool_call_seen`, `follow_up_ok`,
`usage_reported`, `latency_ms`, `error` and `warning`. `ok` needs both turns: a server that
answers turn one and fails the follow-up (some router providers do) fails preflight, with a hint
at the likely cause (another `hf_router.provider`, or the server's tool-calling flags). A server
that reports zero token usage passes with a warning, since legion then cannot count cost. `legion
launch` runs preflight and refuses to apply on a failure; `--skip-preflight` bypasses it.

Preflight never passes silently. A model that was not probed (its key is not in the
controller's environment, or `--skip-preflight`) is reported as `unverified` with a warning
line. `launch` blocks on definite problems (a failed probe) and proceeds on missing evidence
(unverified), unless `--strict` is passed, which counts unverified as failure (`legion update
--check --strict` exits 1 the same way).

## `max_working`

The number of agents of the swarm on an inference block that may be in a turn at once, counted
per agent (an agent is one slot whether its turn makes one request or fifty). The controller
enforces it, on by default:

- **Queue, not reject.** A prompt for an agent that is not in a turn is accepted as usual
  (`POST /agents/{name}/prompt` never refuses it) but stays queued in the controller while the
  block is at its cap. A prompt for an agent already in a turn takes no new slot.
- **Oldest prompt first.** Free slots go to pending prompts in the order they were created,
  across all workers on the block, so a worker that polls often cannot keep re-taking
  freed slots. The worker owning the next prompt in line is woken as soon as a slot frees.
- **Every turn start is admitted the same way.** Admission happens where the controller hands
  queued prompts to workers, so it covers API prompts and `resume_prompt` after a crash (on a
  capped block it is queued behind the assign instead of riding inside it). Anything that
  queues a prompt in the controller is covered too, e.g. a spawned child's first prompt.
- **Counts come from worker heartbeats, corrected by turn events.** A prompt handed out
  takes its slot at once (two polls between heartbeats cannot both take the last slot), and
  the agent's `turn_ended` event releases it at once (events reach the controller every 3 s),
  not at the next heartbeat. The cap is still **soft by about one heartbeat**: a heartbeat
  built before the worker saw a turn change can briefly overrule an event.
- **The real limit belongs at the endpoint.** `max_working` keeps a swarm from overrunning a
  server it shares; it does not protect the server from other clients, and it counts agents,
  not requests. Set the server's own concurrency limits (vLLM `--max-num-seqs`, a gateway
  rate limit) for a hard bound.

`LEGION_MAX_WORKING_ENFORCE=0` on the controller is an escape hatch: report only, no queueing.

## Status

`legion status`, `GET /swarms/{swarm}` (`status.providers`) and the status page show each
inference block by name (`inference`, or `types.<t>.inference` for a type's own): where it
points, each model's context window, and `working / max_working`; with `hf_jobs`, each server
job by model (`status.servers`). `GET /agents/{name}` and
`legion status --agent` show an agent's block (`provider`) and the server it calls
(`endpoint`); `session_started` events carry `provider` and `model_id`.

## Not verified yet

- Cost on the HF router: some router providers stream zero usage, so their tokens show as $0
  with the zero-usage warning.
