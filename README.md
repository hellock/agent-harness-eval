**English** | [中文](README.zh.md)

# Agent Harness Eval

A framework for objectively evaluating and comparing different AI agent systems (harnesses). Instead of testing models, it tests **how well each agent framework orchestrates the same model** — including tool selection, multi-step planning, error recovery, and cost efficiency.

Given the same model and the same set of tasks, different agent harnesses will produce different results due to differences in their agent loops, system prompts, tool organization, and execution strategies. This framework measures those differences across multiple dimensions: task completion rate, output quality, latency, token usage, and cost.

## Quick Start

```bash
git clone https://github.com/hellock/agent-harness-eval agent-harness-eval && cd agent-harness-eval
cp eval.yaml.example eval.local.yaml
cp .env.example .env
# Edit .env with your API keys

uv sync                          # install dependencies
agent-harness-eval --config eval.local.yaml run
```

## Features

- **Out-of-the-box harness benchmarking** — multiple agent frameworks are already adapted behind one CLI, so you can run a full evaluation with the same config, the same tasks, and the same model instead of wiring each harness separately
- **Controlled comparisons that isolate harness effects** — each harness runs under matched API conditions and task inputs, making it easier to attribute differences to agent loop design, tool strategy, and orchestration behavior rather than model drift
- **Richer evaluation than pass/fail alone** — hard outcome checks, trajectory analysis, judge scoring, cost, latency, and token usage are combined so you can study both task success and research-relevant tradeoffs
- **Inspectability and reproducibility by default** — every run produces machine-readable results, canonical traces, per-run request snapshots, and reports that support failure analysis, reruns, and post-hoc comparison

## Setup

### Prerequisites

- **Python 3.12+**
- **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js 22+** — required for npm-based harnesses (OpenClaw, Claude Code, Codex)
- **Docker** (optional) — for stronger OS isolation

Local development remains pinned to Python 3.12 via `.python-version`. Python 3.14 is also a verified supported version and runs in CI.

### Install

```bash
uv sync                # install the eval framework itself
```

This only installs the evaluation framework and its Python dependencies. **Harness CLIs are managed separately depending on your executor mode:**

- **`executor: docker`** (recommended) — harness CLIs are baked into Docker images. The first run auto-builds the image if missing. No local harness installation needed.
- **`executor: host`** — harness CLIs are installed locally under `.harnesses/` per harness on first run. Requires Node.js 22+ for npm-based harnesses (OpenClaw, Claude Code, Codex).

### Verified Harness Versions

| Harness | Version | Host | Docker |
|---------|---------|:----:|:------:|
| OpenClaw | 2026.4.5 | ✅ | ✅ |
| Claude Code | 2.1.92 | ✅ | ✅ |
| Nanobot | 0.1.5 | ✅ | ✅ |
| ZeroClaw | 0.6.8 | ✅ | ✅ |
| Hermes Agent | 0.8.0 | ✅ | ✅ |
| Codex | 0.118.0 | ✅ | ✅ |

### Configure

#### API Relay Configuration

The framework currently supports exactly three provider wire formats:

| `api_format` | Expected endpoint shape | Typical use |
|--------------|-------------------------|-------------|
| `anthropic` | `/v1/messages` | Anthropic-native APIs and Anthropic-compatible relays |
| `openai-chat-completions` | `/v1/chat/completions` | OpenAI Chat Completions-compatible relays |
| `openai-responses` | `/v1/responses` | OpenAI Responses-compatible relays; required by Codex |

Choose the `api_format` that matches the relay's actual HTTP API, not the upstream model family. For example, the same Claude model may need `anthropic` for `claude-code` but `openai-responses` for `codex` if your relay exposes different protocols.

Only a few harnesses have strict protocol requirements you need to remember:

- `claude-code` requires `anthropic`
- `codex` requires `openai-responses`

`base_url` must be the provider root URL, not a concrete endpoint. The framework appends the protocol-specific path automatically.

**`.env`** — API keys only:

```env
# Direct Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx

# Or via relay
EVAL_PROVIDER_MYRELAY_API_KEY=sk-xxx
```

**`eval.yaml`** — evaluation settings and non-secret provider config:

```yaml
providers:
  myrelay:
    base_url: https://relay.example.com/
    api_format: anthropic

provider: myrelay
model: claude-sonnet-4-6
judge_model:
  provider: myrelay
  model: claude-sonnet-4-6
executor: host
concurrency: 1
runs: 1
timeout: 600

harnesses:
  openclaw:
    version: "2026.4.5"
  claude_code:
    version: "2.1.92"
  codex:
    version: "0.118.0"
    provider: relay_responses   # per-harness provider override
  nanobot:
    version: "0.1.5"
    # docker_image: "agent-harness-eval-nanobot:0.1.5"
```

`provider` and `model` are separate top-level fields. Each harness can override the default provider via `harnesses.<name>.provider` — useful when different harnesses require different wire protocols (e.g. claude-code needs `anthropic`, codex needs `openai-responses`).

If you need multiple protocols from the same relay, define multiple provider entries:

```yaml
providers:
  relay_anthropic:
    base_url: https://relay.example.com/anthropic
    api_format: anthropic
  relay_responses:
    base_url: https://relay.example.com/openai
    api_format: openai-responses

provider: relay_anthropic
model: claude-sonnet-4-6

harnesses:
  claude_code:
    version: "2.1.92"
  codex:
    version: "0.118.0"
    provider: relay_responses
```

For local development, prefer keeping your real config in a private file such as `eval.local.yaml` and passing it explicitly with `--config` instead of committing or retaining a populated repo-root `eval.yaml`.

### Docker Execution

Docker execution is controlled by `executor:` in `eval.yaml`:

```yaml
executor: docker   # force Docker
```

`executor` must be set explicitly. Use `executor: host` for direct host execution and `executor: docker` for containerized runs.

For repo-managed harness images (`openclaw`, `claude-code`, `codex`, `nanobot`, `hermes`, `zeroclaw`), the first `run` will automatically build the selected image if it is missing. The image tag is derived from the harness version in `eval.yaml`, for example:

```yaml
harnesses:
  codex:
    version: "0.118.0"
```

That resolves to a managed image tag like `agent-harness-eval-codex:0.118.0`.

If you want to use your own image instead of the repo-managed one, set `harnesses.<name>.docker_image`:

```yaml
executor: docker

harnesses:
  nanobot:
    version: "0.1.5"
    docker_image: "ghcr.io/acme/nanobot-eval:0.1.5"
```

Custom images must include the harness CLI plus verifier-side shell runtime: `bash`. Project-specific test dependencies such as `python`, `node`, or `pytest` remain the task's responsibility. Preflight validates the `bash` requirement before the first run.

Minimal Docker evaluation flow:

```bash
# 1. Configure eval.local.yaml with executor: docker
# 2. Ensure Docker is running
# 3. Start an evaluation
agent-harness-eval --config eval.local.yaml run
```

Optional Docker-specific environment flags:

```env
# Force one image for every harness (advanced / usually unnecessary)
EVAL_DOCKER_IMAGE=ghcr.io/acme/eval-image:latest

# When a task disables internet, use Docker's --network none instead of only env guards
EVAL_DOCKER_NETWORK_NONE=1
```

Use the host executor when you want faster local iteration and Docker when you want stronger isolation or more reproducible harness environments.

## Usage

```bash
# Full evaluation (all harnesses from your local config)
agent-harness-eval --config eval.local.yaml run

# Full evaluation in Docker
agent-harness-eval --config eval.local.yaml run   # with executor: docker in eval.local.yaml

# Single task, single harness
agent-harness-eval --config eval.local.yaml run --task security.01 --harness claude-code

# Specific categories
agent-harness-eval --config eval.local.yaml run --category reasoning,memory

# Different model
agent-harness-eval --config eval.local.yaml run --model openai:gpt-5.4

# Multi-model matrix
agent-harness-eval --config eval.local.yaml run --model anthropic:claude-sonnet-4-6,openai:gpt-5.4

# Regenerate reports from existing results
agent-harness-eval --config eval.local.yaml report --input results/<run-dir>/data/runs.jsonl
```

### CLI Options

```
agent-harness-eval [global options] run [options]

  --config, -c <path>      Path to eval config file (default: ./eval.yaml)

  --model <spec>            Model spec(s), comma-separated (provider:model format)
  --harness <list>          Comma-separated harnesses
  --runs <n>                Runs per task per harness (default: 1)
  --concurrency <n>         Max concurrent task groups (default: 1)
  --executor host|docker    Executor backend (overrides eval.yaml)
  --judge-model <spec>      Model for judge graders (provider:model)
  --tasks-dir <dir>         Tasks directory (default: ./tasks)
  --category <list>         Filter by category
  --task, --task-id <list>  Filter by task ID(s), comma-separated
  --output <dir>            Output directory
  --timeout <sec>           Per-task timeout in seconds
  --reinstall               Reinstall selected harnesses before running
```

### Execution Model

Same-task harnesses **always run in parallel**. `--concurrency` controls parallel task groups:

```
concurrency=1:  task1: [openclaw|claude-code|codex]  ->  task2: [openclaw|claude-code|codex]
concurrency=3:  task1: [a|o|c]  task2: [a|o|c]  task3: [a|o|c]
```

Each run also gets a private filesystem layout under a temp root:

```
<run-root>/
├── input/workspace/     # immutable task seed files
├── runtime/workspace/   # mutable working copy exposed to the harness
├── runtime/state/       # harness-private home/config/session state
└── output/              # reserved for exported artifacts and debug output
```

This keeps task inputs, mutable agent state, and collected outputs separate so
the same contract can later be executed locally or on cluster workers.

`setup.prepare_commands` also follows this run contract: it executes inside the
selected sandbox, with write access to the run workspace/state and a per-run
`HOME`/`XDG_*` rooted under `runtime/state/`. This keeps setup-time caches and
tooling side-effects out of shared host paths.

## Output

```
results/run__<timestamp>__<model>__<harness>__<tasks>/
├── manifest.json                   # run metadata + file index
├── data/
│   ├── preflight.json              # machine-readable preflight diagnostics
│   ├── runs.jsonl                  # all RunResults (crash recovery)
│   └── metrics.json                # machine-readable metrics
├── reports/
│   ├── summary.md                  # scorecard + category/failure/judge analysis
│   └── case-review.md              # detailed per-task summaries
└── traces/
    ├── index.jsonl                 # run_id -> task/harness/model/status index
    └── <run_id>/
        ├── request.json            # prompt + task snapshot for this run
        ├── trace.json              # canonical tool calls + messages
        └── raw/                    # harness-native debug artifacts when available
```

### Sample Report

```md
# Evaluation Summary

- Models: `anthropic:claude-sonnet-4-6`
- Judge model: `anthropic:claude-sonnet-4-6`
- Executor: `docker`
- Runs per task: `3`

| Harness | pass@1 | pass@3 | Avg Quality | Mean Time | Mean Tokens | Median Cost |
|---------|--------|--------|-------------|-----------|-------------|-------------|
| Codex | 82.0% | 94.0% | 0.91 | 18.4s | 14.2k | $0.28 |
| OpenClaw | 79.0% | 92.0% | 0.89 | 16.7s | 12.8k | $0.24 |
| Claude Code | 75.0% | 90.0% | 0.87 | 14.9s | 11.3k | $0.21 |

## Category Breakdown

| Category | Best Harness | Best pass@1 |
|----------|--------------|-------------|
| `coding` | Codex | 88.0% |
| `security` | OpenClaw | 84.0% |
| `memory` | Claude Code | 81.0% |

## Failure Taxonomy

- `provider_error`: 2 runs
- `sandbox_permission_error`: 1 run
- `adapter_output_error`: 1 run
```

## Task Suite

Tasks are organized by harness capability:

| Category | Focus |
|----------|-------|
| `reasoning` | Multi-file analysis, synthesis, planning |
| `implementation` | Large artifact generation, coordinated edits |
| `coding` | Code changes with preserved behavior |
| `security` | Boundary-respecting operations |
| `skills` | Tool composition workflows |
| `memory` | Context retention, native session recall |
| `robustness` | Red herrings, cascading failures |

### Research & Extension Guides

For researchers extending the task suite or developers integrating new runtimes:

- Add a task: [docs/adding-a-task.md](docs/adding-a-task.md)
- Add a harness adapter: [docs/adding-a-harness.md](docs/adding-a-harness.md)
- Add a custom executor: [docs/adding-an-executor.md](docs/adding-an-executor.md)

## License

MIT
