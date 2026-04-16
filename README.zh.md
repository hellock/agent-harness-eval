[English](README.md) | **中文**

# Agent Harness Eval

一个用于客观评估和比较不同 AI Agent 系统（harness）的框架。它不测试模型本身，而是测试**每个 Agent 框架在使用相同模型时的编排能力**，包括工具选择、多步规划、错误恢复和成本效率。

给定相同的模型和相同的任务集，不同的 Agent 系统会因为其 Agent 循环、系统提示词、工具组织方式和执行策略的差异而产生不同的结果。本框架从多个维度衡量这些差异：任务完成率、输出质量、延迟、Token 用量和成本。

## 快速开始

```bash
git clone https://github.com/hellock/agent-harness-eval agent-harness-eval && cd agent-harness-eval
cp eval.yaml.example eval.local.yaml
cp .env.example .env
# 在 .env 中填入你的 API 密钥

uv sync                          # 安装依赖
agent-harness-eval run --config eval.local.yaml
```

## 特性

- **开箱即用的 harness 评测** —— 多个 Agent 框架已经完成适配，并统一在一个 CLI 后面；用同一份配置就能一键跑完整评测，不需要逐个接线
- **能真正隔离 harness 差异的对比** —— 每个 harness 都在相同的任务输入和一致的 API 条件下运行，更容易把结果差异归因到 agent loop、工具策略和编排方式，而不是模型漂移
- **不只统计通过率，还能看研究上关心的取舍** —— 同时结合硬检查、轨迹分析、Judge 评分、成本、耗时和 Token，用来观察任务成功率之外的行为质量与效率权衡
- **默认可解释、可复现** —— 每次运行都会产出机器可读结果、标准化轨迹、单次请求快照和报告，方便做失败分析、重复实验和事后比较

## 安装

### 前置要求

- **Python 3.12+**
- **uv** —— `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js 22+** —— npm 类 harness（OpenClaw、Claude Code、Codex）需要
- **Docker**（可选）—— 用于更强的操作系统级隔离

本地开发默认仍通过 `.python-version` 固定在 Python 3.12。Python 3.14 也已经完成验证，并在 CI 中持续运行。

### 安装依赖

```bash
uv sync                # 安装评测框架本身
```

这一步只安装评测框架及其 Python 依赖。**Harness CLI 的管理方式取决于你选择的执行模式：**

- **`executor: docker`**（推荐）—— Harness CLI 已烘焙进 Docker 镜像。首次运行时若镜像不存在会自动构建，无需在本地安装任何 harness。
- **`executor: host`** —— Harness CLI 在首次运行时自动安装到各自的 `.harnesses/` 目录下。npm 类 harness（OpenClaw、Claude Code、Codex）需要 Node.js 22+。

### 已验证的 Harness 版本

| Harness | 版本 | Host | Docker |
|---------|------|:----:|:------:|
| OpenClaw | 2026.4.5 | ✅ | ✅ |
| Claude Code | 2.1.92 | ✅ | ✅ |
| Nanobot | 0.1.5 | ✅ | ✅ |
| ZeroClaw | 0.6.9 | ✅ | ✅ |
| Hermes Agent | 0.8.0 | ✅ | ✅ |
| Codex | 0.118.0 | ✅ | ✅ |

### 配置

#### API Relay 配置

框架当前只支持 3 种 provider 协议格式：

| `api_format` | 期望的 endpoint 形态 | 典型用途 |
|--------------|----------------------|----------|
| `anthropic` | `/v1/messages` | Anthropic 原生 API，以及兼容 Anthropic Messages 的 relay |
| `openai-chat-completions` | `/v1/chat/completions` | 兼容 OpenAI Chat Completions 的 relay |
| `openai-responses` | `/v1/responses` | 兼容 OpenAI Responses 的 relay；`codex` 必需 |

`api_format` 应该按照 relay 实际暴露的 HTTP 协议来选，而不是按底层模型家族来猜。比如同一个 Claude 模型，如果你的 relay 同时暴露 Anthropic 和 Responses 两种协议，那么 `claude-code` 可能要配 `anthropic`，而 `codex` 需要 `openai-responses`。

只有少数 harness 有需要特别记住的协议要求：

- `claude-code` 只接受 `anthropic`
- `codex` 只接受 `openai-responses`

`base_url` 必须填写 provider root URL，而不是具体 endpoint。协议对应的路径会由框架自动补上。

**`.env`** —— 仅放 API 密钥：

```env
# 直连 Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx

# 或通过中继
EVAL_PROVIDER_MYRELAY_API_KEY=sk-xxx
```

**`eval.yaml`** —— 评估设置和非敏感 Provider 配置：

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
    provider: relay_responses   # 按 harness 单独覆盖 provider
  nanobot:
    version: "0.1.5"
    # docker_image: "agent-harness-eval-nanobot:0.1.5"
```

`provider` 和 `model` 是两个独立的顶层字段。每个 harness 都可以通过 `harnesses.<name>.provider` 覆盖默认 provider；当不同 harness 需要不同协议时，这很有用（例如 claude-code 需要 `anthropic` 格式，codex 需要 `openai-responses` 格式）。

如果同一个 relay 同时提供多种协议，可以在配置里定义多个 provider 条目：

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

本地开发时，推荐把真实配置保存在私有文件里，例如 `eval.local.yaml`，并通过 `--config` 显式传入，而不是长期在仓库根目录保留一个带真实内容的 `eval.yaml`。

### Docker 执行链路

Docker 执行由 `eval.yaml` 里的 `executor:` 控制：

```yaml
executor: docker   # 强制使用 Docker
```

`executor` 必须显式指定。直接在本机运行就写 `executor: host`，需要容器隔离就写 `executor: docker`。

对于仓库自带管理镜像的 harness（`openclaw`、`claude-code`、`codex`、`nanobot`、`hermes`、`zeroclaw`），第一次 `run` 时如果缺少镜像，框架会自动构建所需镜像。这包括共享 base image 和所选 harness image。harness image 的 tag 仍按 `eval.yaml` 里的版本推导，例如：

```yaml
harnesses:
  codex:
    version: "0.118.0"
```

这会解析成类似 `agent-harness-eval-codex:0.118.0` 的 managed image tag。

如果你想使用自己的镜像，而不是仓库管理的镜像，可以设置 `harnesses.<name>.docker_image`：

```yaml
executor: docker

harnesses:
  nanobot:
    version: "0.1.5"
    docker_image: "ghcr.io/acme/nanobot-eval:0.1.5"
```

自定义镜像除了 harness CLI 本身，还必须包含 verifier 侧会用到的 shell 运行时：`bash`。像 `python`、`node`、`pytest` 这类项目级测试依赖，仍然由具体任务负责。框架会在 preflight 阶段先检查 `bash` 这一项。

最小 Docker 评测流程：

```bash
# 1. 在 eval.local.yaml 里设置 executor: docker
# 2. 确保 Docker 已启动
# 3. 启动评测
agent-harness-eval run --config eval.local.yaml
```

可选的 Docker 相关环境变量：

```env
# 为所有 harness 强制指定同一个镜像（高级用法，通常不需要）
EVAL_DOCKER_IMAGE=ghcr.io/acme/eval-image:latest

# 当任务禁用互联网时，使用 Docker 的 --network none，而不只是环境变量约束
EVAL_DOCKER_NETWORK_NONE=1
```

本地快速迭代更适合用 host executor；如果你更关注隔离性或环境可复现性，就用 Docker。

## 使用方法

```bash
# 完整评估（使用本地配置中的所有 harness）
agent-harness-eval run --config eval.local.yaml

# 使用 Docker 完整评估
agent-harness-eval run --config eval.local.yaml   # 前提是 eval.local.yaml 中设置了 executor: docker

# 单任务，单 harness
agent-harness-eval run --config eval.local.yaml --task security.01 --harness claude-code

# 指定类别
agent-harness-eval run --config eval.local.yaml --category reasoning,memory

# 不同模型
agent-harness-eval run --config eval.local.yaml --model openai:gpt-5.4

# 多模型矩阵
agent-harness-eval run --config eval.local.yaml --model anthropic:claude-sonnet-4-6,openai:gpt-5.4

# 从已有结果重新生成报告（默认不会重新跑 grader）
agent-harness-eval report --config eval.local.yaml --input results/<run-dir>/data/runs.jsonl

# 先重跑可重放的 grader，再重建报告
agent-harness-eval report --config eval.local.yaml --input results/<run-dir>/data/runs.jsonl --regrade

# 仅在 --regrade 时覆盖 judge model
agent-harness-eval report --config eval.local.yaml --input results/<run-dir>/data/runs.jsonl --regrade --judge-model openai:gpt-5.4
```

### CLI 选项

```
agent-harness-eval run [选项]

  --config, -c <path>      配置文件路径（默认：./eval.yaml）
  --model <spec>            模型规格，逗号分隔（provider:model 格式）
  --harness <list>          逗号分隔的 harness 列表
  --runs <n>                每个任务每个 harness 的运行次数（默认：1）
  --concurrency <n>         最大并发任务组数（默认：1）
  --executor host|docker    执行后端（覆盖 eval.yaml 中的设置）
  --judge-model <spec>      评分用模型（provider:model 格式）
  --tasks-dir <dir>         任务目录（默认：./tasks）
  --category <list>         按类别筛选
  --task, --task-id <list>  按任务 ID 筛选，逗号分隔
  --output <dir>            输出目录
  --timeout <sec>           每个任务的超时时间（秒）
  --reinstall               运行前重新安装选定的 harness
```

### 执行模型

同一任务的所有 harness **始终并行执行**。`--concurrency` 控制并行的任务组数：

```
concurrency=1:  任务1: [openclaw|claude-code|codex]  ->  任务2: [openclaw|claude-code|codex]
concurrency=3:  任务1: [a|o|c]  任务2: [a|o|c]  任务3: [a|o|c]
```

每个 run 还会在临时目录下获得一套私有文件布局：

```
<run-root>/
├── input/workspace/     # 只读任务初始文件
├── runtime/workspace/   # 暴露给 harness 的可写工作副本
├── runtime/state/       # harness 私有的 home/config/session 状态
└── output/              # 预留给导出产物和调试输出
```

这样任务输入、可变 agent 状态、结果输出会被明确分开，后续迁移到集群 worker
时可以复用同一套执行约定。

`setup.prepare_commands` 现在也遵循这套 run 契约：它会在当前选中的 sandbox
里执行，对当前 run 的 workspace/state 保持可写，并把 `HOME`/`XDG_*` 指到
`runtime/state/` 下的私有目录，避免 setup 阶段把缓存和副作用写到共享宿主机路径。

## 输出

```
results/run__<时间戳>__<模型>__<harness>__<任务>/
├── manifest.json                   # 本次运行元信息 + 文件索引
├── data/
│   ├── preflight.json              # 机器可读 preflight 诊断
│   ├── runs.jsonl                  # 所有 RunResult（崩溃恢复用）
│   └── metrics.json                # 机器可读指标
├── reports/
│   ├── summary.md                  # 评分卡 + 分类/失败/Judge 分析
│   └── case-review.md              # 逐任务详细总结
└── traces/
    ├── index.jsonl                 # run_id -> task/harness/model/status 索引
    └── <run_id>/
        ├── request.json            # 本次运行的 prompt + task 快照
        ├── trace.json              # 标准化工具调用 + 消息轨迹
        └── raw/                    # 可用时保留 harness 原生调试产物
```

### 示例报告

```md
# 评测摘要

- 模型：`anthropic:claude-sonnet-4-6`
- Judge 模型：`anthropic:claude-sonnet-4-6`
- Executor：`docker`
- 每任务运行次数：`3`

| Harness | 通过率 | 平均通过率 | 平均质量 | 平均耗时 | 平均 Token | 平均成本 | 无缓存平均成本 |
|---------|--------|------------|----------|----------|------------|----------|----------------|
| Codex | 82.0% | 94.0% | 0.91 | 18.4s | 14.2k | $0.18 | $0.28 |
| OpenClaw | 79.0% | 92.0% | 0.89 | 16.7s | 12.8k | $0.16 | $0.24 |
| Claude Code | 75.0% | 90.0% | 0.87 | 14.9s | 11.3k | $0.15 | $0.21 |

## 分类表现

| 类别 | 最优 Harness | 最优 pass@1 |
|------|--------------|-------------|
| `coding` | Codex | 88.0% |
| `security` | OpenClaw | 84.0% |
| `memory` | Claude Code | 81.0% |

## 失败归因

- `provider_error`: 2 次
- `sandbox_permission_error`: 1 次
- `adapter_output_error`: 1 次
```

## 任务集

任务按 harness 能力维度组织：

| 类别 | 聚焦点 |
|------|--------|
| `reasoning` | 多文件分析、综合、规划 |
| `implementation` | 大型产物生成、协调编辑 |
| `coding` | 保持行为不变的代码修改 |
| `security` | 遵守边界约束的操作 |
| `skills` | 工具组合工作流 |
| `memory` | 上下文保持、原生会话回忆 |
| `robustness` | 干扰项、级联故障 |

### 研究与扩展指南

如果你是要扩展任务集的研究者，或要接入新 runtime 的开发者，可以从这里开始：

- 添加任务：[docs/adding-a-task.md](docs/adding-a-task.md)
- 添加 harness 适配器：[docs/adding-a-harness.md](docs/adding-a-harness.md)
- 添加自定义执行器：[docs/adding-an-executor.md](docs/adding-an-executor.md)

## 许可证

MIT
