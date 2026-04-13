# Unified base image for agent-harness-eval.
#
# All harness docker images extend this so every task's prepare_commands
# and grader commands run in a consistent environment.
#
# Build:
#   docker build -t agent-harness-eval-base:latest docker/base/
#
# Includes:
#   - Node.js 22 + npm    (coding/robustness tasks)
#   - Python 3 (system)   (skills tasks, fallback)
#   - uv                  (per-task Python version via .python-version)
#   - git, curl, jq, sqlite3, ripgrep

FROM node:22-slim

ENV DEBIAN_FRONTEND=noninteractive

# Single layer: install + clean in one RUN to minimize size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    jq \
    sqlite3 \
    ca-certificates \
    ripgrep \
  && rm -rf /var/lib/apt/lists/* \
  && ln -sf python3 /usr/bin/python

# uv — manages per-task Python versions via .python-version
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1

# Smoke test
RUN node --version && npm --version && python3 --version && git --version && uv --version
