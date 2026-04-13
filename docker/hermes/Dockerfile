FROM agent-harness-eval-base:latest

ARG HERMES_TAG
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN uv pip install --system --no-cache "git+https://github.com/NousResearch/hermes-agent.git@${HERMES_TAG}"
