FROM agent-harness-eval-base:latest

ARG NANOBOT_VERSION
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN uv pip install --system --no-cache "nanobot-ai==${NANOBOT_VERSION}"

# Patch nanobot 0.1.5: add usage metadata support + missing tool_hints stub
COPY patch_usage.py /tmp/patch_usage.py
RUN python3 /tmp/patch_usage.py /usr/local/lib/python3.11/dist-packages && rm /tmp/patch_usage.py
