#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${1:-agent-harness-eval-base:latest}"

docker build --tag "${IMAGE_TAG}" "${SCRIPT_DIR}"
