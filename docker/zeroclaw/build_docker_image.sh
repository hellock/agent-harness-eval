#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERSION="${1:-${ZEROCLAW_VERSION:-}}"
if [[ -z "${VERSION}" ]]; then echo "VERSION required: $0 <version> or ZEROCLAW_VERSION=..." >&2; exit 1; fi
IMAGE_TAG="${2:-agent-harness-eval-zeroclaw:${VERSION}}"

docker build \
  --build-arg "ZEROCLAW_VERSION=${VERSION}" \
  --tag "${IMAGE_TAG}" \
  --file "${PROJECT_ROOT}/docker/zeroclaw/Dockerfile" \
  "${PROJECT_ROOT}"
