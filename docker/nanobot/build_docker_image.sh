#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERSION="${1:-${NANOBOT_VERSION:-}}"
if [[ -z "${VERSION}" ]]; then echo "VERSION required: $0 <version> or NANOBOT_VERSION=..." >&2; exit 1; fi
IMAGE_TAG="${2:-agent-harness-eval-nanobot:${VERSION}}"

docker build --tag "${IMAGE_TAG}" --build-arg "NANOBOT_VERSION=${VERSION}" "${PROJECT_ROOT}/docker/nanobot"
