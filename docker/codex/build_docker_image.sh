#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERSION="${1:-${CODEX_VERSION:-}}"
if [[ -z "${VERSION}" ]]; then echo "VERSION required: $0 <version> or CODEX_VERSION=..." >&2; exit 1; fi
IMAGE_TAG="${2:-agent-harness-eval-codex:${VERSION}}"

docker build --tag "${IMAGE_TAG}" --build-arg "CODEX_VERSION=${VERSION}" "${PROJECT_ROOT}/docker/codex"
