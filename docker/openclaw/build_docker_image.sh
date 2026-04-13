#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERSION="${1:-${OPENCLAW_VERSION:-}}"
if [[ -z "${VERSION}" ]]; then echo "VERSION required: $0 <version> or OPENCLAW_VERSION=..." >&2; exit 1; fi
IMAGE_TAG="${2:-agent-harness-eval-openclaw:${VERSION}}"

docker build --tag "${IMAGE_TAG}" --build-arg "OPENCLAW_VERSION=${VERSION}" "${PROJECT_ROOT}/docker/openclaw"
