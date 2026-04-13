#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERSION="${1:-${HERMES_VERSION:-}}"
if [[ -z "${VERSION}" ]]; then echo "VERSION required: $0 <version> or HERMES_VERSION=..." >&2; exit 1; fi
IMAGE_TAG="${2:-agent-harness-eval-hermes:${VERSION}}"
HERMES_TAG_VALUE="${HERMES_TAG:-v${VERSION}}"

docker build --tag "${IMAGE_TAG}" --build-arg "HERMES_TAG=${HERMES_TAG_VALUE}" "${PROJECT_ROOT}/docker/hermes"
