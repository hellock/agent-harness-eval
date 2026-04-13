#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERSION="${1:-${ZEROCLAW_VERSION:-}}"
if [[ -z "${VERSION}" ]]; then echo "VERSION required: $0 <version> or ZEROCLAW_VERSION=..." >&2; exit 1; fi
IMAGE_TAG="${2:-agent-harness-eval-zeroclaw:${VERSION}}"
SOURCE_BINARY="${ZEROCLAW_BINARY:-${PROJECT_ROOT}/.harnesses/zeroclaw/bin/zeroclaw}"
if [[ ! -f "${SOURCE_BINARY}" ]]; then echo "ZeroClaw binary not found: ${SOURCE_BINARY}" >&2; exit 1; fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
cp "${PROJECT_ROOT}/docker/zeroclaw/Dockerfile" "${TMP_DIR}/Dockerfile"
cp "${SOURCE_BINARY}" "${TMP_DIR}/zeroclaw"

docker build --tag "${IMAGE_TAG}" "${TMP_DIR}"
