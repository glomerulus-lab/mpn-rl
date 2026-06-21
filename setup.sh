#!/bin/bash
# Sets up a virtual environment and installs dependencies.
# Uses UV_CACHE_DIR on the same filesystem as the repo to enable hardlinking,
# which avoids slow cross-filesystem copies on NFS clusters.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_CACHE_DIR="${SCRIPT_DIR}/.uv-cache"

echo "Installing dependencies (cache: ${UV_CACHE_DIR})..."
# uv sync honors [tool.uv.sources] (the cu126 torch pin) and writes uv.lock;
# `uv pip install` ignores sources and would pull the default +cu130 build.
UV_CACHE_DIR="${UV_CACHE_DIR}" uv sync --project "${SCRIPT_DIR}"

echo ""
echo "Done. Activate with:"
echo "  source .venv/bin/activate"
