#!/usr/bin/env bash
# Start the annotation-pipeline dashboard on stardust-gpu4:8509
#
# Sources API keys from ~/.agents/auth/{deepseek,glm,minimax}.env so the
# subagent runtime that backs the dashboard inherits the credentials.
#
# Usage:
#   scripts/serve_dashboard.sh            # foreground (Ctrl-C to stop)
#   scripts/serve_dashboard.sh --bg       # background via nohup
#   scripts/serve_dashboard.sh --stop     # stop any running serve
#
# Defaults: host=0.0.0.0, port=8509, workspace=<repo>/projects
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
VENV_BIN="${VENV_BIN:-/home/derek/miniforge3/envs/annotation-pipeline-skill/bin}"
HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${SERVE_PORT:-8509}"
WORKSPACE="${SERVE_WORKSPACE:-${REPO_ROOT}/projects}"
LOG_FILE="${WORKSPACE}/serve.log"
AUTH_DIR="${HOME}/.agents/auth"

case "${1:-}" in
    --stop)
        pkill -f "annotation-pipeline serve" && echo "stopped" || echo "(no serve running)"
        exit 0
        ;;
esac

# Source API keys (silent if files don't exist).
set -a
for f in deepseek.env glm.env minimax.env; do
    [[ -f "${AUTH_DIR}/${f}" ]] && source "${AUTH_DIR}/${f}"
done
set +a

cd "${REPO_ROOT}"

if [[ "${1:-}" == "--bg" ]]; then
    nohup "${VENV_BIN}/annotation-pipeline" serve \
        --workspace "${WORKSPACE}" \
        --host "${HOST}" \
        --port "${PORT}" \
        > "${LOG_FILE}" 2>&1 < /dev/null &
    disown
    sleep 1
    if pgrep -f "annotation-pipeline serve" > /dev/null; then
        echo "✓ serving on http://${HOST}:${PORT}  (log: ${LOG_FILE})"
    else
        echo "✗ failed to start; check ${LOG_FILE}" >&2
        exit 1
    fi
else
    exec "${VENV_BIN}/annotation-pipeline" serve \
        --workspace "${WORKSPACE}" \
        --host "${HOST}" \
        --port "${PORT}"
fi
