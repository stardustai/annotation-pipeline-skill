#!/usr/bin/env bash
# Start the annotation-pipeline dashboard + runtime worker as detached
# background processes. Idempotent: re-running first stops anything
# already up.
#
# Usage:
#   ./start.sh              # start both
#   ./start.sh --stop       # stop both
#   ./start.sh --status     # show running PIDs + recent log
#
# Defaults: workspace=<repo>/projects, port=8509, runtime project = first
# project under projects/ that has a .annotation-pipeline/ dir.
# Override:
#   SERVE_PORT=8510 ./start.sh
#   SERVE_HOST=127.0.0.1 ./start.sh
#   SERVE_WORKSPACE=/path/to/projects ./start.sh
#   RUNTIME_PROJECT_ROOT=projects/foo ./start.sh
#   API_RELOAD=0 ./start.sh        # disable --reload
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="${REPO_ROOT}/.venv/bin"
HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${SERVE_PORT:-8509}"
WORKSPACE="${SERVE_WORKSPACE:-${REPO_ROOT}/projects}"
SERVE_LOG="${WORKSPACE}/serve.log"
RUNTIME_LOG="${WORKSPACE}/runtime.log"
API_RELOAD="${API_RELOAD:-1}"
AUTH_DIR="${HOME}/.agents/auth"

# Auto-detect runtime project root if not pinned. Picks the first
# directory under WORKSPACE that has a .annotation-pipeline/ subdir.
detect_runtime_project_root() {
    if [[ -n "${RUNTIME_PROJECT_ROOT:-}" ]]; then
        printf '%s' "${RUNTIME_PROJECT_ROOT}"
        return 0
    fi
    for d in "${WORKSPACE}"/*/.annotation-pipeline; do
        [[ -d "$d" ]] || continue
        printf '%s' "$(dirname "$d")"
        return 0
    done
    return 1
}

stop_processes() {
    local stopped_any=0
    if pgrep -f "annotation-pipeline serve" >/dev/null; then
        pkill -TERM -f "annotation-pipeline serve" || true
        stopped_any=1
        echo "→ stopped: annotation-pipeline serve"
    fi
    if pgrep -f "annotation-pipeline runtime run" >/dev/null; then
        pkill -TERM -f "annotation-pipeline runtime run" || true
        stopped_any=1
        echo "→ stopped: annotation-pipeline runtime run"
    fi
    if [[ $stopped_any -eq 0 ]]; then
        echo "(nothing to stop)"
    fi
}

show_status() {
    local found_any=0
    while read -r line; do
        [[ -z "$line" ]] && continue
        found_any=1
        echo "  $line"
    done < <(pgrep -af "annotation-pipeline (serve|runtime)" 2>/dev/null || true)
    if [[ $found_any -eq 0 ]]; then
        echo "  (no annotation-pipeline processes running)"
        return 0
    fi
    if curl -s --max-time 2 -o /dev/null -w "  API HTTP: %{http_code}\n" "http://127.0.0.1:${PORT}/api/projects"; then :; fi
    echo
    echo "Recent serve log (${SERVE_LOG}):"
    [[ -f "$SERVE_LOG" ]] && tail -5 "$SERVE_LOG" 2>/dev/null | sed 's/^/  /' || echo "  (no log yet)"
    echo
    echo "Recent runtime log (${RUNTIME_LOG}):"
    [[ -f "$RUNTIME_LOG" ]] && tail -5 "$RUNTIME_LOG" 2>/dev/null | sed 's/^/  /' || echo "  (no log yet)"
}

case "${1:-}" in
    --stop)
        stop_processes
        exit 0
        ;;
    --status)
        show_status
        exit 0
        ;;
esac

# Source API keys (silent if files don't exist).
set -a
for f in deepseek.env glm.env minimax.env; do
    [[ -f "${AUTH_DIR}/${f}" ]] && source "${AUTH_DIR}/${f}"
done
set +a

# Idempotent start: stop anything previous first so we don't end up with
# two of either process competing for the port / SQLite.
stop_processes
sleep 1

cd "${REPO_ROOT}"
mkdir -p "${WORKSPACE}"

# Start API server.
serve_args=(serve --workspace "${WORKSPACE}" --host "${HOST}" --port "${PORT}")
[[ "${API_RELOAD}" == "1" ]] && serve_args+=(--reload)
nohup "${VENV_BIN}/annotation-pipeline" "${serve_args[@]}" \
    > "${SERVE_LOG}" 2>&1 < /dev/null &
SERVE_PID=$!
disown $SERVE_PID

# Start runtime worker against the auto-detected (or pinned) project.
runtime_project=""
if runtime_project="$(detect_runtime_project_root)"; then
    nohup "${VENV_BIN}/annotation-pipeline" runtime run --project-root "${runtime_project}" \
        > "${RUNTIME_LOG}" 2>&1 < /dev/null &
    RUNTIME_PID=$!
    disown $RUNTIME_PID
else
    RUNTIME_PID=""
    echo "⚠ no .annotation-pipeline/ project found under ${WORKSPACE}; runtime worker NOT started"
fi

# Wait briefly + verify the API came up.
for _ in 1 2 3 4 5; do
    sleep 1
    if curl -s --max-time 2 -o /dev/null "http://127.0.0.1:${PORT}/api/projects"; then
        break
    fi
done

echo
if curl -s --max-time 2 -o /dev/null -w "API serve  → http://${HOST}:${PORT}  (PID ${SERVE_PID}, HTTP %{http_code})\n" "http://127.0.0.1:${PORT}/api/projects"; then :; fi
if [[ -n "${RUNTIME_PID}" ]]; then
    if kill -0 "${RUNTIME_PID}" 2>/dev/null; then
        echo "Runtime    → ${runtime_project}  (PID ${RUNTIME_PID})"
    else
        echo "⚠ runtime exited; tail ${RUNTIME_LOG}"
    fi
fi
echo
echo "Logs:"
echo "  serve:   ${SERVE_LOG}"
echo "  runtime: ${RUNTIME_LOG}"
echo
echo "Stop:    ./start.sh --stop"
echo "Status:  ./start.sh --status"
