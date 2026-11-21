#!/usr/bin/env bash
# Forward arguments to the MeshCore sender; send immediately, then every minute.
set -euo pipefail
trap 'printf "\nStopped.\n" >&2; exit 130' INT
trap 'exit 143' TERM

if [[ $# -eq 0 || ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: ./send_every_60s.sh CONNECTION --region REGION MESSAGE [OPTIONS]

TCP example:
  ./send_every_60s.sh --tcp 192.168.0.1:5051 --region '#au-nsw-ntl' --text 'Hello mesh!' --offline

USB example:
  ./send_every_60s.sh --serial /dev/ttyUSB0 --region '#au-nsw-ntl' --text 'Hello mesh!' --offline

All arguments are passed to the MeshCore send command, including --file,
--profile, and --dry-run. Sends immediately, then every 60 seconds.
Failed sends are logged and retried at the next interval. Stop with Ctrl+C.
HELP
    exit 0
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$project_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    printf 'Missing virtual environment: %s\nFollow the setup in MESHCORE.md.\n' "$python_bin" >&2
    exit 1
fi
cd -- "$project_dir"

# Validate arguments before starting the repeating loop; this does not connect.
"$python_bin" -c 'import sys; from llm_compression.meshcore_app import parser; parser().parse_args(["send", *sys.argv[1:]])' "$@"

trap 'printf "\nStopped.\n" >&2; exit 130' INT
trap 'printf "\nStopped.\n" >&2; exit 143' TERM

while true; do
    next_send=$((SECONDS + 60))
    if "$python_bin" -m llm_compression.meshcore_app send "$@"; then
        :
    else
        status=$?
        printf 'Send failed (exit %s); next attempt at the next interval.\n' "$status" >&2
    fi
    # Include model loading and sending time in the interval. Never overlap sends.
    remaining=$((next_send - SECONDS))
    if ((remaining > 0)); then
        sleep "$remaining"
    fi
done
