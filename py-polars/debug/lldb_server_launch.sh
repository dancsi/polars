#!/usr/bin/env bash
#
# lldb_server_launch.sh
#
# Launch a Python script under RustRover's bundled `lldb-server` in gdbserver
# mode, so RustRover's "Remote Debug" run configuration can connect and hit
# Rust breakpoints in the native polars extension (`polars` .so) called from
# Python.
#
# Why this exists: RustRover has no "Native Application" run config, so it
# cannot launch an arbitrary binary (python) under the debugger. Its
# "Remote Debug" config can only *connect* to an already-running gdbserver.
# This script is the missing "launch" half: it starts lldb-server, which owns
# the python process, and waits until the port is accepting connections.
#
# Intended use: the "Before launch -> Run External tool" step of a RustRover
# "Remote Debug" configuration. In that mode it must background the server and
# return once the port is listening (the default). Run it with --foreground to
# use it standalone in a terminal instead.
#
# Usage:
#   lldb_server_launch.sh [-f|--foreground] [SCRIPT.py] [ARGS...]
#
#   SCRIPT.py   target script to debug (default: $DEBUG_SCRIPT, else repro.py
#               next to this script). ARGS are forwarded to it.
#
# Environment overrides (all optional):
#   DEBUG_HOST    bind address                (default: 127.0.0.1)
#   DEBUG_PORT    listen port                 (default: 1234)
#   LLDB_SERVER   path to lldb-server/        (default: RustRover bundle, falling
#                 debugserver                  back to one on PATH)
#   VENV_PYTHON   python interpreter          (default: <repo>/.venv/bin/python)
#   DEBUG_SCRIPT  default target script       (default: <this dir>/repro.py)
#
# Supports Linux (via RustRover's bundled `lldb-server`) and macOS (via Apple's
# `debugserver`, which speaks the same gdb-remote protocol directly).
set -euo pipefail

log()  { printf '%s\n' "==> $*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- parse args ------------------------------------------------------------
FOREGROUND=0
if [[ "${1:-}" == "-f" || "${1:-}" == "--foreground" ]]; then
  FOREGROUND=1
  shift
fi

# --- resolve locations -----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

HOST="${DEBUG_HOST:-127.0.0.1}"
PORT="${DEBUG_PORT:-1234}"

PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || die "python interpreter not found or not executable: $PYTHON
       set VENV_PYTHON, or create the virtualenv at $REPO_ROOT/.venv"

TARGET="${1:-${DEBUG_SCRIPT:-$SCRIPT_DIR/repro.py}}"
[[ $# -gt 0 ]] && shift || true   # remaining "$@" is forwarded to the script
[[ -f "$TARGET" ]] || die "target script not found: $TARGET"
# Guard against the common footgun of passing a non-Python file (e.g. via the
# External Tool '\$FilePath\$' macro while this launcher, or any other file, is
# the active editor tab): running it as `python <that file>` fails cryptically.
case "$TARGET" in
  *.py) ;;
  *) die "target is not a Python (.py) script: $TARGET
       if the External Tool argument is \$FilePath\$, make sure a .py file is the
       active editor tab — not this launcher. Or hardcode the script in Arguments." ;;
esac

# --- platform detection -----------------------------------------------------
case "$(uname -s)" in
  Linux)  PLATFORM=linux ;;
  Darwin) PLATFORM=mac ;;
  *) die "unsupported platform: $(uname -s) (this script supports Linux and macOS)" ;;
esac

case "$(uname -m)" in
  x86_64|amd64)  ARCH=x64 ;;
  arm64|aarch64) ARCH=aarch64 ;;
  *) die "unsupported architecture: $(uname -m)" ;;
esac

# lldb-server/debugserver: prefer an explicit override, then the RustRover
# bundle, then anything on PATH. Linux ships `lldb-server` (run via its
# `gdbserver` subcommand); macOS ships Apple's `debugserver`, which speaks the
# same gdb-remote protocol directly, so it takes no subcommand.
find_bundled_server() {
  if [[ "$PLATFORM" == linux ]]; then
    local p="$HOME/.local/share/JetBrains/Toolbox/apps/rustrover/bin/lldb/linux/$ARCH/bin/lldb-server"
    [[ -x "$p" ]] && { printf '%s\n' "$p"; return 0; }
    return 1
  fi

  local rel="Contents/bin/lldb/mac/$ARCH/LLDB.framework/Resources/debugserver"
  local c
  for c in \
    "$HOME/Applications/RustRover.app/$rel" \
    "/Applications/RustRover.app/$rel" \
    "/Library/Developer/CommandLineTools/Library/PrivateFrameworks/LLDB.framework/Versions/A/Resources/debugserver" \
    "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver"
  do
    [[ -x "$c" ]] && { printf '%s\n' "$c"; return 0; }
  done

  # JetBrains Toolbox installs RustRover.app under a version-numbered path.
  local toolbox_hit
  toolbox_hit="$(find "$HOME/Library/Application Support/JetBrains/Toolbox/apps" \
    -ipath "*RustRover*/$rel" -print -quit 2>/dev/null || true)"
  [[ -n "$toolbox_hit" ]] && { printf '%s\n' "$toolbox_hit"; return 0; }
  return 1
}

LLDB_SERVER="${LLDB_SERVER:-}"
if [[ -z "$LLDB_SERVER" ]]; then
  if bundled="$(find_bundled_server)"; then
    LLDB_SERVER="$bundled"
  elif [[ "$PLATFORM" == linux ]] && command -v lldb-server >/dev/null 2>&1; then
    LLDB_SERVER="$(command -v lldb-server)"
  elif [[ "$PLATFORM" == mac ]] && command -v debugserver >/dev/null 2>&1; then
    LLDB_SERVER="$(command -v debugserver)"
  else
    die "lldb-server/debugserver not found (checked the RustRover bundle and PATH)
       set LLDB_SERVER to your RustRover bundle's lldb-server (Linux) or debugserver (macOS)"
  fi
fi
[[ -x "$LLDB_SERVER" ]] || die "lldb-server not executable: $LLDB_SERVER"

# lldb-server needs its `gdbserver` subcommand; debugserver takes none.
if [[ "$(basename "$LLDB_SERVER")" == lldb-server ]]; then
  SERVER_ARGS=(gdbserver)
else
  SERVER_ARGS=()
fi

# macOS ships bash 3.2 (last GPLv2 release), where `"${arr[@]}"` on a
# zero-element array raises "unbound variable" under `set -u`. Drop nounset
# here rather than working around it at each call site: every variable
# referenced from this point on is already validated above.
set +u

# --- port helpers ----------------------------------------------------------
# Print PIDs of processes LISTENing on $PORT (owned by this user), if any.
listener_pids() {
  # NB: must always exit 0 — a trailing grep with no match would otherwise
  # abort the script via `set -e` when used in `pids="$(listener_pids)"`.
  if command -v ss >/dev/null 2>&1; then
    ss -ltnpH "sport = :$PORT" 2>/dev/null \
      | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
  fi
}

port_listening() { [[ -n "$(listener_pids)" ]]; }

# Free the port if a previous (possibly crashed) session left a server behind.
# Without this, a stale server would silently accept the next connection and
# you'd debug the wrong, old python process.
kill_stale() {
  local pids; pids="$(listener_pids)"
  [[ -z "$pids" ]] && return 0
  log "port $PORT already in use by pid(s): $pids — terminating"
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 20); do
    port_listening || return 0
    sleep 0.1
  done
  kill -9 $pids 2>/dev/null || true
  sleep 0.2
  port_listening && die "could not free port $PORT (pid(s): $pids)"
  return 0
}

# --- launch ----------------------------------------------------------------
kill_stale

if [[ "$FOREGROUND" == "1" ]]; then
  log "lldb-server : $LLDB_SERVER"
  log "python      : $PYTHON"
  log "target      : $TARGET $*"
  log "listening on $HOST:$PORT — start the RustRover 'Remote Debug' config now"
  # Replace this shell; Ctrl-C then stops the server directly.
  exec "$LLDB_SERVER" "${SERVER_ARGS[@]}" "$HOST:$PORT" -- "$PYTHON" "$TARGET" "$@"
fi

# Background mode (default), suitable as a blocking "Before launch" task:
# start the server detached, then return 0 only once it is listening so the
# Remote Debug config never races ahead of a not-yet-bound port.
LOG_FILE="$SCRIPT_DIR/.lldb-server.log"
: > "$LOG_FILE"
# setsid (Linux/util-linux) fully detaches into a new session; macOS has no
# setsid, so fall back to nohup, which is enough to survive this shell exiting.
if command -v setsid >/dev/null 2>&1; then
  setsid "$LLDB_SERVER" "${SERVER_ARGS[@]}" "$HOST:$PORT" -- "$PYTHON" "$TARGET" "$@" \
    >"$LOG_FILE" 2>&1 </dev/null &
else
  nohup "$LLDB_SERVER" "${SERVER_ARGS[@]}" "$HOST:$PORT" -- "$PYTHON" "$TARGET" "$@" \
    >"$LOG_FILE" 2>&1 </dev/null &
fi
server_pid=$!

for _ in $(seq 1 100); do   # up to ~10s
  if port_listening; then
    log "lldb-server listening on $HOST:$PORT (pid $server_pid)"
    log "target: $PYTHON $TARGET $*"
    log "server log: $LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    printf 'error: lldb-server exited before it started listening. Output:\n' >&2
    cat "$LOG_FILE" >&2
    exit 1
  fi
  sleep 0.1
done

kill "$server_pid" 2>/dev/null || true
printf 'error: timed out waiting for lldb-server to listen on %s:%s. Output:\n' "$HOST" "$PORT" >&2
cat "$LOG_FILE" >&2
exit 1
