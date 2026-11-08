#!/bin/sh
# Build wattsentry._core into ./.venv (team/infra.md §4), and probe the worker's Rust toolchain.
#
#   sh rust/build.sh build            # setup step: never exits non-zero; writes rust/.build_status
#   sh rust/build.sh status           # cmd step: prints the probe/build result as LAB_METRIC lines
#   sh rust/build.sh status --require # same, but exits 1 if the extension did not build/import
#
# Uses cargo if present, else installs a minimal rustup into $HOME/.cargo (cached on the worker).
# The cargo target dir lives in $HOME/.cache so later runs on the same worker build incrementally.

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STATUS="$ROOT/rust/.build_status"
PY="$ROOT/.venv/bin/python"

now() { date +%s.%N 2>/dev/null || date +%s; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f", b - a }'; }

build() {
  : >"$STATUS"
  rec() { echo "$1" >>"$STATUS"; }
  echo "== rust probe: host=$(hostname) nproc=$(nproc 2>/dev/null || echo ?) cpu=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//')"
  rec "LAB_METRIC worker_nproc=$(nproc 2>/dev/null || echo 0)"

  if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
    PATH="$HOME/.cargo/bin:$PATH"; export PATH
    echo "== cargo found in \$HOME/.cargo (cached rustup)"
    rec "LAB_METRIC cargo_preinstalled=1"
  elif command -v cargo >/dev/null 2>&1; then
    echo "== cargo found on PATH: $(command -v cargo)"
    rec "LAB_METRIC cargo_preinstalled=1"
  else
    echo "== no cargo: installing minimal rustup into \$HOME/.cargo"
    rec "LAB_METRIC cargo_preinstalled=0"
    t0=$(now)
    if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.$$.sh \
        && sh /tmp/rustup-init.$$.sh -y --profile minimal --no-modify-path >/tmp/rustup.$$.log 2>&1; then
      t1=$(now)
      echo "== rustup installed in $(elapsed "$t0" "$t1") s"
      rec "LAB_METRIC rustup_install_s=$(elapsed "$t0" "$t1")"
    else
      echo "!! rustup install FAILED (no network to static.rust-lang.org?):"
      tail -20 /tmp/rustup.$$.log 2>/dev/null
      rec "LAB_METRIC rust_build_ok=0"
      rec "FAILED: rustup install"
      rm -f /tmp/rustup-init.$$.sh /tmp/rustup.$$.log
      return 0
    fi
    rm -f /tmp/rustup-init.$$.sh /tmp/rustup.$$.log
    PATH="$HOME/.cargo/bin:$PATH"; export PATH
  fi
  echo "== $(cargo --version 2>&1)"
  echo "== $(rustc --version 2>&1)"

  CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/wattsentry-core-target}"; export CARGO_TARGET_DIR
  VIRTUAL_ENV="$ROOT/.venv"; export VIRTUAL_ENV
  echo "== maturin develop --release (target dir $CARGO_TARGET_DIR)"
  t0=$(now)
  LOG=/tmp/maturin.$$.log
  "$ROOT/.venv/bin/maturin" develop --release --uv -m "$ROOT/rust/wattsentry_core/Cargo.toml" >"$LOG" 2>&1
  rc=$?
  grep -v '^ *Compiling ' "$LOG" | tail -40
  rm -f "$LOG"
  if [ "$rc" -eq 0 ] && "$PY" -c "import wattsentry._core as c; print('== wattsentry._core', c.__version__, 'threads', c.num_threads())"; then
    t1=$(now)
    echo "== build OK in $(elapsed "$t0" "$t1") s"
    rec "LAB_METRIC rust_build_s=$(elapsed "$t0" "$t1")"
    rec "LAB_METRIC rust_build_ok=1"
    rec "OK"
  else
    t1=$(now)
    echo "!! build FAILED after $(elapsed "$t0" "$t1") s; kernels fall back to Python"
    rec "LAB_METRIC rust_build_ok=0"
    rec "FAILED: maturin develop (see setup log)"
  fi
  return 0
}

status() {
  if [ ! -f "$STATUS" ]; then
    echo "!! rust build status missing (setup did not run rust/build.sh build)"
    echo "LAB_METRIC rust_build_ok=0"
  else
    grep '^LAB_METRIC' "$STATUS"
    grep -v '^LAB_METRIC' "$STATUS" | sed 's/^/== rust build: /'
  fi
  if [ "$1" = "--require" ]; then
    if ! "$PY" -c "import wattsentry._core" 2>/dev/null; then
      echo "!! RUST BUILD FAILED: wattsentry._core does not import; see the setup log above" >&2
      exit 1
    fi
  fi
  return 0
}

case "$1" in
  build) build ;;
  status) shift; status "$@" ;;
  *) echo "usage: sh rust/build.sh build | status [--require]" >&2; exit 2 ;;
esac
