#!/usr/bin/env bash
# Blue-green venv swap. Run AFTER the deps install has populated a fresh .venv_new.
# Import-verifies the new venv boots app.main (with the live service's env loaded so config
# validation passes), then atomically swaps .venv_new -> .venv and re-verifies. On ANY
# failure it exits non-zero BEFORE swapping (or rolls back), leaving the live venv untouched.
# Kept as a file (not inline in deploy.yml's ssh heredoc) to avoid shell-escaping pitfalls.
set -uo pipefail
cd /var/www/mivaa-pdf-extractor

if [ ! -d .venv_new ]; then
  echo "bluegreen: no .venv_new present — nothing to swap"
  exit 1
fi

# The deploy shell exports LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu for the pip build, but the
# service runs WITHOUT it — forcing system libs ahead of the wheels' bundled native libs
# SEGFAULTS them at import. Clear it so the verify matches the live service's runtime env (the
# env-load below re-sets it only if the service actually uses it).
unset LD_LIBRARY_PATH

# Load the running service's exact environment (API keys etc.) so 'import app.main' — which
# validates config at import time — succeeds during verification.
MAINPID="$(systemctl show -p MainPID --value mivaa-pdf-extractor 2>/dev/null || echo 0)"
if [ "${MAINPID:-0}" != "0" ] && [ -r "/proc/${MAINPID}/environ" ]; then
  set -a
  # shellcheck disable=SC1090
  source <(tr '\0' '\n' < "/proc/${MAINPID}/environ" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=')
  set +a
fi

echo "bluegreen: verifying .venv_new boots app.main ..."
if ! ./.venv_new/bin/python -c "import app.main"; then
  echo "bluegreen: NEW venv failed to import app.main — NOT swapping; live venv untouched"
  rm -rf .venv_new
  exit 1
fi

# Atomic-ish swap. The running process holds its open inodes on the old dir, so renaming it
# does not affect the live service — it keeps serving until the later systemctl restart.
rm -rf .venv_old
mv .venv .venv_old
mv .venv_new .venv

# Console-script shebangs AND the activate script's VIRTUAL_ENV hard-code the build path
# (.venv_new). Rewrite every reference to .venv now that it has been renamed.
grep -rlIF '.venv_new' .venv/bin 2>/dev/null | xargs -r sed -i 's#\.venv_new#.venv#g' || true

if ! ./.venv/bin/python -c "import app.main"; then
  echo "bluegreen: POST-SWAP import failed — rolling back to previous venv"
  rm -rf .venv
  mv .venv_old .venv
  exit 1
fi

echo "bluegreen: swapped in OK (live service still serving old venv until restart)"
