#!/bin/bash
# The new code is on disk; a running ingestion keeps the old process until it ends. Eight hours is
# longer than any real job, so past that the job is stuck and the restart goes ahead.
set -uo pipefail

SERVICE="${SERVICE:-mivaa-pdf-extractor}"
API="${API:-http://localhost:8000}"
INTERVAL="${INTERVAL:-120}"
MAX_WAIT="${MAX_WAIT:-28800}"
started=$(date +%s)

while :; do
  response=$(curl -s -X POST -m 30 -H "Content-Type: application/json" -H "X-Admin-Token: ${MIVAA_DEPLOY_TOKEN:-}" \
    "$API/api/admin/pause-for-deploy?max_wait_seconds=15" || echo '{}')
  if [ "$(echo "$response" | jq -r '.ready // false' 2>/dev/null)" = "true" ]; then
    echo "no ingestion running: restarting $SERVICE"
    break
  fi
  curl -s -X POST -m 15 -H "X-Admin-Token: ${MIVAA_DEPLOY_TOKEN:-}" "$API/api/admin/resume-from-deploy" >/dev/null || true
  if [ $(( $(date +%s) - started )) -ge "$MAX_WAIT" ]; then
    echo "still busy after ${MAX_WAIT}s: treating the job as stuck and restarting $SERVICE"
    break
  fi
  sleep "$INTERVAL"
done

systemctl restart "$SERVICE"
