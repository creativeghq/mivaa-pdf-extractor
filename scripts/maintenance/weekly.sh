#!/usr/bin/env bash
# Weekly self-maintenance of the MIVAA server (#590): drain, update, reboot if needed, verify, clean, report.
# A fixed recipe. It takes no input from the network; the platform can only ask for it to run (see poll).
# Usage: weekly.sh run [scheduled|on_demand] | weekly.sh resume | weekly.sh poll
set -uo pipefail

ENV_FILE=/etc/mivaa-maintenance.env
STATE=/var/lib/mivaa-maintenance
APP=/var/www/mivaa-pdf-extractor
HERE="$(cd "$(dirname "$0")" && pwd)"
LOCAL_HEALTH=http://127.0.0.1:8000/health
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_SUSPEND=1 NEEDRESTART_MODE=l

[ -r "$ENV_FILE" ] || { echo "missing $ENV_FILE" >&2; exit 78; }
# shellcheck disable=SC1090
. "$ENV_FILE"
: "${MAINT_SERVER:?}" "${MAINT_SECRET:?}" "${MAINT_REPORT_URL:?}" "${MAINT_PUBLIC_HEALTH:?}"
mkdir -p "$STATE/backups"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
used_bytes() { df -B1 --output=used / | tail -1 | tr -d ' '; }

signed_post() { # $1 action, $2 json body file
  local ts sig
  ts=$(date +%s)
  # Sign the file's exact bytes: $(cat) would drop the trailing newline that curl still sends.
  sig=$( { printf '%s.' "$ts"; cat "$2"; } | openssl dgst -sha256 -hmac "$MAINT_SECRET" -hex | awk '{print $NF}')
  curl -sS -m 60 -X POST "$MAINT_REPORT_URL?server=$MAINT_SERVER&action=$1" \
    -H 'Content-Type: application/json' -H "x-kai-timestamp: $ts" -H "x-kai-signature: $sig" \
    --data-binary @"$2"
}

# One JSON line per step: name, status (ok|failed|warning|skipped), seconds, detail, output tail.
record() {
  jq -nc --arg name "$1" --arg status "$2" --arg detail "$3" --arg out "$(printf '%s' "${4:-}" | tail -c 3000)" \
    --argjson secs "${5:-0}" --argjson data "${6:-null}" \
    '{name:$name,status:$status,detail:$detail,output:$out,seconds:$secs,data:$data,at:(now|todate)}' >> "$STATE/steps.jsonl"
}

versions() {
  jq -nc --arg kernel "$(uname -r)" --arg kernel_installed "$(ls /boot/vmlinuz-* 2>/dev/null | sed 's#.*/vmlinuz-##' | sort -V | tail -1)" \
    --arg nginx "$(nginx -v 2>&1 | sed 's#.*/##')" --arg os "$(. /etc/os-release; echo "$PRETTY_NAME")" \
    --argjson upgradable "$(apt list --upgradable 2>/dev/null | tail -n +2 | grep -c . || true)" \
    '{kernel:$kernel,kernel_installed:$kernel_installed,nginx:$nginx,os:$os,upgradable:$upgradable}'
}

deploy_token() {
  local pid; pid=$(systemctl show -p MainPID --value mivaa-pdf-extractor)
  [ "${pid:-0}" -gt 0 ] && tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^MIVAA_DEPLOY_TOKEN=//p' | head -1
}

admin_call() { # $1 path; never fails the run, returns the body
  local tok; tok=$(deploy_token)
  [ -n "$tok" ] || { echo '{"error":"no deploy token"}'; return 0; }
  curl -s -m 960 -X POST -H "X-Admin-Token: $tok" "http://127.0.0.1:8000/api/admin/$1" || echo '{"error":"call failed"}'
}

begin() {
  rm -f "$STATE/steps.jsonl"
  jq -nc --arg id "$(cat /proc/sys/kernel/random/uuid)" --arg trigger "$1" --arg started "$(now)" \
    --argjson before "$(versions)" --argjson used "$(used_bytes)" \
    '{id:$id,trigger:$trigger,started_at:$started,versions_before:$before,disk_used_before:$used,rebooted:false}' > "$STATE/run.json"
}

set_run() { jq -c "$1" "$STATE/run.json" > "$STATE/run.tmp" && mv "$STATE/run.tmp" "$STATE/run.json"; }

preflight() {
  local t0=$SECONDS r
  while :; do
    r=$(admin_call 'pause-for-deploy?max_wait_seconds=600')
    if [ "$(jq -r '.ready // false' <<<"$r" 2>/dev/null)" = "true" ]; then
      record preflight ok "MIVAA drained: no document is processing." "$r" $((SECONDS - t0)); return 0
    fi
    if jq -e '.error' <<<"$r" >/dev/null 2>&1 && ! systemctl is-active -q mivaa-pdf-extractor; then
      record preflight warning "MIVAA was not running, so there was nothing to drain." "$r" $((SECONDS - t0)); return 0
    fi
    admin_call resume-from-deploy >/dev/null
    if [ "$(TZ=Europe/Athens date +%H)" -ge 4 ] && [ "$(TZ=Europe/Athens date +%H)" -lt 20 ]; then
      record preflight skipped "Documents were still processing at 04:00 Athens time; nothing was changed." "$r" $((SECONDS - t0))
      return 1
    fi
    sleep 1800
  done
}

backup_config() {
  local f="$STATE/backups/etc-$(date +%Y%m%d-%H%M%S).tgz"
  tar czf "$f" /etc/nginx /etc/systemd/system/mivaa-pdf-extractor.service /etc/systemd/system/mivaa-pdf-extractor.service.d 2>/dev/null
  echo "$f" > "$STATE/last_backup"
  find "$STATE/backups" -name 'etc-*.tgz' -mtime +28 -delete
  record backup ok "Config saved to $f" "" 0
}

update_packages() {
  local t0=$SECONDS out rc
  # Phased updates are held back by `upgrade`; naming them installs them.
  out=$( { apt-get update -q && apt-get -y -o Dpkg::Options::=--force-confold -o Dpkg::Options::=--force-confdef --with-new-pkgs upgrade \
    && apt-get -y -o Dpkg::Options::=--force-confold install $(apt list --upgradable 2>/dev/null | tail -n +2 | cut -d/ -f1); } 2>&1 )
  rc=$?
  local n; n=$(grep -c '^Setting up' <<<"$out" || true)
  if [ $rc -eq 0 ]; then record update ok "$n package(s) updated." "$out" $((SECONDS - t0))
  else record update failed "apt exited $rc after updating $n package(s)." "$out" $((SECONDS - t0)); fi
}

upstream_check() {
  local v page gap
  v=$(nginx -v 2>&1 | sed 's#.*/##')
  page=$(curl -fsS -m 30 https://nginx.org/en/security_advisories.html) || { record upstream warning "nginx.org advisories could not be fetched." "" 0; return; }
  if gap=$(printf '%s' "$page" | /usr/bin/python3 -I "$HERE/nginx_upstream_gap.py" "$v"); then
    set_run ".upstream_gap = $gap"
    if [ "$(jq length <<<"$gap")" -eq 0 ]; then record upstream ok "nginx $v is not affected by any nginx.org advisory." "" 0
    else record upstream warning "nginx $v is affected by $(jq length <<<"$gap") nginx.org advisory(ies)." "$gap" 0; fi
  else
    record upstream warning "The nginx.org advisory page could not be parsed." "$gap" 0
  fi
}

restart_or_reboot() {
  if [ -f /var/run/reboot-required ]; then
    record reboot ok "Rebooting: $(tr '\n' ' ' < /var/run/reboot-required.pkgs 2>/dev/null)" "" 0
    set_run '.rebooted = true'
    echo resume > "$STATE/phase"
    systemctl reboot
    exit 0
  fi
  local out; out=$(needrestart -r a -q 2>&1)
  record restart ok "No reboot needed; services on old libraries restarted." "$out" 0
}

wait_for() { # $1 seconds, then a command
  local n=$1; shift
  for _ in $(seq 1 "$n"); do "$@" >/dev/null 2>&1 && return 0; sleep 1; done; return 1
}

verify() {
  local problems=() out
  if ! nginx -t >/dev/null 2>&1 || ! systemctl is-active -q nginx; then
    out=$(nginx -t 2>&1)
    local b; b=$(cat "$STATE/last_backup" 2>/dev/null)
    if [ -n "$b" ] && tar xzf "$b" -C / etc/nginx && nginx -t >/dev/null 2>&1 && systemctl restart nginx; then
      record nginx_rollback warning "nginx failed after the update; its config was restored from $b." "$out" 0
    else
      record nginx_rollback failed "nginx failed and restoring its config did not fix it." "$out" 0
      problems+=("nginx down")
    fi
  fi
  for s in nginx mivaa-pdf-extractor languagetool; do
    systemctl is-active -q "$s" || { systemctl restart "$s"; sleep 5; systemctl is-active -q "$s" || problems+=("$s not running"); }
  done
  wait_for 180 sh -c "curl -fsS -m 5 $LOCAL_HEALTH | jq -e '.status==\"healthy\"'" || problems+=("MIVAA /health not healthy")
  wait_for 60 curl -fsS -m 10 -o /dev/null "$MAINT_PUBLIC_HEALTH" || problems+=("public $MAINT_PUBLIC_HEALTH not 200")
  local failed; failed=$(systemctl --failed --no-legend --plain | awk '{print $1}' | grep -v '^cloud-init' | tr '\n' ' ')
  [ -z "$failed" ] || problems+=("failed units: $failed")
  local v; v=$(versions)
  [ "$(jq -r .kernel <<<"$v")" = "$(jq -r .kernel_installed <<<"$v")" ] || problems+=("running kernel is not the newest installed")
  [ -f /var/run/reboot-required ] && problems+=("a reboot is still pending")
  if [ ${#problems[@]} -eq 0 ]; then record verify ok "nginx, MIVAA and LanguageTool are up; local and public health pass." "" 0
  else record verify failed "$(IFS='; '; echo "${problems[*]}")" "$(systemctl --failed --no-legend 2>&1)" 0; fi
  admin_call resume-from-deploy >/dev/null
}

clean() {
  local before after out=""
  before=$(used_bytes)
  apt-get clean
  journalctl --vacuum-time=7d -q
  mkdir -p /etc/systemd/journald.conf.d
  if [ ! -f /etc/systemd/journald.conf.d/90-mivaa-cap.conf ]; then
    printf '[Journal]\nSystemMaxUse=300M\n' > /etc/systemd/journald.conf.d/90-mivaa-cap.conf; systemctl restart systemd-journald
  fi
  out+=$(find /var/log -type f \( -name '*.[0-9]' -o -name '*.gz' -o -name '*.old' \) -not -path '*/journal/*' -mtime +7 -print -delete 2>&1)
  out+=$(find /tmp /var/tmp -mindepth 1 -maxdepth 1 -mtime +7 ! -name 'systemd-private-*' ! -name 'snap-private-tmp' ! -name 'hsperfdata_*' -print -exec rm -rf {} + 2>&1)
  rm -rf /root/.cache/pip
  # ctime, not mtime: the deploy's swap is a rename, which keeps the old mtime of a venv it just retired.
  if [ -d "$APP/.venv_old" ] && [ -n "$(find "$APP/.venv_old" -maxdepth 0 -ctime +14)" ]; then rm -rf "$APP/.venv_old"; out+=$'\n'"$APP/.venv_old"; fi
  after=$(used_bytes)
  set_run ".freed_bytes = $((before - after))"
  record clean ok "Freed $(( (before - after) / 1048576 )) MB." "$out" 0
}

finish() {
  local verdict=ok reason
  if grep -q '"status":"skipped"' "$STATE/steps.jsonl"; then verdict=skipped
  elif jq -e 'select(.name=="verify" and .status=="failed")' "$STATE/steps.jsonl" >/dev/null; then
    verdict=failed; grep -q '"nginx_rollback","status":"failed"' "$STATE/steps.jsonl" && verdict=critical
  elif grep -qE '"status":"(failed|warning)"' "$STATE/steps.jsonl"; then verdict=degraded; fi
  reason=$(jq -rs '[.[] | select(.status!="ok") | .name + ": " + .detail] | join(" | ")' "$STATE/steps.jsonl")
  jq -c --arg verdict "$verdict" --arg reason "$reason" --arg finished "$(now)" --argjson after "$(versions)" \
    --argjson used "$(used_bytes)" --argjson total "$(df -B1 --output=size / | tail -1 | tr -d ' ')" \
    --slurpfile steps "$STATE/steps.jsonl" \
    '. + {verdict:$verdict,reason:$reason,finished_at:$finished,versions_after:$after,disk_used_after:$used,disk_total_bytes:$total,steps:$steps}' \
    "$STATE/run.json" > "$STATE/report.json"
  rm -f "$STATE/phase"
  local resp
  for attempt in 1 2 3 4 5; do
    resp=$(signed_post report "$STATE/report.json") && jq -e '.stored' <<<"$resp" >/dev/null 2>&1 && break
    sleep $((attempt * 60))
  done
  echo "maintenance $verdict: $reason"
}

case "${1:-run}" in
  run)
    exec 9>/run/mivaa-maintenance.lock; flock -n 9 || { echo "already running"; exit 0; }
    begin "${2:-scheduled}"
    if preflight; then backup_config; update_packages; upstream_check; restart_or_reboot; verify; clean; fi
    finish ;;
  resume)
    [ "$(cat "$STATE/phase" 2>/dev/null)" = resume ] || exit 0
    exec 9>/run/mivaa-maintenance.lock; flock -n 9 || exit 0
    record boot ok "Back up after the reboot on kernel $(uname -r)." "" 0
    verify; clean; finish ;;
  poll)
    echo '{}' > "$STATE/poll.json"
    r=$(signed_post poll "$STATE/poll.json") || exit 0
    [ "$(jq -r '.run_requested // false' <<<"$r" 2>/dev/null)" = true ] && systemctl start --no-block mivaa-maintenance-ondemand.service
    exit 0 ;;
  *) echo "usage: $0 run|resume|poll" >&2; exit 64 ;;
esac
