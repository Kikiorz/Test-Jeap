#!/usr/bin/env bash
# Report whether the two evaluation policy servers are answering.
set -uo pipefail

for entry in "baseline:8001" "adapter:8000"; do
  name="${entry%%:*}"
  port="${entry##*:}"
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${port}/healthz" || true)"
  printf '%-9s :%s healthz=%s\n' "$name" "$port" "${code:-none}"
done
