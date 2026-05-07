#!/bin/bash
# wait-for-health.sh — 轮询 health 端点直到服务返回期望版本
#
# 用法: ./scripts/wait-for-health.sh <health_url> <expected_version> <max_seconds>
# 例: ./scripts/wait-for-health.sh https://bz-rag-canary-production.up.railway.app/api/health 1.0.0 120

set -euo pipefail

HEALTH_URL="${1:?Usage: wait-for-health.sh <health_url> <expected_version> <max_seconds>}"
EXPECTED_VERSION="${2:?Usage: wait-for-health.sh <health_url> <expected_version> <max_seconds>}"
MAX_SECONDS="${3:-120}"
INTERVAL=5

echo "Waiting for ${HEALTH_URL} to be healthy (version containing '${EXPECTED_VERSION}', timeout ${MAX_SECONDS}s)"

elapsed=0
while [ "${elapsed}" -lt "${MAX_SECONDS}" ]; do
  RESPONSE=$(curl -sS --max-time 10 "${HEALTH_URL}" 2>/dev/null || true)

  if [ -n "${RESPONSE}" ]; then
    STATUS=$(echo "${RESPONSE}" | jq -r '.status' 2>/dev/null || true)
    VERSION=$(echo "${RESPONSE}" | jq -r '.version' 2>/dev/null || true)

    if [ "${STATUS}" = "ok" ]; then
      if echo "${VERSION}" | grep -q "${EXPECTED_VERSION}"; then
        echo "Health check passed: status=${STATUS}, version=${VERSION} (${elapsed}s elapsed)"
        exit 0
      fi
      echo "  status=ok but version mismatch: got '${VERSION}', want '${EXPECTED_VERSION}' (${elapsed}s)"
    else
      echo "  status='${STATUS}' (${elapsed}s)"
    fi
  else
    echo "  no response (${elapsed}s)"
  fi

  sleep "${INTERVAL}"
  elapsed=$((elapsed + INTERVAL))
done

echo "ERROR: Health check timed out after ${MAX_SECONDS}s"
exit 1
