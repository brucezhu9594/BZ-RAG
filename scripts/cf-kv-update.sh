#!/bin/bash
# cf-kv-update.sh — 更新 Cloudflare KV 中的 canary_weight，控制金丝雀流量比例
#
# 用法: ./scripts/cf-kv-update.sh <weight>
#   weight 是 0-100 的整数，代表 canary 接收的流量百分比
#   stable 自动等于 100 - weight
#
# 例:
#   ./scripts/cf-kv-update.sh 0     # 100% stable
#   ./scripts/cf-kv-update.sh 5     # 5% canary, 95% stable
#   ./scripts/cf-kv-update.sh 100   # 100% canary
#
# 必需环境变量: CF_API_TOKEN, CF_ACCOUNT_ID, KV_NAMESPACE_ID
#
# 这是 Mira 的 cf-lb-weights.sh 的等价物 —— 我们用 Worker + KV 自实现金丝雀
# 而不是付费 LB pool，因为 BZ-RAG 没有自有域名走免费 workers.dev 路线

set -euo pipefail

WEIGHT="${1:?Usage: cf-kv-update.sh <canary_weight 0-100>}"

if ! [[ "${WEIGHT}" =~ ^[0-9]+$ ]] || [ "${WEIGHT}" -lt 0 ] || [ "${WEIGHT}" -gt 100 ]; then
  echo "ERROR: weight must be integer 0-100, got: ${WEIGHT}"
  exit 1
fi

: "${CF_API_TOKEN:?CF_API_TOKEN is required}"
: "${CF_ACCOUNT_ID:?CF_ACCOUNT_ID is required}"
: "${KV_NAMESPACE_ID:?KV_NAMESPACE_ID is required}"

echo "Updating canary_weight to ${WEIGHT} (stable=$((100 - WEIGHT))%, canary=${WEIGHT}%)"

RESPONSE=$(curl -sS -X PUT \
  "https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/storage/kv/namespaces/${KV_NAMESPACE_ID}/values/canary_weight" \
  -H "Authorization: Bearer ${CF_API_TOKEN}" \
  -H "Content-Type: text/plain" \
  --data-raw "${WEIGHT}")

SUCCESS=$(echo "${RESPONSE}" | jq -r '.success')

if [ "${SUCCESS}" != "true" ]; then
  echo "ERROR: Cloudflare KV API returned success=false"
  echo "${RESPONSE}" | jq .
  exit 1
fi

echo "KV updated successfully"
echo "${RESPONSE}" | jq '{success: .success, errors: .errors}'
