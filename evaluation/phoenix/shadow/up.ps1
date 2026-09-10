# 起影子 canary：stable(8101) + canary(8102) + 分流器(8100)。
#
# 两个实例用不同 APP_VERSION、相同 PHOENIX_PROJECT_NAME=bz-rag-canary，
# 这样 trace 落在同一个 project 里、靠 APP_VERSION 区分版本——与设计文档 §3
# 「一个 Phoenix 实例、两个 project：CI 的与线上的分开」一致。
#
# 前提：本机 Milvus(19530) 与 Phoenix(6006) 都在跑（scripts/phoenix-up.ps1）。
#
# **必须在交互式终端里运行**：两个后端用 Start-Process 各开一个窗口，
# 在非交互会话（CI、后台任务）里开不出来，结果是只有分流器起来、两个后端缺席。
# 非交互场景请直接分别起三个进程：
#   $env:PHOENIX_PROJECT_NAME='bz-rag-canary'; $env:APP_VERSION='stable'
#   python -m uvicorn api.main:app --port 8101
#   （canary 同理换 APP_VERSION 与 8102，再起 router 的 8100）
$ErrorActionPreference = "Stop"
$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = $env:NO_PROXY

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repo
Write-Host "repo: $repo"

# 两个后端各开一个窗口，分流器留在前台（Ctrl+C 一次全停不了，三个窗口各自关）。
$common = "`$env:NO_PROXY='localhost,127.0.0.1'; `$env:no_proxy='localhost,127.0.0.1'; " +
          "`$env:PHOENIX_PROJECT_NAME='bz-rag-canary'; Set-Location '$repo'; "

Start-Process powershell -ArgumentList @(
  "-NoExit", "-Command",
  $common + "`$env:APP_VERSION='stable'; python -m uvicorn api.main:app --port 8101"
)
Start-Process powershell -ArgumentList @(
  "-NoExit", "-Command",
  $common + "`$env:APP_VERSION='canary'; python -m uvicorn api.main:app --port 8102"
)

Write-Host "stable -> http://127.0.0.1:8101   (APP_VERSION=stable)"
Write-Host "canary -> http://127.0.0.1:8102   (APP_VERSION=canary)"
Write-Host "router -> http://127.0.0.1:8100   (前台运行，Ctrl+C 停止)"
Write-Host "权重见 evaluation/phoenix/shadow/weight.json"

$env:PHOENIX_PROJECT_NAME = "bz-rag-canary"
python -m uvicorn evaluation.phoenix.shadow.router:app --port 8100
