# 起本地 Phoenix（评估 CI/CD 用）。
# localhost 必须进 NO_PROXY：本机 Privoxy 会拦 127.0.0.1，表现为莫名其妙的连接失败。
$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = $env:NO_PROXY
$env:PHOENIX_WORKING_DIR = Join-Path $PSScriptRoot "..\.phoenix"
if (-not (Test-Path $env:PHOENIX_WORKING_DIR)) {
    New-Item -ItemType Directory -Force $env:PHOENIX_WORKING_DIR | Out-Null
}
Write-Host "Phoenix working dir: $env:PHOENIX_WORKING_DIR"
Write-Host "UI: http://localhost:6006"
phoenix serve
