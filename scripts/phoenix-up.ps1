# 起本地 Phoenix（评估 CI/CD 用）。
# localhost 必须进 NO_PROXY：本机 Privoxy 会拦 127.0.0.1，表现为莫名其妙的连接失败。
$ErrorActionPreference = "Stop"

$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = $env:NO_PROXY
$env:PHOENIX_WORKING_DIR = Join-Path $PSScriptRoot "..\.phoenix"
if (-not (Test-Path $env:PHOENIX_WORKING_DIR)) {
    New-Item -ItemType Directory -Force $env:PHOENIX_WORKING_DIR | Out-Null
}
Write-Host "Phoenix working dir: $env:PHOENIX_WORKING_DIR"
Write-Host "UI: http://localhost:6006"

# phoenix.exe 这个 console_script 常被 pip 装进用户级 Scripts 目录
# （例如 %APPDATA%\Python\PythonXXX\Scripts），不一定在 PATH 里——这台机器上
# 实测就撞过这个坑（Task 1 Step 5）。优先走 PATH 无关的
# `python -m phoenix.server.main serve`：只要 python 解释器在 PATH 里，
# 不管 phoenix.exe 装在哪个 Scripts 目录都能起来。
# 找不到 python，或 python 没装 arize-phoenix，才退回检查 `phoenix` 命令本身；
# 两条路都走不通就打印清楚的 ERROR 并非零退出。
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue

if ($pythonCmd) {
    python -c "import phoenix" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Starting via: python -m phoenix.server.main serve"
        python -m phoenix.server.main serve
        exit $LASTEXITCODE
    }
    Write-Host "ERROR: 找到 python ($($pythonCmd.Source)) 但没装 arize-phoenix。"
    Write-Host 'ERROR: 运行 pip install arize-phoenix "arize-phoenix-client[pytest,evals]" openinference-instrumentation-langchain pyyaml 后重试。'
    exit 1
}

$phoenixCmd = Get-Command phoenix -ErrorAction SilentlyContinue
if (-not $phoenixCmd) {
    Write-Host "ERROR: 找不到 python 解释器，也找不到 phoenix 命令。"
    Write-Host "ERROR: phoenix.exe 通常装在 pip 的用户级 Scripts 目录（例如 %APPDATA%\Python\PythonXXX\Scripts），需要把它加进 PATH；"
    Write-Host "ERROR: 更简单的办法是确保 python 在 PATH 里，本脚本会优先用 python -m phoenix.server.main serve 启动，与 PATH 里哪个 Scripts 目录无关。"
    exit 1
}

Write-Host "Starting via: phoenix serve"
phoenix serve
exit $LASTEXITCODE
