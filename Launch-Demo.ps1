$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$demoPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $demoPython)) {
    $demoPython = Join-Path $PSScriptRoot '../../work/venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $demoPython)) {
    throw 'Create .venv and install requirements as explained in README.md first.'
}
& $demoPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
