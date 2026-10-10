# =============================================================================
# DeepTrace Forensic Terminal // Live Demo Launcher (PowerShell)
# Precomputes demo caches and launches Streamlit with preloaded demo clip.
# =============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " 🔬 DeepTrace Digital Forensics Terminal // Live Demo" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Locate Python binary
$PythonBin = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $PythonBin = ".\.venv\Scripts\python.exe"
}

# 1. Precompute cached results
Write-Host "[1/3] Verifying and updating precomputed demo caches..." -ForegroundColor Gray
& $PythonBin scripts/precompute.py --data-dir data

# 2. Select preloaded demo clip
$DemoClip = "data\real\synth_real_01.mp4"
if (-not (Test-Path $DemoClip)) {
    $found = Get-ChildItem -Path data -Recurse -Filter "*.mp4" | Select-Object -First 1
    if ($found) { $DemoClip = $found.FullName }
}

Write-Host "[2/3] Preloading demo target: $DemoClip" -ForegroundColor Green
$env:DEEPTRACE_DEMO = $DemoClip

# 3. Launch Streamlit
Write-Host "[3/3] Launching Streamlit Digital Forensics Terminal UI..." -ForegroundColor Yellow
streamlit run app.py

