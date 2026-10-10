#!/usr/bin/env bash
# =============================================================================
# DeepTrace Forensic Terminal // Live Demo Launcher
# Precomputes demo caches and launches Streamlit with preloaded demo clip.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "============================================================"
echo " 🔬 DeepTrace Digital Forensics Terminal // Live Demo"
echo "============================================================"

# Ensure Python environment
PYTHON_BIN="python"
if [ -d ".venv" ]; then
    if [ -f ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
    elif [ -f ".venv/Scripts/python.exe" ]; then
        PYTHON_BIN=".venv/Scripts/python.exe"
    fi
fi

# 1. Precompute cached results so live demo loads with 0ms waiting
echo "[1/3] Verifying and updating precomputed demo caches..."
"$PYTHON_BIN" scripts/precompute.py --data-dir data

# 2. Select preloaded demo clip
DEMO_CLIP="data/real/synth_real_01.mp4"
if [ ! -f "$DEMO_CLIP" ]; then
    # Fallback to first available mp4 in data
    DEMO_CLIP=$(find data -type f -name "*.mp4" 2>/dev/null | head -n 1)
fi

echo "[2/3] Preloading demo target: $DEMO_CLIP"
export DEEPTRACE_DEMO="$DEMO_CLIP"

# 3. Launch Streamlit Forensic Terminal
echo "[3/3] Launching Streamlit Digital Forensics Terminal UI..."
streamlit run app.py

