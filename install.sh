#!/usr/bin/env bash
set -e

# === CONFIGURATION ===
REPO_URL="https://github.com/potassco/pddl-instances.git"
SUBMODULE_DIR="pddl-instances"
VENV_DIR=".venv"
SCRIPT_NAME="analyze_pddl_domains.py"

echo "=== PDDL Instances Analyzer Setup ==="

# --- 1. Clone or update the PDDL instances repository as a submodule ---
if [ -d "$SUBMODULE_DIR/.git" ]; then
    echo "[INFO] Submodule already exists, updating..."
    git submodule update --remote --merge "$SUBMODULE_DIR"
else
    echo "[INFO] Adding PDDL instances repository as a submodule..."
    git submodule add "$REPO_URL" "$SUBMODULE_DIR" || {
        echo "[WARN] Submodule add failed — trying regular clone."
        git clone "$REPO_URL" "$SUBMODULE_DIR"
    }
fi

# --- 2. Create and activate virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "[INFO] Virtual environment activated."

# --- 3. Install required dependencies ---
echo "[INFO] Installing dependencies..."
pip install --upgrade pip
pip install unified-planning

echo "=== DONE ==="
