#!/usr/bin/env bash
set -euo pipefail

# AiTutor Local Edge Server Launcher (macOS Apple Silicon & Linux)
# Resolves project root portably without hardcoded paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== AiTutor: 100% Local Multimodal AI Tutor ==="
echo "Project root: $PROJECT_ROOT"

# 1. Virtual Environment Setup
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment (.venv)..."
    if ! python3 -m venv "$VENV_DIR" 2>/dev/null; then
        if command -v uv >/dev/null 2>&1; then
            uv venv --python 3.11 --seed "$VENV_DIR"
        else
            python3 -m venv "$VENV_DIR"
        fi
    fi
    "$VENV_DIR/bin/pip" install --upgrade pip || true
    echo "Installing dependencies from requirements.txt..."
    "$VENV_DIR/bin/pip" install -r "$PROJECT_ROOT/requirements.txt"
else
    echo "[1/4] Virtual environment exists at $VENV_DIR."
fi

# 2. Verify / Download Offline Model Weights & Assets
echo "[2/4] Verifying local model weights and vendor assets..."
"$VENV_DIR/bin/python" "$PROJECT_ROOT/scripts/download_models.py"

# 3. Check Local Ollama Engine
echo "[3/4] Checking Ollama LLM service (http://localhost:11434)..."
if curl -s -f -m 3 http://localhost:11434/api/tags > /dev/null 2>&1; then
    TAGS_OUTPUT=$(curl -s http://localhost:11434/api/tags)
    if echo "$TAGS_OUTPUT" | grep -q -E '(qwen2\.5:7b|llama3\.2:3b)'; then
        echo "Found recommended local model in Ollama."
    else
        echo "WARNING: Recommended model (qwen2.5:7b-instruct or llama3.2:3b) not detected in Ollama."
        echo "To pull recommended model, run:"
        echo "  ollama pull qwen2.5:7b-instruct"
        echo "  (or: ollama pull llama3.2:3b)"
    fi
else
    echo "WARNING: Ollama does not appear to be running on http://localhost:11434."
    echo "Start Ollama in another terminal:"
    echo "  ollama serve"
    echo "(Or install Ollama via 'brew install ollama' on macOS)"
fi

# 4. Launch FastAPI Uvicorn Server
echo "[4/4] Starting AiTutor server on http://127.0.0.1:8000..."
exec "$VENV_DIR/bin/uvicorn" backend.main:app --host 127.0.0.1 --port 8000 "$@"
