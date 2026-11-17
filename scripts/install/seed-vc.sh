#!/usr/bin/env bash
# Copyright 2026 Creative Stack Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Provision an isolated Seed-VC environment for zero-shot Singing Voice
# Conversion (SVC) and real-time GUI experimentation (`Plachtaa/seed-vc`).
#
# Pinned to an audited commit SHA per AGENTS.md supply-chain rules and kept in
# its own virtualenv (`~/.local/share/creative-stack/seed-vc/venv`) so its pinned
# ML dependencies never collide with creative-stack's repo `.venv`.
set -euo pipefail

SEED_VC_COMMIT="${SEED_VC_COMMIT:-51383efd921027683c89e5348211d93ff12ac2a8}"
ROOT_DIR="${CREATIVE_STACK_SEED_VC:-$HOME/.local/share/creative-stack/seed-vc}"
REPO_DIR="$ROOT_DIR/repo"
VENV_DIR="$ROOT_DIR/venv"

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ] && [ -x "/Library/Frameworks/Python.framework/Versions/3.12/bin/uv" ]; then
  UV_BIN="/Library/Frameworks/Python.framework/Versions/3.12/bin/uv"
fi
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
[ -n "$UV_BIN" ] || { echo "❌ uv is required to provision seed-vc" >&2; exit 1; }

command -v git >/dev/null 2>&1 || { echo "❌ git is required to clone seed-vc" >&2; exit 1; }

echo "▸ Provisioning Plachtaa/seed-vc @ ${SEED_VC_COMMIT:0:10} -> $ROOT_DIR"
mkdir -p "$ROOT_DIR"

if [ -d "$REPO_DIR/.git" ]; then
  CUR_SHA="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  if [ "$CUR_SHA" != "$SEED_VC_COMMIT" ]; then
    git -C "$REPO_DIR" fetch --depth 1 origin "$SEED_VC_COMMIT"
    git -C "$REPO_DIR" checkout --force FETCH_HEAD
  fi
else
  rm -rf "$REPO_DIR"
  git init "$REPO_DIR"
  git -C "$REPO_DIR" remote add origin https://github.com/Plachtaa/seed-vc.git
  git -C "$REPO_DIR" fetch --depth 1 origin "$SEED_VC_COMMIT"
  git -C "$REPO_DIR" checkout --force FETCH_HEAD
fi

# Prefer a Tcl/Tk-capable Python 3.10-3.12 so both CLI `inference.py` and
# Tkinter `real-time-gui.py` work cleanly with prebuilt binary wheels.
PY_SPEC="3.12"
if [ -x "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12" ]; then
  PY_SPEC="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  PY_SPEC="$(command -v python3.11)"
elif command -v python3.10 >/dev/null 2>&1; then
  PY_SPEC="$(command -v python3.10)"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$UV_BIN" venv --python "$PY_SPEC" "$VENV_DIR"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/cs-uv-cache}"

# Install stable release wheels (avoiding upstream's broken inline `--pre`
# nightly lines in requirements-mac.txt while keeping exact library compatibility).
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
  "torch>=2.4,<2.7" \
  "torchaudio>=2.4,<2.7" \
  "torchvision>=0.19,<0.22" \
  "numpy>=1.26.4,<2.0" \
  "scipy>=1.13.1" \
  "librosa>=0.10.2" \
  "soundfile>=0.12.1" \
  "sounddevice>=0.5.0" \
  "huggingface-hub>=0.28.1" \
  "accelerate" \
  "munch>=4.0.0" \
  "einops>=0.8.0" \
  "descript-audio-codec>=1.0.0" \
  "pydub>=0.25.1" \
  "jiwer>=3.0.3" \
  "transformers>=4.46.3,<5.0" \
  "FreeSimpleGUI>=5.1.1" \
  "pyyaml" \
  "python-dotenv" \
  "hydra-core>=1.3.2"

"$VENV_DIR/bin/python" -c "import torch, torchaudio, librosa, transformers, yaml, FreeSimpleGUI; print('✓ seed-vc imports OK | mps:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
echo "✓ $VENV_DIR/bin/python ($REPO_DIR/inference.py)"
