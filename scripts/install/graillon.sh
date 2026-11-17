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

# Install Auburn Sounds Graillon 3 Free Edition (VST3) into user plugin space.
#
# Extracts the signed Universal VST3 bundle (`Auburn Sounds Graillon 3.vst3`) on
# macOS via `pkgutil --expand-full` (no root/sudo needed) and copies the native
# Linux VST3/LV2 bundles on Linux (`~/.vst3`, `~/.lv2`).
set -euo pipefail

VERSION="${GRAILLON_VERSION:-3.2}"
URL="https://www.auburnsounds.com/downloads/Graillon-FREE-${VERSION}.zip"
BUNDLE_NAME="Auburn Sounds Graillon 3.vst3"

if [ "$(uname -s)" = "Darwin" ]; then
  VST3_DIR="${CREATIVE_STACK_VST3_DIR:-$HOME/Library/Audio/Plug-Ins/VST3}"
else
  VST3_DIR="${CREATIVE_STACK_VST3_DIR:-$HOME/.vst3}"
fi

if [ -d "$VST3_DIR/$BUNDLE_NAME" ]; then
  echo "✓ Graillon $VERSION already installed at $VST3_DIR/$BUNDLE_NAME"
  exit 0
fi

echo "▸ Installing Auburn Sounds Graillon $VERSION Free -> $VST3_DIR/$BUNDLE_NAME"

for tool in curl unzip; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "❌ $tool is required to install Graillon" >&2
    exit 1
  }
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/cs-graillon.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

ARCHIVE="$WORK/Graillon-FREE-${VERSION}.zip"
curl -fsSL --retry 3 -o "$ARCHIVE" "$URL"
unzip -q "$ARCHIVE" -d "$WORK/unpacked"

mkdir -p "$VST3_DIR"

if [ "$(uname -s)" = "Darwin" ]; then
  PKG_FILE="$(find "$WORK/unpacked" -name "Graillon-3-FREE-*.pkg" | head -1)"
  [ -n "$PKG_FILE" ] || { echo "❌ no macOS .pkg found in $URL" >&2; exit 1; }
  pkgutil --expand-full "$PKG_FILE" "$WORK/pkg"
  SRC_BUNDLE="$(find "$WORK/pkg" -type d -name "$BUNDLE_NAME" | head -1)"
  [ -n "$SRC_BUNDLE" ] || { echo "❌ $BUNDLE_NAME not found in expanded pkg" >&2; exit 1; }
  cp -R "$SRC_BUNDLE" "$VST3_DIR/$BUNDLE_NAME"
  xattr -dr com.apple.quarantine "$VST3_DIR/$BUNDLE_NAME" 2>/dev/null || true
else
  SRC_BUNDLE="$(find "$WORK/unpacked" -type d -name "$BUNDLE_NAME" | head -1)"
  [ -n "$SRC_BUNDLE" ] || { echo "❌ Linux $BUNDLE_NAME not found in archive" >&2; exit 1; }
  cp -R "$SRC_BUNDLE" "$VST3_DIR/$BUNDLE_NAME"
  LV2_SRC="$(find "$WORK/unpacked" -type d -name "Auburn Sounds Graillon 3.lv2" | head -1)"
  if [ -n "$LV2_SRC" ]; then
    mkdir -p "$HOME/.lv2"
    cp -R "$LV2_SRC" "$HOME/.lv2/Auburn Sounds Graillon 3.lv2"
  fi
fi

[ -d "$VST3_DIR/$BUNDLE_NAME" ] || { echo "❌ installation failed: $VST3_DIR/$BUNDLE_NAME missing" >&2; exit 1; }
echo "✓ $VST3_DIR/$BUNDLE_NAME"
