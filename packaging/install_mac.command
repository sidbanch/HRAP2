#!/bin/bash
# Installs HRAP as an app: double-click this file, or run
#   curl -fsSL https://raw.githubusercontent.com/sidbanch/HRAP2/main/packaging/install_mac.command | bash
# HRAP.app goes in ~/Applications and keeps itself up to date from GitHub.
# Running it again repairs or reinstalls; motor files are never touched.
set -euo pipefail

REPO="${HRAP_REPO:-sidbanch/HRAP2}"
BRANCH="${HRAP_BRANCH:-main}"
ROOT="${HRAP_ROOT:-$HOME/Library/Application Support/HRAP}"
APP="${HRAP_APP:-$HOME/Applications/HRAP.app}"

echo "Installing HRAP from $REPO ($BRANCH)…"

# uv manages Python and packages; it installs to ~/.local/bin without admin rights.
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
UV="$(command -v uv)"

mkdir -p "$ROOT"
[[ -x "$ROOT/venv/bin/python" ]] || "$UV" venv --python 3.12 "$ROOT/venv"
PY="$ROOT/venv/bin/python"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if [[ -n "${HRAP_SOURCE:-}" ]]; then  # install a local checkout instead (testing)
    SRC="$HRAP_SOURCE"
    SHA="local"
else
    SHA="$(curl -fsSL "https://api.github.com/repos/$REPO/commits/$BRANCH" | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["sha"])')"
    curl -fsSL "https://codeload.github.com/$REPO/zip/$SHA" -o "$TMP/src.zip"
    unzip -q "$TMP/src.zip" -d "$TMP"
    SRC="$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
fi
echo "Installing packages (the first time takes a minute)…"
"$PY" "$SRC/src/hrap/update.py" "$ROOT" "$REPO" "$BRANCH" --uv "$UV" --source "$SRC" --sha "$SHA"

# The app bundle only launches the installed Python; updates replace the package, not the bundle.
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>HRAP</string>
  <key>CFBundleDisplayName</key><string>HRAP</string>
  <key>CFBundleIdentifier</key><string>com.hcat.hrap</string>
  <key>CFBundleExecutable</key><string>HRAP</string>
  <key>CFBundleIconFile</key><string>HRAP</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
cat > "$APP/Contents/MacOS/HRAP" <<LAUNCH
#!/bin/bash
exec "$PY" -m hrap "\$@"
LAUNCH
chmod +x "$APP/Contents/MacOS/HRAP"
sips -s format icns "$SRC/src/hrap/resources/icon.ico" --out "$APP/Contents/Resources/HRAP.icns" >/dev/null 2>&1 || true
"$PY" - "$ROOT" "$APP" <<'INFO'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / "install.json"
info = json.loads(path.read_text()); info["app"] = sys.argv[2]; path.write_text(json.dumps(info, indent=2) + "\n")
INFO

echo "Done. HRAP is in $APP (drag it to the Dock to keep it there)."
[[ -n "${HRAP_NO_OPEN:-}" ]] || open "$APP"
