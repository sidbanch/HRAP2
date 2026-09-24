#!/bin/bash
# Double-click in Finder to start HRAP on macOS.
set -e
cd "$(dirname "$0")"

launch_failed() {
    echo
    echo "HRAP could not start. See the error above."
    if [[ -t 0 ]]; then read -r -p "Press Return to close…"; fi
    exit 1
}
trap launch_failed ERR
caffeinate -i -w "$$" &

if [[ ! -x .venv/bin/python ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 .venv
    else
        python3 -c 'import sys; assert sys.version_info >= (3, 10), "HRAP requires Python 3.10 or later"'
        python3 -m venv .venv
    fi
fi

.venv/bin/python -c 'import sys; sys.exit("HRAP requires Python 3.10 or later. Rename the existing .venv folder and relaunch to create a supported environment." if sys.version_info < (3, 10) else 0)'

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if ! .venv/bin/python -c 'import hrap.gui.main, CoolProp' 2>/dev/null; then
    echo "Installing HRAP dependencies…"
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python .venv/bin/python -e .
    else
        .venv/bin/python -m pip install -e .
    fi
fi

.venv/bin/python -m hrap
