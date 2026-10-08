"""Runs are kept as one JSON file each, in a runs folder inside the motor's export folder (beside its file).

A run's file holds its settings and headline numbers, not its curves; those are simulated again when it's shown.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from PySide6.QtWidgets import QMessageBox

from hrap import APP_NAME


def run_files(folders: Iterable[Path], kind: str) -> list[Path]:
    """The saved runs of one kind ("sim" or "sweep") in these folders."""
    return [p for folder in dict.fromkeys(folders) for p in folder.glob(f"{kind} *.json")]


def new_run_file(folder: Path, kind: str) -> tuple[Path, int]:
    """The next free numbered file for a run of this kind, and its number."""
    used = [int(m[1]) for p in folder.glob(f"{kind} *.json") if (m := re.fullmatch(rf"{kind} (\d+)\.json", p.name))]
    number = max(used, default=0) + 1
    return folder / f"{kind} {number}.json", number


def read_run(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("motor"), dict) else None


def write_run(path: Path, data: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=float) + "\n")
        tmp.replace(path)
    except OSError as exc:
        QMessageBox.warning(None, APP_NAME, f"Couldn't save the run to {path}. It stays listed only until the list "
                            f"next reloads (opening, closing or saving a motor).\n\n{exc}")


def delete_run(path: Path):
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
