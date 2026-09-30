"""Self-updating installs: the installer puts HRAP in its own folder, and the app updates it from GitHub.

An install folder holds ``install.json`` (repo, branch, installed commit, uv's path) beside the ``venv``
the app runs in. Running from a source checkout instead, there's no install.json and nothing here acts;
update that with git.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

INFO = "install.json"


@dataclass(frozen=True)
class Commit:
    sha: str
    message: str  # first line
    date: str     # ISO 8601


def install_root() -> Path | None:
    """The install folder this app runs from, or None for a source checkout."""
    root = Path(sys.prefix).parent  # <root>/venv
    return root if (root / INFO).is_file() else None


def read_info(root: Path) -> dict:
    return json.loads((root / INFO).read_text())


def write_info(root: Path, info: dict):
    tmp = root / f"{INFO}.tmp"
    tmp.write_text(json.dumps(info, indent=2) + "\n")
    tmp.replace(root / INFO)


def latest(repo: str, branch: str, timeout: float = 8.0) -> Commit:
    """The newest commit on a branch, from GitHub's API (no login; 60 checks an hour per network)."""
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/commits/{branch}",
                                 headers={"Accept": "application/vnd.github+json", "User-Agent": "HRAP-updater"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return Commit(data["sha"], data["commit"]["message"].splitlines()[0], data["commit"]["committer"]["date"])


def install(root: Path, repo: str, sha: str, uv: str | None = None, source: Path | None = None):
    """Install one commit of the repo into the install's venv. ``source`` installs a local folder instead (testing)."""
    python = root / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    with tempfile.TemporaryDirectory() as tmp:
        if source is None:
            archive = Path(tmp) / "src.zip"
            req = urllib.request.Request(f"https://codeload.github.com/{repo}/zip/{sha}", headers={"User-Agent": "HRAP-updater"})
            with urllib.request.urlopen(req, timeout=60) as r, open(archive, "wb") as f:
                shutil.copyfileobj(r, f)
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmp)
            source = next(p for p in Path(tmp).iterdir() if p.is_dir())
        uv = uv or shutil.which("uv")
        cmd = ([uv, "pip", "install", "--python", str(python), "--reinstall-package", "hrap", str(source)] if uv
               else [str(python), "-m", "pip", "install", "--force-reinstall", "--no-deps", str(source)])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not uv:  # pip's --no-deps skips new dependencies; pick those up separately
            subprocess.run([str(python), "-m", "pip", "install", str(source)], check=True, capture_output=True, text=True)


def update(root: Path, commit: Commit):
    info = read_info(root)
    install(root, info["repo"], commit.sha, info.get("uv"))
    info.update(sha=commit.sha, message=commit.message, date=commit.date)
    write_info(root, info)


def main(argv: list[str] | None = None) -> int:
    """First install, used by the installers: ``python update.py <root> <repo> <branch> [--source DIR --sha SHA]``.

    It only needs the standard library, so the installers run it straight from the downloaded source.
    """
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("root", type=Path)
    p.add_argument("repo")
    p.add_argument("branch")
    p.add_argument("--uv")
    p.add_argument("--source", type=Path, help="install this already-downloaded folder instead of downloading")
    p.add_argument("--sha", default="", help="the commit --source is (so the first update check knows)")
    a = p.parse_args(argv)
    commit = Commit(a.sha or "local", "", "") if a.source else latest(a.repo, a.branch)
    install(a.root, a.repo, commit.sha, a.uv, a.source)
    write_info(a.root, {"repo": a.repo, "branch": a.branch, "sha": commit.sha, "message": commit.message,
                        "date": commit.date, "uv": a.uv})
    print(f"Installed {a.repo}@{a.branch} ({commit.sha[:7]}): {commit.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
