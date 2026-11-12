"""Flying HRAP motors on an OpenRocket rocket, in the user's own OpenRocket install.

The flights run in a separate process (hrap.io.openrocket_worker), which loads OpenRocket's Java simulator. This
module is what the app needs around that: finding the install, reading a .ork file's own simulations, and the job
and replies passed to and from the worker.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

MARK = "HRAP-FLIGHT "  # starts each reply line from the worker, as OpenRocket also logs to the same output
OLDEST = (24, 12)  # the first OpenRocket whose simulator starts without its window


@dataclass(frozen=True)
class OpenRocket:
    folder: Path
    jar: Path
    jvm: Path  # the Java it ships with
    version: str

    @property
    def supported(self) -> bool:
        return tuple(int(n) for n in re.findall(r"\d+", self.version)[:2]) >= OLDEST


def _install_folders() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        return [Path("/Applications/OpenRocket.app"), home / "Applications/OpenRocket.app"]
    if sys.platform == "win32":
        roots = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
        return [Path(r) / "OpenRocket" for r in roots if r] + [Path(r) / "Programs/OpenRocket" for r in roots if r]
    return [Path("/opt/OpenRocket"), home / "OpenRocket", home / ".local/share/OpenRocket"]


def find_openrocket(folder: str | Path | None = None) -> OpenRocket | None:
    """The OpenRocket installed in this folder, or in the usual places when none is given."""
    jvm_name = {"darwin": "libjvm.dylib", "win32": "jvm.dll"}.get(sys.platform, "libjvm.so")
    for root in [Path(folder)] if folder else _install_folders():
        if not root.is_dir():
            continue
        jar = next(iter(sorted(root.rglob("OpenRocket*.jar"), reverse=True)), None)
        jvm = next(iter(root.rglob(jvm_name)), None)
        if jar is None or jvm is None:
            continue
        try:
            with zipfile.ZipFile(jar) as z:
                version = re.search(r"^build\.version=(\S+)", z.read("build.properties").decode(), re.M)
        except (OSError, KeyError, zipfile.BadZipFile):
            version = None
        named = re.search(r"OpenRocket-(\d[\d.]*\d)", jar.name)
        return OpenRocket(root, jar, jvm, version[1] if version else named[1] if named else "")
    return None


@dataclass(frozen=True)
class OrkSimulation:
    """One of a .ork file's own simulations: its launch conditions, and what OpenRocket last got from it."""
    name: str
    rod_length: float  # m
    rod_angle: float  # degrees from vertical
    wind: float  # m/s
    multi_level_wind: bool
    motor: str  # the motor it was saved with
    apogee: float | None  # m, as saved


@dataclass(frozen=True)
class Ork:
    path: Path
    name: str
    simulations: tuple[OrkSimulation, ...]


def read_ork(path: str | Path) -> Ork:
    """A .ork file's rocket name and simulations. Raises ValueError if it isn't an OpenRocket file."""
    path = Path(path)
    try:
        raw = path.read_bytes()
        if raw[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                raw = z.read(next(n for n in z.namelist() if n.endswith(".ork")))
        elif raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        root = ElementTree.fromstring(raw)
    except (OSError, StopIteration, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise ValueError(f"Couldn't read {path.name} as an OpenRocket file: {exc}") from None
    if root.tag != "openrocket":
        raise ValueError(f"{path.name} isn't an OpenRocket file.")
    motors = {m.get("configid"): m.findtext("designation") or "" for m in root.iter("motor")}

    def number(sim, name: str) -> float:
        try:
            return float(sim.findtext(f"conditions/{name}") or 0.0)
        except ValueError:
            return 0.0

    simulations = []
    for sim in root.iterfind("simulations/simulation"):
        data = sim.find("flightdata")
        saved = data.get("maxaltitude") if data is not None else None
        simulations.append(OrkSimulation(
            sim.findtext("name") or f"Simulation {len(simulations) + 1}", number(sim, "launchrodlength"),
            number(sim, "launchrodangle"), number(sim, "windaverage"),
            sim.findtext("conditions/windmodeltype") == "MultiLevel",
            motors.get(sim.findtext("conditions/configid"), ""), float(saved) if saved else None))
    return Ork(path, root.findtext("rocket/name") or path.stem, tuple(simulations))


def worker_command() -> list[str]:
    return [sys.executable, "-m", "hrap.io.openrocket_worker"]


def job(openrocket: OpenRocket, ork: Path, simulation: int, motor_files: list[Path]) -> bytes:
    """What the worker reads: fly each motor file (RSE or ENG) on the rocket under one of its simulations' conditions."""
    return json.dumps({"jar": str(openrocket.jar), "jvm": str(openrocket.jvm), "ork": str(ork), "simulation": simulation,
                       "motors": [str(p) for p in motor_files]}).encode()


def reply(line: str) -> dict | None:
    """A reply from one line of the worker's output, or None for anything else on it.

    Per motor: {"motor": index, then either "error" or apogee, max_speed, max_mach, max_accel, rail_speed, stability,
    time_to_apogee (SI; stability in calibers as it leaves the rail), warnings, and t / altitude / speed curves up to
    a little past apogee}. {"error": ...} alone means nothing could be flown.
    """
    if not line.startswith(MARK):
        return None
    try:
        return json.loads(line[len(MARK):])
    except ValueError:
        return None
