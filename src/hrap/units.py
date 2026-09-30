"""SI internal units with MATLAB HRAP conversion factors."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Affine:
    offset: float
    scale: float

    def to_si(self, x: float) -> float:
        return self.scale * x + self.offset

    def from_si(self, y: float) -> float:
        return (y - self.offset) / self.scale


# MATLAB-compatible linear factors (see HRAP.mlapp createComponents / RunSimulation)
LENGTH = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "in": 0.0254,
    "ft": 0.3048,
}
AREA = {
    "mm^2": 1e-6,
    "cm^2": 1e-4,
    "m^2": 1.0,
    "in^2": 0.0254 ** 2,
}
VOLUME = {
    "in^3": 0.0254 ** 3,
    "ft^3": 0.3048 ** 3,
    "cm^3": 0.01 ** 3,
    "cc": 0.01 ** 3,
    "L": 0.001,
    "Gal": 0.00378541,
    "m^3": 1.0,
}
PRESSURE = {
    "psi": 101325.0 / 14.696,
    "psf": 101325.0 / 14.696 * 144.0,
    "atm": 101325.0,
    "MPa": 1_000_000.0,
    "kPa": 1000.0,
    "Bar": 100_000.0,
    "bar": 100_000.0,
    "Pa": 1.0,
}
MASS = {
    "lbm": 0.453592,
    "oz": 0.0283495,
    "g": 0.001,
    "kg": 1.0,
}
DENSITY = {
    "lb/in^3": 1.0 / (2.205 * 0.0254 ** 3),
    "lb/ft^3": 1.0 / (2.205 * 0.3048 ** 3),
    "g/cm^3": 1000.0,
    "kg/m^3": 1.0,
}
FORCE = {
    "N": 1.0,
    "kN": 1000.0,
    "lbf": 4.4482216152605,
}
TEMPERATURE = {
    "K": Affine(0.0, 1.0),
    "C": Affine(273.15, 1.0),
    "F": Affine(273.15 - 32.0 * 5.0 / 9.0, 5.0 / 9.0),
    "R": Affine(0.0, 1.0 / 1.8),
}

UNIT_GROUPS = {
    "length": LENGTH,
    "area": AREA,
    "volume": VOLUME,
    "pressure": PRESSURE,
    "mass": MASS,
    "density": DENSITY,
    "force": FORCE,
    "temperature": TEMPERATURE,
}

# Display aliases used in the MATLAB GUI
LENGTH_ITEMS = ["mm", "cm", "m", "in", "ft"]
AREA_ITEMS = ["in^2", "mm^2", "cm^2", "m^2"]
VOLUME_ITEMS = ["cm^3", "L", "in^3", "ft^3", "Gal", "m^3"]
PRESSURE_ITEMS = ["psi", "atm", "kPa", "Bar", "MPa", "Pa", "psf"]
MASS_ITEMS = ["kg", "g", "lbm", "oz"]
TEMP_ITEMS = ["K", "C", "F", "R"]
DENSITY_ITEMS = ["kg/m^3", "g/cm^3", "lb/in^3", "lb/ft^3"]
FORCE_ITEMS = ["N", "kN", "lbf"]


def to_si(value: float, unit: str, group: str | None = None) -> float:
    table = UNIT_GROUPS[group] if group else _lookup_group(unit)
    factor = table[unit]
    if isinstance(factor, Affine):
        return factor.to_si(value)
    return value * factor


def from_si(value: float, unit: str, group: str | None = None) -> float:
    table = UNIT_GROUPS[group] if group else _lookup_group(unit)
    factor = table[unit]
    if isinstance(factor, Affine):
        return factor.from_si(value)
    return value / factor


def _lookup_group(unit: str) -> dict:
    for table in UNIT_GROUPS.values():
        if unit in table:
            return table
    raise KeyError(f"Unknown unit {unit!r}")


def compatible_units(old_unit: str, new_unit: str) -> bool:
    """True if both labels belong to the same quantity (length, pressure, ...)."""
    if old_unit == new_unit:
        return True
    try:
        return _lookup_group(old_unit) is _lookup_group(new_unit)
    except KeyError:
        return False


def convert(value: float, old_unit: str, new_unit: str, group: str | None = None) -> float:
    """Rewrite a display value so the underlying SI quantity is unchanged."""
    if old_unit == new_unit:
        return float(value)
    return from_si(to_si(value, old_unit, group), new_unit, group)


# Convenience SI constants
_mm = 0.001
_cm = 0.01
_in = 0.0254
_ft = 0.3048
_atm = 101325.0
_psi = 101325.0 / 14.696


@dataclass(frozen=True)
class DisplayUnits:
    """Presentation choices only; engine and exported numeric data stay in SI."""

    pressure: str = "psi"
    length: str = "in"
    mass: str = "kg"
    force: str = "N"
    volume: str = "L"
    temperature: str = "K"
    speed: str = "m/s"

    def unit(self, quantity: str) -> str:
        if quantity == "mass_flow":
            return f"{self.mass}/s"
        if quantity == "impulse":
            return f"{self.force}·s"
        if quantity == "area":
            return f"{self.length}²"
        if quantity == "ratio":
            return ""
        return getattr(self, quantity)

    def value(self, si: float, quantity: str) -> float:
        if quantity == "ratio":
            return si
        if quantity == "mass_flow":
            return from_si(si, self.mass, "mass")
        if quantity == "impulse":
            return from_si(si, self.force, "force")
        if quantity == "area":
            return si / LENGTH[self.length] ** 2
        if quantity == "speed":
            return from_si(si, self.speed.removesuffix("/s"), "length")
        return from_si(si, self.unit(quantity), quantity)

    def text(self, si: float, quantity: str, precision: int = 4) -> str:
        value = self.value(si, quantity)
        # Whole numbers once they need more digits than the precision, instead of 1.041e+04.
        number = f"{value:.0f}" if abs(value) >= 10 ** precision else f"{value:.{precision}g}"
        return f"{number} {self.unit(quantity)}".strip()
