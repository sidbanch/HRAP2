"""NAR / NFPA impulse class. Port of MATLAB util/impulse.m."""
from __future__ import annotations


def impulse_class(total_impulse: float) -> tuple[str | float, float]:
    """Return (letter, percent through class) for total impulse in N*s."""
    impulse = float(total_impulse)
    bands = [
        ("A", 0.0, 1.25),
        ("B", 1.25, 5.0),
        ("C", 5.0, 10.0),
        ("D", 10.0, 20.0),
        ("E", 20.0, 40.0),
        ("F", 40.0, 80.0),
        ("G", 80.0, 160.0),
        ("H", 160.0, 320.0),
        ("I", 320.0, 640.0),
        ("J", 640.0, 1280.0),
        ("K", 1280.0, 2560.0),
        ("L", 2560.0, 5120.0),
        ("M", 5120.0, 10240.0),
        ("N", 10240.0, 20480.0),
        ("O", 20480.0, 40960.0),
        ("P", 40960.0, 81920.0),
        ("Q", 81920.0, 163840.0),
        ("R", 163840.0, 327680.0),
        ("S", 327680.0, 655369.0),  # MATLAB uses 655369, not 655360
        ("T", 655360.0, 1310720.0),
    ]
    if impulse <= 1.25:
        return "A", 100.0 * (impulse - 0.0) / 1.25
    # MATLAB util/impulse.m uses these denominators (B is 2.5, not 5-1.25=3.75).
    denoms = {
        "B": 2.5, "C": 5.0, "D": 10.0, "E": 20.0, "F": 40.0, "G": 80.0,
        "H": 160.0, "I": 320.0, "J": 640.0, "K": 1280.0, "L": 2560.0,
        "M": 5120.0, "N": 10240.0, "O": 20480.0, "P": 40960.0, "Q": 81920.0,
        "R": 163840.0, "S": 327680.0, "T": 655360.0,
    }
    for letter, lo, hi in bands[1:]:
        if impulse > lo and impulse <= hi:
            return letter, 100.0 * (impulse - lo) / denoms[letter]
    return float("nan"), float("nan")
