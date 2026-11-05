from pathlib import Path

import numpy as np
import pytest

from hrap.io.measured import load_measured_csv
from hrap.units import to_si

IRL = Path(__file__).resolve().parent / "fixtures" / "hps01_irl_sample.csv"


def test_irl_csv_uses_tplus_ms_and_weight_lbs():
    curve = load_measured_csv(IRL)
    assert curve.label == "hps01_irl_sample"
    assert abs(curve.t[0] - (-0.114)) < 1e-9
    assert abs(curve.t[2] - 0.0) < 1e-12
    assert curve.F_thr is not None and curve.P_tnk is not None
    assert abs(curve.F_thr[2] - to_si(214.86, "lbf", "force")) < 1e-6
    assert abs(curve.P_tnk[2] - to_si(431.973, "psi", "pressure")) < 1e-3
    assert float(np.max(curve.F_thr)) == pytest.approx(to_si(419.411, "lbf", "force"))


def test_hrap_export_csv_stays_in_si(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("t,F_thr,P_tnk,P_cmbr\n0,0,5e6,1e5\n0.5,1800,4.8e6,2e6\n1.0,900,3e6,1.2e6\n", encoding="utf-8")
    curve = load_measured_csv(path)
    assert list(curve.t) == [0.0, 0.5, 1.0]
    assert curve.F_thr[1] == pytest.approx(1800.0)
    assert curve.P_tnk[0] == pytest.approx(5e6)
    assert curve.P_cmbr[2] == pytest.approx(1.2e6)


def test_full_hps01_irl_csv_if_present():
    path = Path("HCAT motor files/Development/HPS01-2/HPS01-2 IRL Data.csv")
    if not path.is_file():
        pytest.skip("local IRL CSV is not in the tree")
    curve = load_measured_csv(path)
    assert curve.F_thr is not None and curve.P_tnk is not None
    assert curve.t[0] < 0.0 <= curve.t[-1]
    assert float(np.max(curve.F_thr)) > to_si(400.0, "lbf", "force")


def test_missing_time_column_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Weight_Lbs\n10\n20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="time column"):
        load_measured_csv(path)
