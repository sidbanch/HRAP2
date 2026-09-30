import numpy as np
import pytest

from hrap.engine.sim import run
from hrap.engine.sizing import SizingTargets, size_motor
from hrap.io.config import bundled_motor, resolve

PSI = 6894.757


def test_sized_motor_starts_at_the_target_chamber_pressure():
    cfg = bundled_motor("example_98mm")
    cfg.update(reg_model="Shifting OF", prop_a=0.198, prop_n=0.325)
    z = size_motor(cfg, SizingTargets(P_cmbr=450 * PSI, burn_time=6.0, OF=6.0, port_D=0.06))
    # Scale Cd by the exact hole ratio so rounding the hole count doesn't hide a mismatch.
    cfg.update(noz_thrt=z.throat_D, noz_thrt_unit="m", noz_def="Nozzle Expansion Ratio", noz_ex=z.ER,
               inj_Cd=cfg["inj_Cd"] * z.holes / cfg["inj_N"], grn_ID=0.06, grn_ID_unit="m",
               grn_L=z.grain_L, grn_L_unit="m", grn_OD=0.1, grn_OD_unit="m")
    s, x = resolve(cfg)
    x, o = run(s, x)
    i = int(np.argmax(o.P_cmbr))  # end of chamber filling; the tank has only cooled slightly
    assert o.P_cmbr[i] == pytest.approx(450 * PSI, rel=0.03)
    assert o.F_thr[i] == pytest.approx(z.thrust, rel=0.03)
    assert o.OF[i] == pytest.approx(6.0, rel=0.03)
