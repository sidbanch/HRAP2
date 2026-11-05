import math
import sys

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from hrap.gui.main import MainWindow
from hrap.units import to_si


def test_mass_tab_sets_tank_and_tca_length_and_shows_the_motor(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    prefs = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    win = MainWindow(prefs)
    try:
        design_tank, design_tca = win._design_lengths()
        assert win.mass_preview.viz._model.name
        assert abs(win.tnk_L.si("length") - design_tank) < 1e-4
        assert abs(win.cmbr_L.si("length") - design_tca) < 1e-4

        before = win._form_layout()
        pre, post = before.grn0 - before.plate1, before.x_case - before.grn1
        win.tnk_L.set_si(design_tank + 0.15, "length")
        win.cmbr_L.set_si(design_tca + 0.20, "length")
        win._on_length_edited()
        lay = win._form_layout()
        assert win._tank_len_custom and win._tca_len_custom
        assert abs(lay.tnk_L - (design_tank + 0.15)) < 1e-6
        assert abs(lay.cmbr_L - (design_tca + 0.20)) < 1e-6
        assert abs((lay.grn0 - lay.plate1) - (pre + 0.10)) < 1e-6
        assert abs((lay.x_case - lay.grn1) - (post + 0.10)) < 1e-6
        assert abs(win.mass_preview.viz._model.tnk_L - lay.tnk_L) < 1e-9

        cfg = win._form_to_cfg()
        assert abs(to_si(cfg["tnk_L"], cfg["tnk_L_unit"], "length") - lay.tnk_L) < 1e-6
        assert abs(to_si(cfg["cmbr_L"], cfg["cmbr_L_unit"], "length") - lay.cmbr_L) < 1e-6

        short = design_tca * 0.5
        win.cmbr_L.set_si(short, "length")
        win._on_length_edited()
        assert win._tca_len_custom
        assert abs(win._form_layout().cmbr_L - short) < 1e-6
        assert abs(win.cmbr_L.si("length") - short) < 1e-4
    finally:
        win._session_timer.stop()
        win.close()


def test_motor_tab_internal_length_sets_volume_and_cd_can_replace_cda(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    prefs = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    win = MainWindow(prefs)
    try:
        sp = win.sizing_page
        sp.inj_type.setCurrentText("Holes")
        sp.tank_D.set_si(0.10, "length")
        sp.tank_L.set_si(0.50, "length")
        sp._on_tank_length()
        length, diameter, volume = sp.tank_geometry()
        expect = length * 0.25 * math.pi * diameter ** 2
        assert abs(length - 0.50) < 1e-5
        assert abs(diameter - 0.10) < 1e-6
        assert abs(volume - expect) < 1e-9
        assert abs(sp.tank_V.si("volume") - expect) < 1e-6

        sp.cd_as.setCurrentText("Cd")
        sp.inj_Cd.setValue(0.65)
        assert sp.cd_as.currentText() == "Cd"
        assert abs(sp.cd() - 0.65) < 1e-9
        cfg = sp.motor_cfg()
        assert cfg["inj_cd_as"] == "Cd"
        assert abs(cfg["inj_Cd"] - 0.65) < 1e-9
        assert abs(to_si(cfg["tnk_V"], cfg["tnk_V_unit"], "volume") - volume) < 1e-6
    finally:
        win._session_timer.stop()
        win.close()
