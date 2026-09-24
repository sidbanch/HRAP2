import time

import pytest


@pytest.fixture(scope="module")
def app():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance() or QApplication([])
        yield application


@pytest.fixture
def window(app):
    from hrap.gui.main import MainWindow

    win = MainWindow()
    win.tmax.setValue(.02)
    win.show()
    yield win
    wait_for_run(win)
    win.close()
    from PySide6.QtCore import QCoreApplication, QEvent
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def wait_for_run(window):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + 5
    while window._thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(10)
    assert window._thread is None


def test_close_waits_for_worker(window):
    window._run()
    assert not window._form.isEnabled()
    assert not window._file_menu.isEnabled()
    assert not window._examples_menu.isEnabled()
    assert not window.close()
    assert window.isVisible()
    wait_for_run(window)
    assert not window.isVisible()
    assert window._worker is None


def test_editing_inputs_invalidates_completed_results(window):
    window._run()
    wait_for_run(window)
    assert window._output is not None
    assert window._form.isEnabled()
    assert window._result_cfg["mtr_nm"] == window._settings.mtr_nm
    window.grn_L.spin.setValue(window.grn_L.spin.value() + 1)
    assert window._output is None
    assert window._result_cfg is None
    assert not window.summary.toPlainText()


def test_missing_coolprop_reports_error_and_reenables_run(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    def unavailable(*args):
        raise ImportError("CoolProp missing")

    messages = []
    monkeypatch.setattr("hrap.advanced.fluid.coolprop_sat", unavailable)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: messages.append(args[-1]))
    window.adv_on.setChecked(True)
    window.ox_fluid.setCurrentText("NitrousOxide (CoolProp)")
    window._run()
    wait_for_run(window)
    assert window.run_btn.isEnabled()
    assert window._output is None
    assert len(messages) == 1 and "CoolProp missing" in messages[0]


def test_loading_restores_manufacturer(window):
    from hrap.io.config import default_cfg

    cfg = default_cfg()
    cfg["mfg"] = "Saved manufacturer"
    window.mfg.setText("Previous manufacturer")
    window._cfg_to_form(cfg)
    assert window._form_to_cfg()["mfg"] == "Saved manufacturer"


def test_display_units_update_results_without_changing_simulation(window, tmp_path):
    import numpy as np
    from PySide6.QtCore import QPointF, QSettings, Qt
    from PySide6.QtWidgets import QApplication
    from hrap.units import DisplayUnits

    window._prefs = QSettings(str(tmp_path / "units.ini"), QSettings.Format.IniFormat)
    window._run()
    wait_for_run(window)
    output = window._output
    pressure = output.P_tnk.copy()
    config = window._form_to_cfg()
    for units, pressure_scale in (
        (DisplayUnits(pressure="bar", length="mm"), 1e-5),
        (DisplayUnits(pressure="psi", length="in", mass="lbm", force="lbf", temperature="F"), 14.696 / 101325),
    ):
        window._set_display_units(units)
        QApplication.processEvents()
        assert window._output is output
        assert window._form_to_cfg() == config
        np.testing.assert_array_equal(output.P_tnk, pressure)
        plots = window._plots
        assert list(plots) == ["force", "pressure", "mass_flow"]
        np.testing.assert_allclose(plots["pressure"].listDataItems()[0].yData, pressure * pressure_scale)
        np.testing.assert_allclose(plots["mass_flow"].listDataItems()[0].yData,
                                   output.mdot_o * units.value(1, "mass_flow"))
        assert plots["pressure"].getAxis("left").labelUnits == units.pressure
        assert units.pressure in window.summary.toPlainText()
        assert units.force in window.summary.toPlainText()
        view = window._motor_view(1)
        assert view.overlay_tank[1] == units.text(pressure[1], "pressure")
        assert units.unit("mass_flow") in view.inj_lines[4]
        assert view.display_units.length == units.length
        window.plot.setCurrentWidget(window._plot_widgets["pressure"])
        QApplication.processEvents()
        point = plots["pressure"].vb.mapViewToScene(QPointF(output.t[1], pressure[1] * pressure_scale))
        window._mouse_moved("pressure", point)
        assert units.pressure in window.plot_readout.text()
        assert window._hover_index is not None
        assert window._prefs.value("displayUnits/pressure") == units.pressure
    plots["force"].setXRange(.002, .008, padding=0)
    QApplication.processEvents()
    np.testing.assert_allclose(plots["pressure"].vb.viewRange()[0], [.002, .008], atol=1e-6)
    window._set_display_units(DisplayUnits(pressure="bar"))
    QApplication.processEvents()
    assert window.plot.currentWidget() is window._plot_widgets["pressure"]
    np.testing.assert_allclose(window._plots["pressure"].vb.viewRange()[0], [.002, .008], atol=1e-6)
    window.trace_list.findItems("O/F", Qt.MatchFlag.MatchExactly)[0].setCheckState(Qt.CheckState.Checked)
    np.testing.assert_allclose(window._plots["ratio"].vb.viewRange()[0], [.002, .008], atol=1e-6)
    for i in range(window.trace_list.count()):
        window.trace_list.item(i).setCheckState(Qt.CheckState.Unchecked)
    assert not window._plots
    window.trace_list.item(1).setCheckState(Qt.CheckState.Checked)
    assert list(window._plots) == ["pressure"]


def test_small_metric_ruler_ticks_are_distinct():
    from hrap.gui.viz import _tick_label

    assert _tick_label(.05, .05) == "0.05"
    assert _tick_label(.10, .05) == "0.10"
