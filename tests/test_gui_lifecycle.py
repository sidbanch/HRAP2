import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def app():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance() or QApplication([])
        yield application


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from hrap.gui.main import MainWindow

    # Closing a tab with unsaved edits asks first; don't let the question block the tests.
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Discard)
    win = MainWindow(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    win.sim.tmax.setValue(.02)
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
    assert not window.sim.form.isEnabled()
    assert not window._file_menu.isEnabled()
    assert not window._examples_menu.isEnabled()
    assert not window.close()
    assert window.isVisible()
    wait_for_run(window)
    assert not window.isVisible()
    assert window._worker is None


def test_quitting_keeps_unsaved_edits_for_next_time(window, tmp_path):
    from PySide6.QtCore import QSettings

    from hrap.gui.main import MainWindow
    from hrap.gui.study import StudyRun
    from hrap.io.config import default_cfg, load_json, save_json

    first = window._shown
    first.path = str(tmp_path / "first.json")
    save_json(first.path, first.saved)
    window.name.setText("Edited first motor")
    window._add_motor(default_cfg(), title="Second motor")
    window.name.setText("Edited second motor")
    window.tabs.setCurrentIndex(2)
    StudyRun(default_cfg(), [("grain_L", [0.3])], ["Shifting OF"], name="Named run",
             file=window._runs_dir(first) / "sweep 1.json").save()
    assert window.close()  # no prompt: the session keeps the edits
    assert load_json(first.path)["mtr_nm"] != "Edited first motor"  # the file itself isn't touched

    again = MainWindow(QSettings(window._prefs.fileName(), QSettings.Format.IniFormat))
    try:
        motors = again._motors()
        assert [m.title for m in motors] == ["Edited first motor", "Edited second motor"]  # tabs show the Name
        assert [m.cfg["mtr_nm"] for m in motors] == ["Edited first motor", "Edited second motor"]
        assert all(again._edited(m) for m in motors)
        assert motors[0].path == first.path and motors[1].path == ""
        assert again.tabs.currentIndex() == 2
        assert [r.name for r in again.study_page.runs] == []  # the second motor's sweeps: each motor lists its own
        again.motor_tabs.setCurrentIndex(0)
        assert [r.name for r in again.study_page.runs] == ["Named run"]
    finally:
        again._session_timer.stop()
        again.deleteLater()


def test_editing_inputs_invalidates_completed_results(window):
    window._run()
    wait_for_run(window)
    assert window._output is not None
    assert window.sim.form.isEnabled()
    assert window._result_cfg["mtr_nm"] == window._settings.mtr_nm
    grain_L = window.sizing_page.grain_L.spin
    grain_L.setValue(grain_L.value() + 1)
    assert window._output is None
    assert window._result_cfg is None
    assert not window.sim.summary.toPlainText()


def test_missing_coolprop_reports_error_and_reenables_run(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    def unavailable(*args):
        raise ImportError("CoolProp missing")

    messages = []
    monkeypatch.setattr("hrap.advanced.fluid.coolprop_sat", unavailable)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: messages.append(args[-1]))
    window.sim.adv_on.setChecked(True)
    window.sim.ox_fluid.setCurrentText("NitrousOxide (CoolProp)")
    window._run()
    wait_for_run(window)
    assert window.sim.run_btn.isEnabled()
    assert window._output is None
    assert len(messages) == 1 and "CoolProp missing" in messages[0]


def test_display_units_update_results_without_changing_simulation(window, tmp_path):
    import numpy as np
    from PySide6.QtCore import QSettings
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
        panel = window.sim.plots
        plots = panel.plots
        assert {("Thrust", "force"), ("Pressure", "pressure"), ("Flow", "mass_flow")} <= set(plots)
        np.testing.assert_allclose(plots["Pressure", "pressure"].listDataItems()[0].yData, pressure * pressure_scale)
        np.testing.assert_allclose(plots["Flow", "mass_flow"].listDataItems()[0].yData,
                                   output.mdot_o * units.value(1, "mass_flow"))
        assert plots["Pressure", "pressure"].getAxis("left").labelUnits == units.pressure
        assert units.pressure in window.sim.summary.toPlainText()
        assert units.force in window.sim.summary.toPlainText()
        view = window._motor_view(1)
        assert view.overlay_tank[1] == units.text(pressure[1], "pressure", 3)
        assert units.unit("mass_flow") in dict(view.inj_card[1])["Oxidizer flow"]
        assert view.display_units.length == units.length
        panel.tabs.setCurrentWidget(panel.pages["Pressure"])
        QApplication.processEvents()
        panel.hover(float(output.t[1]))
        assert units.pressure in panel.pages["Pressure"].rows["P_tnk"].value.text()
        assert window._hover_index is not None
        assert window._prefs.value("displayUnits/pressure") == units.pressure
    panel.plots["Thrust", "force"].setXRange(.002, .008, padding=0)
    QApplication.processEvents()
    np.testing.assert_allclose(panel.plots["Pressure", "pressure"].vb.viewRange()[0], [.002, .008], atol=1e-6)
    window._set_display_units(DisplayUnits(pressure="bar"))
    QApplication.processEvents()
    assert panel.tabs.currentWidget() is panel.pages["Pressure"]
    np.testing.assert_allclose(panel.plots["Pressure", "pressure"].vb.viewRange()[0], [.002, .008], atol=1e-6)
    np.testing.assert_allclose(panel.plots["Flow", "ratio"].vb.viewRange()[0], [.002, .008], atol=1e-6)
    # A unit's plot shows while one of its lines is on.
    for key in ("P_tnk", "P_cmbr", "inj_dP"):
        panel.pages["Pressure"].rows[key].clicked.emit()
    assert not [q for tab, q in panel.plots if tab == "Pressure"]
    panel.pages["Pressure"].rows["T_tnk"].clicked.emit()
    assert [q for tab, q in panel.plots if tab == "Pressure"] == ["temperature"]


def test_small_metric_ruler_ticks_are_distinct():
    from hrap.gui.viz import _tick_label

    assert _tick_label(.05, .05) == "0.05"
    assert _tick_label(.10, .05) == "0.10"


def wait_for_study(window):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + 15
    while window.study_page.busy() and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(10)
    assert not window.study_page.busy()


def test_study_snapshot_units_history_and_open_case(window, tmp_path, monkeypatch):
    import json

    from PySide6.QtWidgets import QApplication, QFileDialog

    from hrap.engine.study import case_cfg
    from hrap.io.config import resolve
    from hrap.units import DisplayUnits

    page = window.study_page
    page.models.setCurrentIndex(page.models.findText("Both"))
    page.axes[0].set_axis("grain_L", [.25, .3])
    page.axes[1].set_axis("")
    page._run()
    assert not window.motor_tabs.isEnabled()
    wait_for_study(window)
    result = page.result
    assert len(result.cases) == 4
    assert not result.error
    assert page.table.rowCount() == 2 and page.table.columnCount() == 2
    page.table.item(0, 0).setSelected(True)
    page.table.item(1, 0).setSelected(True)
    assert len(page.plot.listDataItems()) == 2
    window._set_display_units(DisplayUnits(length="mm", pressure="bar"))
    assert "250 mm" in page.table.horizontalHeaderItem(0).text()
    assert len(page.plot.listDataItems()) == 2
    path = str(result.file)
    saved = json.loads(result.file.read_text())
    assert saved["motor"] == result.cfg
    page.table.clearSelection()
    page.table.item(0, 0).setSelected(True)
    selected = page._selected()[0]
    expected, _ = resolve(case_cfg(result.cfg, selected.values, selected.model))
    motors = window.motor_tabs.count()
    page._open()  # sized off the UI thread, as it may load an STL grain
    deadline = time.monotonic() + 15
    while window.motor_tabs.count() == motors and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    actual, _ = resolve(window._form_to_cfg())
    assert actual.grn_L == pytest.approx(expected.grn_L)
    assert actual.regression_model == expected.regression_model
    assert page.result is None  # the case opened as a new motor, which has no sweeps yet
    assert window._edited(window._shown)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (path, ""))
    page._load()
    assert page._base_override == result.cfg
    assert page.axes[0].read()[1] == pytest.approx([.25, .3])
    page.axes[0].unit.setCurrentText("mm")
    assert page.axes[0].read()[1] == pytest.approx([.25, .3])


def test_close_waits_for_study_and_stops_queue(window):
    page = window.study_page
    page.axes[0].set_axis("grain_L", [.2 + i * .01 for i in range(30)])
    page.axes[1].set_axis("")
    page._run()
    assert not window.close()
    wait_for_study(window)
    assert not window.isVisible()
    assert page.result.stopped
    assert len(page.result.cases) <= 8


def test_overlay_csv_loads_irl_thrust_and_tank_pressure(window):
    path = Path(__file__).resolve().parent / "fixtures" / "hps01_irl_sample.csv"
    assert window._reload_measured(str(path))
    assert window._measured is not None
    assert window._measured.F_thr is not None
    assert window._measured.P_tnk is not None
    assert "measured:F_thr" in window.sim.plots.pages["Thrust"].rows
    window._clear_measured()
    assert window._measured is None
