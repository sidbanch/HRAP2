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
def window(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from hrap.gui.main import MainWindow

    # Closing asks about unsaved motors; don't let the question block the tests.
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Discard)
    win = MainWindow(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
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


@pytest.mark.parametrize("outcome", ["save", "cancel", "discard", "cancel_path", "write_error"])
def test_quit_with_unsaved_motors(window, tmp_path, monkeypatch, outcome):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from hrap.io.config import default_cfg, load_json, save_json

    first = window._shown
    first.path = str(tmp_path / "first.json")
    save_json(first.path, first.saved)
    original = load_json(first.path)
    window.mfg.setText("Edited first motor")
    window._add_motor(default_cfg(), title="Second motor")
    second = window._shown
    window.mfg.setText("Edited second motor")
    second_path = tmp_path / "second.json"
    errors = []

    def answer(parent, title, text, buttons, default):
        assert buttons & QMessageBox.StandardButton.SaveAll
        assert default == QMessageBox.StandardButton.SaveAll
        return {
            "cancel": QMessageBox.StandardButton.Cancel,
            "discard": QMessageBox.StandardButton.Discard,
        }.get(outcome, QMessageBox.StandardButton.SaveAll)

    with monkeypatch.context() as patch:
        patch.setattr(QMessageBox, "question", answer)
        patch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[-1]))
        patch.setattr(QFileDialog, "getSaveFileName", lambda *args: (
            "" if outcome == "cancel_path" else str(second_path), "",
        ))

        def write(path, cfg):
            if outcome == "write_error" and path == str(second_path):
                raise PermissionError("Permission denied")
            save_json(path, cfg)

        patch.setattr("hrap.gui.main.save_json", write)
        assert window.close() == (outcome in ("save", "discard"))

    assert load_json(first.path)["mfg"] == (
        original["mfg"] if outcome in ("cancel", "discard") else "Edited first motor"
    )
    if outcome == "save":
        assert load_json(second_path)["mfg"] == "Edited second motor"
        assert not any(window._edited(m) for m in window._motors())
        assert window._prefs.value("openMotors") == [first.path, str(second_path)]
    else:
        assert not second_path.exists()
        assert second.path == ""
        assert window._edited(second)
    assert bool(errors) == (outcome == "write_error")


def test_editing_inputs_invalidates_completed_results(window):
    window._run()
    wait_for_run(window)
    assert window._output is not None
    assert window._form.isEnabled()
    assert window._result_cfg["mtr_nm"] == window._settings.mtr_nm
    grain_L = window.sizing_page.grain_L.spin
    grain_L.setValue(grain_L.value() + 1)
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

    from PySide6.QtWidgets import QFileDialog

    from hrap.engine.study import case_cfg
    from hrap.io.config import resolve
    from hrap.units import DisplayUnits

    page = window.study_page
    page.axes[0].set_axis("grain_L", [.25, .3])
    page.axes[1].set_axis("")
    page.models.setCurrentIndex(1)
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
    path = str(tmp_path / "study.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (path, ""))
    page._save()
    saved = json.loads((tmp_path / "study.json").read_text())
    assert saved["motor"] == result.cfg
    page.table.clearSelection()
    page.table.item(0, 0).setSelected(True)
    selected = page._selected()[0]
    expected, _ = resolve(case_cfg(result.cfg, selected.values, selected.model))
    page._open()
    actual, _ = resolve(window._form_to_cfg())
    assert actual.grn_L == pytest.approx(expected.grn_L)
    assert actual.regression_model == expected.regression_model
    assert page.result is result
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
