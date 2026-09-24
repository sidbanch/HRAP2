"""Dark / light palettes for the HRAP desktop UI."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory


DARK = {
    "window": "#16181d",
    "base": "#1e222a",
    "alt": "#252a33",
    "text": "#e8eaed",
    "disabled": "#7d8490",
    "accent": "#5b8def",
    "accent2": "#3d6fd4",
    "border": "#3a404c",
    "plot_bg": "#1a1d23",
    "plot_fg": "#e6e8ee",
}

LIGHT = {
    "window": "#f4f5f7",
    "base": "#ffffff",
    "alt": "#ebecef",
    "text": "#1b1d21",
    "disabled": "#8a9099",
    "accent": "#2f5fbf",
    "accent2": "#244a96",
    "border": "#c5c9d1",
    "plot_bg": "#ffffff",
    "plot_fg": "#1b1d21",
}


def _palette(colors: dict) -> QPalette:
    p = QPalette()
    w = QColor(colors["window"])
    b = QColor(colors["base"])
    a = QColor(colors["alt"])
    t = QColor(colors["text"])
    d = QColor(colors["disabled"])
    acc = QColor(colors["accent"])
    p.setColor(QPalette.ColorRole.Window, w)
    p.setColor(QPalette.ColorRole.WindowText, t)
    p.setColor(QPalette.ColorRole.Base, b)
    p.setColor(QPalette.ColorRole.AlternateBase, a)
    p.setColor(QPalette.ColorRole.Text, t)
    p.setColor(QPalette.ColorRole.Button, a)
    p.setColor(QPalette.ColorRole.ButtonText, t)
    p.setColor(QPalette.ColorRole.Highlight, acc)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.ToolTipBase, b)
    p.setColor(QPalette.ColorRole.ToolTipText, t)
    p.setColor(QPalette.ColorRole.PlaceholderText, d)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, d)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, d)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, d)
    return p


def apply_theme(app: QApplication, name: str = "dark") -> dict:
    colors = DARK if name != "light" else LIGHT
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setPalette(_palette(colors))
    accent = colors["accent"]
    app.setStyleSheet(
        f"""
        QMainWindow, QDialog {{ background: {colors['window']}; }}
        QGroupBox {{
            font-weight: 600;
            border: 1px solid {colors['border']};
            border-radius: 0px;
            margin-top: 12px;
            padding: 10px 8px 8px 8px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
            background: {colors['base']};
            border: 1px solid {colors['border']};
            border-radius: 2px;
            padding: 2px 4px;
            min-height: 22px;
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 0px;
            height: 0px;
            border: none;
        }}
        QPushButton {{
            background: {colors['alt']};
            border: 1px solid {colors['border']};
            border-radius: 2px;
            padding: 6px 14px;
        }}
        QPushButton#runButton {{
            background: {accent};
            color: white;
            font-weight: 700;
            border: none;
            padding: 8px 22px;
        }}
        QPushButton#runButton:hover {{ background: {colors['accent2']}; }}
        QPushButton#runButton:disabled {{ background: {colors['disabled']}; color: {colors['base']}; }}
        QWidget#configHeader {{
            background: {colors['base']};
            border: 1px solid {colors['border']};
            border-radius: 0px;
        }}
        QWidget#collapsibleBox {{
            background: {colors['base']};
            border: 1px solid {colors['border']};
        }}
        QPushButton#collapseHeader {{
            color: {colors['text']};
            font-weight: 600;
            padding: 8px 10px;
            background: {colors['alt']};
            border: none;
            text-align: left;
            border-radius: 0px;
        }}
        QPushButton#collapseHeader:hover {{ background: {colors['base']}; }}
        QPushButton#collapseHeader:checked {{ background: {colors['alt']}; }}
        QWidget#collapseBody {{ background: {colors['base']}; }}
        QFrame#motorPanel {{
            background: {colors['plot_bg']};
            border: 1px solid {colors['border']};
        }}
        QLabel#motorStatsStrip {{
            color: {colors['text']};
            padding: 8px 12px;
            background: {colors['base']};
            border-top: 1px solid {colors['border']};
        }}
        QSplitter::handle {{ background: {colors['border']}; }}
        QFrame#sizingCard {{
            background: {colors['base']};
            border: 1px solid {colors['border']};
            border-radius: 6px;
        }}
        QFrame#applyBar {{
            background: {colors['base']};
            border-top: 1px solid {colors['border']};
        }}
        QFrame#cardDivider {{ color: {colors['border']}; }}
        QLabel#cardTitle {{ font-weight: 700; font-size: 14px; }}
        QLabel#cardLabel {{ color: {colors['disabled']}; }}
        QLabel#cardValue {{ font-weight: 600; }}
        QLabel#sizingError {{
            color: #e06c75;
            padding: 8px 12px;
            border: 1px solid #e06c75;
            border-radius: 6px;
        }}
        QTabWidget#pageTabs::pane {{ border: none; }}
        QTabWidget#pageTabs > QTabBar::tab {{
            padding: 8px 22px;
            font-weight: 600;
            color: {colors['disabled']};
            background: transparent;
            border: none;
            border-top: 2px solid transparent;
        }}
        QTabWidget#pageTabs > QTabBar::tab:selected {{ color: {colors['text']}; border-top: 2px solid {accent}; }}
        QTabWidget#pageTabs > QTabBar::tab:hover {{ color: {colors['text']}; }}
        QStatusBar {{ background: {colors['alt']}; }}
        QProgressBar {{
            background: {colors['base']};
            border: 1px solid {colors['border']};
            border-radius: 0px;
            text-align: center;
            color: {colors['text']};
            max-height: 18px;
        }}
        QProgressBar::chunk {{ background: {accent}; }}
        QMenuBar {{ background: {colors['alt']}; }}
        QScrollArea {{ border: none; }}
        QCheckBox {{ spacing: 8px; }}
        """
    )
    return colors
