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
    # What things are, the same everywhere: oxidizer blue, fuel orange, red only for limits and the case wall.
    "oxidizer": "#5b8def",
    "fuel": "#f08a3c",
    "limit": "#e06c75",
    "warn": "#e5c07b",
    "green": "#98c379",
    "purple": "#c678dd",
    "teal": "#56b6c2",
    "grey": "#abb2bf",
    "glass": "#d1dbed",  # the unburned fuel, see-through, in the 3D grain
    # Sweep result tiles, under white text.
    "tile_ok": "#2f7d4f",
    "tile_warn": "#a87a22",
    "tile_bad": "#a8413b",
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
    "oxidizer": "#2f5fbf",
    "fuel": "#e07020",
    "limit": "#c8414c",
    "warn": "#b07d12",
    "green": "#3f8a2e",
    "purple": "#9a3fb5",
    "teal": "#1f8a99",
    "grey": "#5c6370",
    "glass": "#404d61",
    "tile_ok": "#2f7d4f",
    "tile_warn": "#a87a22",
    "tile_bad": "#a8413b",
}


def colors_for(name: str = "dark") -> dict:
    return LIGHT if name == "light" else DARK


def mix(a: str, b: str, share: float) -> QColor:
    """Colour a, with `share` of colour b mixed in."""
    ca, cb = QColor(a), QColor(b)
    return QColor.fromRgbF(*((1 - share) * x + share * y for x, y in zip(ca.getRgbF()[:3], cb.getRgbF()[:3])))


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
    colors = colors_for(name)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setPalette(_palette(colors))
    accent = colors["accent"]
    app.setStyleSheet(
        f"""
        QMainWindow, QDialog {{ background: {colors['window']}; }}
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
        QPushButton#switch {{ padding: 4px 12px; }}
        QPushButton#switch:checked {{ background: {accent}; border-color: {accent}; color: white; }}
        QPushButton#runButton {{
            background: {accent};
            color: white;
            font-weight: 700;
            border: none;
            padding: 8px 22px;
        }}
        QPushButton#runButton:hover {{ background: {colors['accent2']}; }}
        QPushButton#runButton:disabled {{ background: {colors['disabled']}; color: {colors['base']}; }}
        QPushButton#runButton[upToDate="true"] {{ background: {colors['alt']}; color: {colors['disabled']}; border: 1px solid {colors['border']}; }}
        QPushButton#runButton[upToDate="true"]:hover {{ color: {colors['text']}; }}
        QWidget#configHeader {{
            background: {colors['base']};
            border: 1px solid {colors['border']};
            border-radius: 0px;
        }}
        QFrame#motorPanel {{
            background: {colors['plot_bg']};
            border: 1px solid {colors['border']};
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
        QLabel#notApplied {{ color: {colors['warn']}; }}
        QToolButton#builtBadge {{ background: transparent; border: none; padding: 0; color: {colors['disabled']}; }}
        QToolButton#builtBadge:hover {{ color: {colors['text']}; text-decoration: underline; }}
        QAbstractSpinBox[changed="true"] {{ border-left: 3px solid {colors['warn']}; }}
        QLabel#sizingError {{
            color: {colors['limit']};
            padding: 8px 12px;
            border: 1px solid {colors['limit']};
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
