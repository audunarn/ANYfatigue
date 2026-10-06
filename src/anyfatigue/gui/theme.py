"""Light workbench styling, matching ANYfem's Qt theme (``anyfem.ui.qt.theme``).

Only presentation is shared with ANYfem, never model or engineering content;
keep this in step with ANYfem's theme (ANYsection keeps a copy the same way).
"""

from PySide6.QtGui import QFont

STYLESHEET = """
    QMainWindow, QDialog, QDockWidget, QWidget#TaskWorkspace { background: #f3f5f8; }
    QWidget { color: #253247; }
    QMenuBar, QToolBar { background: #ffffff; border: none; spacing: 5px; }
    QMenuBar::item { padding: 7px 12px; }
    QMenuBar::item:selected, QMenu::item:selected { background: #e5efff; color: #164b96; }
    QMenu { background: white; border: 1px solid #ccd5e0; padding: 5px; }
    QMenu::item { padding: 7px 24px; }
    QToolBar { padding: 5px; border-bottom: 1px solid #dce2eb; }
    QToolBar::separator { background: #dce2eb; width: 1px; margin: 5px 8px; }
    QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 6px; }
    QToolButton:hover { background: #edf3fd; border-color: #d1dff4; }
    QToolButton:pressed, QToolButton:checked { background: #dce9fb; }
    QDockWidget::title { background: #edf1f6; padding: 9px 12px; font-weight: bold; }
    QLabel#WorkspaceHeading { color: #68768a; font-size: 11px; font-weight: bold; }
    QLabel#Hint { color: #68768a; }
    QLabel#Notice { color: #7a4b00; background: #fff4dc; border: 1px solid #f0d9a8;
        border-radius: 4px; padding: 6px 8px; }
    QLabel#Verdict { font-size: 15px; font-weight: bold; padding: 8px 10px; border-radius: 4px; }
    QLabel#Verdict[state="pass"] { color: #14532d; background: #dcfce7; border: 1px solid #86efac; }
    QLabel#Verdict[state="fail"] { color: #7f1d1d; background: #fee2e2; border: 1px solid #fca5a5; }
    QLabel#Verdict[state="idle"] { color: #475569; background: #eef1f5; border: 1px solid #dce2eb; }
    QScrollArea { border: none; background: #f8fafc; }
    QScrollArea > QWidget > QWidget { background: #f8fafc; }
    QTabWidget::pane { border: none; }
    QTabBar::tab { background: #e9edf3; padding: 8px 14px; border: none; }
    QTabBar::tab:selected { background: white; color: #1659ad; }
    QSplitter::handle { background: #dce2eb; }
    QSplitter::handle:horizontal { width: 1px; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
        background: white; border: 1px solid #cbd5e1; border-radius: 4px; padding: 5px 7px;
        selection-background-color: #245fab; selection-color: white;
    }
    QComboBox { min-height: 20px; padding-right: 22px; }
    QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border-color: #2563b4; }
    QLineEdit[invalid="true"] { border-color: #dc2626; background: #fef2f2; }
    QAbstractItemView { background: white; alternate-background-color: #f5f8fc;
        border: 1px solid #dce2eb; selection-background-color: #dce9fb; selection-color: #163e73; }
    QAbstractItemView::item { padding: 4px; }
    QHeaderView::section { background: #edf1f6; color: #54647a; padding: 7px;
        border: none; border-bottom: 1px solid #dce2eb; }
    QPushButton { background: white; border: 1px solid #cbd5e1; border-radius: 4px;
        padding: 7px 12px; min-height: 18px; }
    QPushButton:hover { background: #edf3fd; border-color: #9bb9df; }
    QPushButton:pressed { background: #dce9fb; }
    QPushButton[primary="true"] { background: #245fab; border-color: #245fab; color: white; font-weight: bold; }
    QPushButton[primary="true"]:hover { background: #194f96; }
    QPushButton:focus, QToolButton:focus { border: 2px solid #2563b4; }
    QWidget:disabled { color: #8592a4; }
    QPushButton:disabled, QLineEdit:disabled, QComboBox:disabled { background: #eef1f5; border-color: #dce2eb; }
    QPushButton[primary="true"]:disabled { background: #eef1f5; color: #8592a4; border-color: #dce2eb; }
    QCheckBox { spacing: 7px; padding: 3px; }
    QGroupBox { border: 1px solid #dce2eb; border-radius: 5px; margin-top: 12px; padding-top: 10px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; font-weight: bold; }
    QStatusBar { background: white; border-top: 1px solid #dce2eb; padding: 3px; }
    QToolTip { background: #253247; color: white; border: none; padding: 6px; }
"""


def apply_theme(widget) -> None:
    font = QFont(widget.font())
    font.setPointSizeF(10)
    widget.setFont(font)
    widget.setStyleSheet(STYLESHEET)
