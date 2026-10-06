"""Optional PySide6 front end; importing ANYfatigue never loads Qt."""

from __future__ import annotations


def main(argv=None):
    from .window import main as launch

    return launch(argv)


def __getattr__(name):
    if name == "MainWindow":
        from .window import MainWindow

        return MainWindow
    raise AttributeError(name)
