"""The existing GUI must not silently discard a parent-region selection."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import replace
import pytest
try:
    from PySide6.QtWidgets import QApplication
except ImportError as error:
    pytest.skip(f'Qt unavailable (ANYfatigue optional-GUI gate): {error}',allow_module_level=True)
from anyfatigue import ParentPlate, examples
from anyfatigue.errors import InputError
from anyfatigue.gui.window import MainWindow
pytestmark=pytest.mark.qt


def test_parent_definition_rejected_before_mutating_forms():
    app=QApplication.instance() or QApplication([])
    window=MainWindow();window._quiet=True
    try:
        source,original=examples.synthetic_plate(steps=60)
        window.load_source(source,original)
        parent=ParentPlate(tuple(int(i) for i in source.mesh.element_ids),'element_average')
        locations=(replace(original.locations[0],parent_plate=parent),)+original.locations[1:]
        changed=replace(original,locations=locations,name='must-not-replace-original')
        with pytest.raises(InputError,match='Parent-plate definitions'):
            window.set_definition(changed)
        assert window.collect_definition()==original
    finally:
        window.wait();window.close();app.processEvents()
