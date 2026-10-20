"""These tests run with either the published or the indexed owner package."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from anyfatigue import MultidirectionalSettings, get_curve
from anyfatigue.errors import InputError, MissingDependencyError
from anyfatigue.extract import ElementLocation
from anyfatigue.rainflow import count_indexed_rainflow_cycles, count_rainflow_cycles
import anyfatigue.rainflow as adapter
from anyfatigue.stress_io import source_from_history
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings, assess_time_series


def test_old_backend_keeps_scalar_and_global_mode_but_rejects_associated(monkeypatch):
    actual=adapter._backend()
    monkeypatch.setattr(adapter,'_backend',lambda: SimpleNamespace(count_cycles=actual.count_cycles))
    assert count_rainflow_cycles([1,-1,1]).total_cycles==1
    curve=get_curve('D','air')
    source=source_from_history([0,1,2],np.array([[1,0,0,0,0,0],[-1,0,0,0,0,0],[1,0,0,0,0,0]]))
    md=MultidirectionalSettings.uniform(curve,gamma=1,gamma_basis='test',curve_basis='test',step_deg=45)
    settings=TimeSeriesSettings(curve,20,1,1,(LoadBlock('r',source.case_labels),),multidirectional=md)
    loc=ElementLocation((1,),surfaces=('top',),weld_direction=(0,1,0))
    assert assess_time_series(source,[loc],settings).critical.damage>0
    with pytest.raises(MissingDependencyError,match='No fallback'):
        assess_time_series(source,[loc],replace(settings,multidirectional=replace(md,shear_association='cycle_associated')))


@pytest.mark.parametrize('version',[None,0,2,'1'])
def test_unknown_contract_is_not_accepted(monkeypatch,version):
    fake=SimpleNamespace(INDEXED_CYCLES_VERSION=version,count_cycles_indexed=lambda *a,**k: ())
    monkeypatch.setattr(adapter,'_backend',lambda:fake)
    with pytest.raises(MissingDependencyError,match='contract v1'): count_indexed_rainflow_cycles([0,1,0])


@pytest.mark.parametrize('change',[
    {'start_index':True},{'start_index':-1},{'end_index':30},{'end_index':0},
    {'count':1.0},{'kind':'unknown'},{'range':100.0},{'mean':np.inf},
    {'closing_index':None},{'closing_index':True},
])
def test_owner_contract_violations_fail_closed(monkeypatch,change):
    values=dict(range=2.,mean=0.,count=.5,start_index=0,end_index=1,closing_index=2,kind='initial_half')
    values.update(change)
    fake=SimpleNamespace(INDEXED_CYCLES_VERSION=1,count_cycles_indexed=lambda *a,**k: [SimpleNamespace(**values)])
    monkeypatch.setattr(adapter,'_backend',lambda:fake)
    with pytest.raises(InputError,match='contract violation'): count_indexed_rainflow_cycles([1,-1,1])
