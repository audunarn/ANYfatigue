"""Missing stresses and known zeros must remain distinguishable at boundaries."""
from dataclasses import replace
import json

import numpy as np
import pytest

from anyfatigue import MultidirectionalSettings, TensorCompleteness, get_curve
from anyfatigue.errors import InputError
from anyfatigue.extract import ElementLocation
from anyfatigue.model import ArrayStressSource, StressSource
from anyfatigue.stress_io import (materialise, read_history_csv, read_stress_npz,
                                 source_from_history, write_stress_npz)
from anyfatigue.tensor import TENSOR_ORDER, completeness_of, require_in_plane_tensor
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings, assess_time_series


def assess(source,*,multi=True):
    curve=get_curve('D','air')
    md=MultidirectionalSettings.uniform(curve,gamma=1,gamma_basis='test',curve_basis='test',step_deg=45)
    settings=TimeSeriesSettings(curve,20,1,1,(LoadBlock('history',source.case_labels),),
                                multidirectional=md if multi else None)
    return assess_time_series(source,[ElementLocation((1,),surfaces=('top',),weld_direction=(0,1,0))],settings)


def csv_source(tmp_path,header='time,stress',values='0,10\n1,-10\n2,10\n',**kwargs):
    file=tmp_path/'history.csv';file.write_text(header+'\n'+values)
    return read_history_csv(file,**kwargs)


def test_legacy_scalar_csv_remains_usable_but_not_as_an_implicit_multiaxial_tensor(tmp_path):
    source=csv_source(tmp_path)
    assert source.tensor_completeness.provided == ('xx',)
    assert source.tensor_completeness.assumed_zero == ()
    assert assess(source,multi=False).critical.damage > 0
    with pytest.raises(InputError,match='yy, xy'):
        assess(source)


def test_complete_in_plane_tensor_does_not_need_unknown_transverse_components_in_xy_plane(tmp_path):
    source=csv_source(tmp_path,'time,xx,yy,xy','0,10,5,3\n1,-10,-5,-3\n2,10,5,3\n')
    assert not source.tensor_completeness.complete
    assert source.tensor_completeness.missing == ('zz','yz','xz')
    assert assess(source).critical.damage > 0


def test_full_global_tensor_and_zero_assumptions_are_different_but_equally_complete(tmp_path):
    full=csv_source(tmp_path,'time,xx,yy,zz,xy,yz,xz','0,10,0,0,0,0,0\n1,-10,0,0,0,0,0\n2,10,0,0,0,0,0\n')
    completed=csv_source(tmp_path,assumed_zero_components=('yy','zz','xy','yz','xz'),
                         assumption_basis='Prescribed uniaxial test in global x')
    assert full.tensor_completeness.provided == TENSOR_ORDER
    assert completed.tensor_completeness.assumed_zero == ('yy','zz','xy','yz','xz')
    assert full.tensor_completeness.complete and completed.tensor_completeness.complete
    assert assess(full).critical.damage == assess(completed).critical.damage


def test_a_numerically_zero_supplied_column_is_still_provided_not_assumed(tmp_path):
    source=csv_source(tmp_path,'time,xx,yy,xy','0,10,0,0\n1,-10,0,0\n2,10,0,0\n')
    assert source.tensor_completeness.provided == ('xx','yy','xy')
    assert source.tensor_completeness.assumed_zero == ()


def test_inclined_surface_requires_the_additional_global_tensor_components(tmp_path):
    source=csv_source(tmp_path,'time,xx,yy,xy','0,10,5,3\n1,-10,-5,-3\n2,10,5,3\n')
    with pytest.raises(InputError,match='missing tensor components'):
        require_in_plane_tensor(source,(1,1,1))
    with pytest.raises(InputError):
        require_in_plane_tensor(source,(1e-20,0,1))  # no angle threshold hides missing components


@pytest.mark.parametrize('assumed,basis',[(('yy',),''),(('xx',),'test'),(('foo',),'test'),(None,'test'),(('yy','yy'),'test')])
def test_invalid_zero_assumptions_fail_closed(tmp_path,assumed,basis):
    with pytest.raises(InputError):
        csv_source(tmp_path,assumed_zero_components=assumed,assumption_basis=basis)


def test_nonzero_data_cannot_claim_assumed_zero():
    data=np.zeros((3,6));data[:,0]=[10,-10,10];data[0,1]=1
    declaration=TensorCompleteness(('xx',),('yy',),'prescribed zero')
    with pytest.raises(InputError,match='declared zero'):
        source_from_history([0,1,2],data,tensor_completeness=declaration)


def test_complete_array_contract_remains_available_for_programmatic_sources():
    source=source_from_history([0,1,2],np.zeros((3,6)))
    assert source.tensor_completeness.complete
    assert source.tensor_completeness.provided == TENSOR_ORDER


@pytest.mark.parametrize('complete',[False,True])
def test_npz_and_materialisation_do_not_launder_missing_components(tmp_path,complete):
    options=({'assumed_zero_components':('yy','zz','xy','yz','xz'),'assumption_basis':'test'} if complete else {})
    original=csv_source(tmp_path,**options)
    copied=materialise(original)
    restored=read_stress_npz(write_stress_npz(copied,tmp_path/'stress.npz'))
    assert restored.tensor_completeness == original.tensor_completeness
    assert restored.provenance['tensor_completeness'] == original.tensor_completeness.to_dict()
    if complete:
        assert assess(restored).critical.damage > 0
    else:
        with pytest.raises(InputError): assess(restored)


def rewrite_metadata(file,change):
    with np.load(file,allow_pickle=False) as data:
        fields={k:data[k] for k in data.files}
    meta=json.loads(str(fields['meta']));change(meta)
    fields['meta']=np.array(json.dumps(meta));np.savez(file,**fields)


def test_legacy_npz_remains_scalar_readable_but_completeness_is_unknown(tmp_path):
    source=csv_source(tmp_path)
    file=write_stress_npz(source,tmp_path/'legacy.npz')
    def old(meta):
        meta['version']=1;meta.pop('tensor_completeness')
    rewrite_metadata(file,old)
    restored=read_stress_npz(file)
    assert restored.tensor_completeness == TensorCompleteness()
    assert assess(restored,multi=False).critical.damage > 0
    with pytest.raises(InputError): assess(restored)
    # Re-saving must retain unknown status, rather than trust padded arrays/provenance.
    assert read_stress_npz(write_stress_npz(restored,tmp_path/'resaved.npz')).tensor_completeness == TensorCompleteness()


@pytest.mark.parametrize('change',[
    lambda m:m.pop('tensor_completeness'),lambda m:m.update(unit='Pa'),
    lambda m:m.update(tensor_order=['xx','yy','zz','xy','xz','yz']),
    lambda m:m['tensor_completeness'].update(frame='local'),
    lambda m:m['tensor_completeness'].update(provided=['xx','xx']),
])
def test_new_file_contract_rejects_missing_or_conflicting_metadata(tmp_path,change):
    file=write_stress_npz(csv_source(tmp_path),tmp_path/'bad.npz')
    rewrite_metadata(file,change)
    with pytest.raises(InputError): read_stress_npz(file)


def test_unknown_third_party_contract_is_not_inferred_from_six_array_slots():
    assert completeness_of(StressSource()) == TensorCompleteness()
    with pytest.raises(InputError): require_in_plane_tensor(StressSource(),(0,0,1))


def test_duplicated_csv_columns_are_not_silently_selected(tmp_path):
    with pytest.raises(InputError,match='not repeat'):
        csv_source(tmp_path,'time,xx,xx','0,1,2\n1,2,3\n')


def test_report_provenance_retains_engineering_basis(tmp_path):
    source=csv_source(tmp_path,assumed_zero_components=('yy','zz','xy','yz','xz'),assumption_basis='Prescribed global-x test')
    result=assess(source)
    assert result.provenance['tensor_completeness']['basis'] == 'Prescribed global-x test'
    assert result.provenance['tensor_completeness']['provided'] == ['xx']


@pytest.mark.parametrize('which',['top','bottom'])
def test_complex_frequency_amplitudes_cannot_be_imported_as_real_history(which):
    tensor=np.zeros((3,6));complex_tensor=tensor.astype(complex)+1j
    with pytest.raises(InputError,match='real time-domain'):
        source_from_history([0,1,2],complex_tensor if which=='top' else tensor,
                            bottom=complex_tensor if which=='bottom' else tensor)
