"""Hand-worked companion-channel windows and end-to-end indexed fatigue checks."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from anyfatigue import MultidirectionalSettings, TensorCompleteness, get_curve
from anyfatigue.cli import main
from anyfatigue.cycle_association import ASSOCIATION_POLICY, associate_shear_cycles, _RangeTree
from anyfatigue.damage import SECONDS_PER_YEAR
from anyfatigue.errors import InputError
from anyfatigue.extract import ElementLocation
from anyfatigue.project import AnalysisDefinition
from anyfatigue.rainflow import count_indexed_rainflow_cycles
from anyfatigue.report import result_to_dict, report_markdown
from anyfatigue.stress_io import source_from_history, write_stress_npz
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings, assess_time_series
from test_parent_plate import junction, location

CURVE = get_curve('D', 'free_corrosion')
LOC = ElementLocation((1,), surfaces=('top',), weld_direction=(0,1,0))


def md(curve=CURVE, chunk=None):
    return MultidirectionalSettings.uniform(curve, gamma=1.0, gamma_basis='Analytical test only',
        curve_basis='Uniform classification only for this test', step_deg=45,
        shear_association='cycle_associated', cycle_chunk_size=chunk)


def history(normal, shear=None, times=None):
    n=np.asarray(normal,dtype=float)
    x=np.zeros((len(n),6));x[:,0]=n
    if shear is not None: x[:,3]=shear
    return source_from_history(np.arange(len(n)) if times is None else times,x)


def settings(source, multi=None):
    return TimeSeriesSettings(CURVE,25,2/SECONDS_PER_YEAR,1,
        (LoadBlock('record', source.case_labels),), multidirectional=multi or md())


def assess(source, multi=None, keep=False):
    return assess_time_series(source,[LOC],settings(source,multi),keep_series=keep)


def support(normal,shear,**kw):
    cycles=count_indexed_rainflow_cycles(normal,chunk_size=kw.pop('chunk_size',None))
    return cycles,associate_shear_cycles(normal,shear,cycles,**kw)


def test_return_branch_extremum_and_fractional_endpoint_are_both_included():
    # Inner loop 2 -> 3 -> 2 closes halfway from sample 3 (normal=3)
    # to sample 4 (normal=1). Its interpolated shear is 20, not 0 or 40.
    normal=[0,4,2,3,1,5,0];shear=[0,0,0,0,40,0,200]
    indexed,a=support(normal,shear,times=[0,1,2,3,7,8,10])
    assert a.shear_ranges[:2] == pytest.approx([20,40])
    assert a.closure_left_indices[:2].tolist() == [3,4]
    assert a.closure_fractions[:2] == pytest.approx([.5,.75])
    assert a.closure_times[:2] == pytest.approx([5,7.75])
    assert indexed.cycles.ranges[:2] == pytest.approx([1,3])
    assert a.support_kinds[:2] == ('full_return','full_return')


def test_large_pulse_after_closure_does_not_contaminate_earlier_inner_cycles():
    n=[0,4,2,3,1,5,0]
    _,a=support(n,[0,0,0,0,40,0,0]);_,b=support(n,[0,0,0,0,40,0,1e9])
    np.testing.assert_array_equal(a.shear_ranges[:2],b.shear_ranges[:2])
    assert b.shear_ranges[2] > a.shear_ranges[2]  # genuinely inside the outer window


def test_exact_return_excludes_later_plateau_and_shear_peak():
    n=[0,4,2,3,2,2,1,5,0]
    _,a=support(n,[0,0,0,1,2,300,400,0,0])
    assert a.closure_left_indices[0] == 4 and a.closure_fractions[0] == 0
    assert a.shear_ranges[0] == 2


def test_adjacent_equal_half_cycles_see_full_out_of_phase_shear():
    n=[30,0,-30,0,30];s=[0,50,0,-50,0]
    indexed,a=support(n,s)
    assert indexed.cycles.counts.tolist() == [.5,.5]
    assert a.shear_ranges.tolist() == [100,100]
    assert a.support_start_indices.tolist() == [0,0]
    assert a.closure_left_indices.tolist() == [4,4]
    assert a.paired_cycle_indices.tolist() == [1,0]
    assert a.support_kinds == ('paired_halves','paired_halves')
    result=assess(history(n,s)).critical
    zero=next(d for d in result.directions if d.angle_deg == 0)
    expected=.5/CURVE.cycles_to_failure(math.hypot(60,90))  # duration 4, exposure 2 s
    assert zero.damage == pytest.approx(expected,rel=1e-12)


def test_unmatched_boundary_half_is_explicit_and_not_given_a_fictitious_return():
    _,a=support([30,0,-30],[0,50,0])
    assert a.support_kinds == ('half_interval',)
    assert a.paired_cycle_indices.tolist() == [-1]
    assert a.shear_ranges.tolist() == [50]
    assert a.closure_left_indices.tolist() == [2]


def test_pairing_does_not_use_a_hidden_stress_tolerance():
    indexed,a=support([1,-1,1.0000000001],[0,1,0])
    assert len(indexed.cycles)==2
    assert a.paired_cycle_indices.tolist()==[-1,-1]


def test_nested_windows_use_the_original_chronology_including_excursions():
    n=[0,4,2,3,1,5,0]
    _,a=support(n,[0,0,120,0,0,0,0])
    assert a.shear_ranges[0]==120 and a.shear_ranges[1]==120
    assert a.support_start_indices[:2].tolist()==[2,1]


@pytest.mark.parametrize('chunk',[1,2,3,4,7,50])
def test_computational_chunks_preserve_damage_and_full_audit(chunk):
    source=history([0,40,20,30,10,50,0],[0,4,-8,10,40,0,1])
    a=assess(source);b=assess(source,md(chunk=chunk))
    assert b.critical.damage==a.critical.damage
    assert b.critical.parts[0].cycle_association.to_dict()==a.critical.parts[0].cycle_association.to_dict()
    assert b.critical.parts[0].cycle_damage.tolist()==a.critical.parts[0].cycle_damage.tolist()


def test_different_load_blocks_are_not_joined_even_with_shared_values():
    source=history([30,-30,30,-30],[0,50,0,-50])
    st=settings(source)
    st=replace(st,blocks=(LoadBlock('first',source.case_labels[:2],.5),LoadBlock('second',source.case_labels[2:],.5)))
    p=assess_time_series(source,[LOC],st).critical
    assert sum(part.damage for part in p.parts)==pytest.approx(p.damage)
    for part in p.parts:
        assert part.cycle_association.support_kinds==('half_interval',)
        assert part.cycle_association.paired_cycle_indices.tolist()==[-1]
        assert part.cycle_association.start_indices.tolist()==[0]  # block-local indices


def test_uniaxial_limit_and_corrections_are_applied_once():
    source=history([30,-30,30]);st=settings(source)
    st=replace(st,scf=1.3,thickness_mm=50,dff=2,multidirectional=replace(st.multidirectional,gamma=1.2))
    p=assess_time_series(source,[LOC],st).critical
    expected=1/CURVE.cycles_to_failure(60*1.3*1.2*2**CURVE.k)
    assert p.damage==pytest.approx(expected,rel=1e-12)
    assert p.usage==pytest.approx(2*expected)
    assert p.critical_angle_deg==0
    assert p.parts[0].cycle_damage.sum()==p.damage


@pytest.mark.parametrize('method,curve_name',[('hotspot_a','D'),('hotspot_b','E')])
def test_parent_patch_extraction_both_faces_and_saved_cli_audit(tmp_path,method,curve_name):
    source,parent=junction();curve=get_curve(curve_name,'free_corrosion')
    loc=location(parent,method=method)
    st=TimeSeriesSettings(curve,20,2/SECONDS_PER_YEAR,1,(LoadBlock('record',source.case_labels),),multidirectional=md(curve))
    definition=AnalysisDefinition('time_series',st,(loc,))
    path=definition.save(tmp_path/'indexed.json')
    assert AnalysisDefinition.load(path)==definition
    result=definition.run(source,keep_series=False)
    assert len(result.points)==4
    for p in result.points:
        assert p.parts[0].series is None
        assert p.parts[0].cycle_association is not None
        assert p.parts[0].cycle_damage.sum()==p.damage
    data=result_to_dict(result,include_cycles=True)
    json.dumps(data,allow_nan=False)
    cycles=data['points'][0]['parts'][0]['cycles']
    assert cycles['association']['policy']==ASSOCIATION_POLICY
    assert sum(cycles['design_life_damage'])==data['points'][0]['damage']
    assert 'cycle_associated' in report_markdown(result)
    assert result.provenance['indexed_counter_contract']==1
    spath=write_stress_npz(source,tmp_path/'source.npz')
    assert main(['run',str(path),'--stress',str(spath),'--out',str(tmp_path),'--include-cycles'])==0
    cli=json.loads((tmp_path/'indexed-result.json').read_text())
    assert cli['points'][0]['parts'][0]['cycles']['association']==cycles['association']


def test_missing_tensor_component_remains_an_error():
    values=np.zeros((3,6));values[:,0]=[30,-30,30]
    source=source_from_history([0,1,2],values,tensor_completeness=TensorCompleteness(('xx',)))
    with pytest.raises(InputError,match='missing tensor components'): assess(source)


def test_constant_tensor_produces_empty_but_valid_audit():
    p=assess(history([30,30,30],[10,10,10])).critical
    assert p.damage==0
    assert p.parts[0].cycle_association.shear_ranges.size==0
    json.dumps(p.parts[0].cycle_association.to_dict(),allow_nan=False)


def test_pure_shear_is_still_detected_on_other_planes():
    p=assess(history([0,0,0],[40,-40,40])).critical
    assert p.normal_to_weld_damage==0
    assert p.damage==pytest.approx(1/CURVE.cycles_to_failure(80),rel=1e-12)
    assert 'no resolved normal-master cycle' in p.notes


def test_associated_ranges_do_not_exceed_the_same_records_global_envelope():
    rng=np.random.default_rng(2871)
    for _ in range(12):
        source=history(rng.integers(-30,31,80),rng.integers(-50,51,80))
        a=assess(source);g=assess(source,replace(md(),shear_association='global_range'))
        assert a.critical.damage <= g.critical.damage*(1+1e-12)


def test_tree_queries_match_direct_original_sample_operations():
    rng=np.random.default_rng(14519);x=rng.normal(size=123);tree=_RangeTree(x)
    for _ in range(500):
        a=int(rng.integers(0,len(x)));b=int(rng.integers(a,len(x)))
        assert tree.extrema(a,b)==(min(x[a:b+1]),max(x[a:b+1]))
        target=float(rng.normal())
        for above in (False,True):
            indices=np.flatnonzero(x[a:b+1]>=target if above else x[a:b+1]<=target)
            expected=a+int(indices[0]) if len(indices) else None
            assert tree.first_crossing(a,b,target,above)==expected


def test_random_full_closures_agree_with_independent_raw_linear_scan():
    rng=np.random.default_rng(9231)
    for _ in range(60):
        n=rng.normal(size=100);s=rng.normal(size=100)
        indexed,a=support(n,s)
        for i,count in enumerate(indexed.cycles.counts):
            if count!=1: continue
            first,turn,last=map(int,(indexed.start_indices[i],indexed.end_indices[i],indexed.closing_indices[i]))
            target=n[first]
            for j in range(turn+1,last+1):
                if (n[j]>=target) if target>n[turn] else (n[j]<=target): break
            f=(target-n[j-1])/(n[j]-n[j-1])
            endpoint=(1-f)*s[j-1]+f*s[j]
            samples=list(s[first:j])+[endpoint]
            assert a.shear_ranges[i]==pytest.approx(max(samples)-min(samples),rel=1e-12)
            assert a.closure_left_indices[i]+a.closure_fractions[i]==pytest.approx(j-1+f)


@pytest.mark.parametrize('chunk',[0,-1,True,2.5,'4'])
def test_bad_chunk_settings_fail_closed(chunk):
    with pytest.raises(InputError): md(chunk=chunk)


def test_global_mode_cannot_silently_ignore_computational_chunk_setting():
    with pytest.raises(InputError): replace(md(chunk=1),shear_association='global_range')


@pytest.mark.parametrize('normal,shear,times',[
    ([0,1],[0],None),([0,np.nan],[0,1],None),([0,1],[1j,0],None),
    ([0,1],[0,1],[0]),([0,1],[0,1],[1,0]),
])
def test_invalid_history_is_rejected(normal,shear,times):
    indexed=count_indexed_rainflow_cycles([0,1])
    with pytest.raises(InputError): associate_shear_cycles(normal,shear,indexed,times=times)


def test_indexed_results_cannot_be_applied_to_a_different_normal_history():
    indexed=count_indexed_rainflow_cycles([0,2,0])
    with pytest.raises(InputError,match='do not belong'):
        associate_shear_cycles([0,3,0],[0,1,0],indexed)


def test_full_cycle_sign_reversal_preserves_windows_and_ranges():
    n=np.array([0,4,2,3,1,5,0.]);s=np.array([0,2,-3,1,4,0,2.])
    _,a=support(n,s);_,b=support(-n,-s)
    np.testing.assert_array_equal(a.closure_left_indices,b.closure_left_indices)
    np.testing.assert_allclose(a.closure_fractions,b.closure_fractions,rtol=1e-14)
    np.testing.assert_allclose(a.shear_ranges,b.shear_ranges,rtol=1e-14)


def test_rigid_frame_rotation_preserves_end_to_end_parent_assessment():
    angle=math.radians(33); q=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
    source,p=junction(); rotated,rp=junction(rotation=q)
    loc=location(p);rloc=location(rp)
    rloc=replace(rloc,weld_direction=tuple(q@np.array([0,1,0])),
                 readout=replace(rloc.readout,direction=tuple(q@np.array([1,0,0]))))
    st=settings(source)
    a=assess_time_series(source,[loc],st);b=assess_time_series(rotated,[rloc],st)
    assert b.critical.damage==pytest.approx(a.critical.damage,rel=1e-12)
    assert b.critical.critical_angle_deg==a.critical.critical_angle_deg


def test_nonintegral_external_index_table_fails_with_typed_error():
    indexed=count_indexed_rainflow_cycles([0,2,0])
    indexed=replace(indexed,start_indices=indexed.start_indices.astype(float))
    with pytest.raises(InputError,match='integer arrays'):
        associate_shear_cycles([0,2,0],[0,1,0],indexed)


def test_supplied_global_mode_serialization_is_unchanged():
    config=replace(md(),shear_association='global_range')
    body=config.to_dict()
    assert 'cycle_chunk_size' not in body
    assert MultidirectionalSettings.from_dict(body)==config
