"""Analytical connected-plate regressions for explicit recovery support."""
from dataclasses import replace
import json

import numpy as np
import pytest

from anyfatigue import ParentPlate, TensorCompleteness, get_curve
from anyfatigue.cli import main
from anyfatigue.damage import SECONDS_PER_YEAR
from anyfatigue.errors import ExtractionError, InputError
from anyfatigue.extract import LineLocation, PointLocation, ReadOut, extract, location_from_dict
from anyfatigue.measures import in_plane_components
from anyfatigue.model import ArrayStressSource, StressSource, SurfaceMesh
from anyfatigue.project import AnalysisDefinition
from anyfatigue.report import report_markdown, result_to_dict
from anyfatigue.stress_io import read_stress_npz, write_stress_npz
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings, assess_time_series


def junction(*, bracket=10000., left=800., renumber=False, rotation=None):
    # A plate extends both sides of x=0; the bracket occupies x=0, z>=0.
    ids = np.arange(1, 11)
    xyz = np.array([[0,0,0],[.04,0,0],[.04,.1,0],[0,.1,0],[-.04,0,0],
                    [-.04,.1,0],[0,0,.04],[0,.1,.04],[.08,0,0],[.08,.1,0.]])
    eids = np.array([20,30,10,1])
    corners = np.array([[1,2,3,4],[2,9,10,3],[5,1,4,6],[1,4,8,7]])
    values = np.zeros((3,4,6))
    values[:,:,0] = np.outer([1,-1,1], [100,200,left,bracket])
    values[:,:,1] = .2*values[:,:,0]
    values[:,:,3] = .03*values[:,:,0]
    parent_ids = (20,30)
    if renumber:
        eids = np.array([1,900,30,200])
        parent_ids = (1,900)
        order = np.array([3,1,2,0])
        eids, corners, values = eids[order], corners[order], values[:,order]
    if rotation is not None:
        xyz = xyz @ rotation.T + np.array([1.,-2.,3.])
        matrix = np.zeros(values.shape[:2] + (3,3))
        matrix[...,0,0], matrix[...,1,1] = values[...,0], values[...,1]
        matrix[...,0,1] = matrix[...,1,0] = values[...,3]
        matrix = rotation @ matrix @ rotation.T
        values = matrix[..., [0,1,2,0,1,0], [0,1,2,1,2,2]]
    mesh = SurfaceMesh(ids,xyz,eids,corners)
    # Poison the global supplied nodal field: a parent patch must not use it.
    nodal = np.full((3,mesh.n_nodes,6), 99999.)
    source = ArrayStressSource(mesh, ('a','b','c'), values, -.5*values,
                               node_top=nodal, node_bottom=-nodal, times=(0,1,2),
                               provenance={'mesh_hash':'junction-v1'})
    return source, ParentPlate(parent_ids, 'element_average', name='support +x', mesh_id='junction-v1')


def location(parent, *, kind='line', method='hotspot_a', thickness=20):
    options = dict(parent_plate=parent, readout=ReadOut(method, thickness_mm=thickness,
                                                       direction=(1,0,0)),
                   weld_direction=(0,1,0), name='toe')
    return (LineLocation((1,4), **options) if kind == 'line' else
            PointLocation(((0,.025,0),(0,.075,0)), **options))


@pytest.mark.parametrize('kind', ['line','point'])
@pytest.mark.parametrize('method,expected', [('direct',100.),('hotspot_a',100.),('hotspot_b',112.5)])
def test_selected_parent_recovery_has_analytical_values_on_both_faces(kind, method, expected):
    source, parent = junction()
    points = extract(source, location(parent, kind=kind, method=method))
    assert len(points) == 4
    for p in points:
        factor = 1 if p.surface == 'top' else -.5
        want = factor*np.outer([1,-1,1], [expected,.2*expected,0,.03*expected,0,0])
        np.testing.assert_allclose(p.tensors, want, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p.normal,[0,0,1],atol=1e-14)
        assert 'parent-only element average' in p.notes
        assert 'unscoped nodal field not used' in p.notes


@pytest.mark.parametrize('kind',['line','point'])
def test_attached_plate_and_opposite_side_cannot_contaminate_the_parent(kind):
    a, parent = junction(bracket=1., left=2.)
    b, _ = junction(bracket=1e12, left=-1e13)
    loc = location(parent,kind=kind)
    for x,y in zip(extract(a,loc),extract(b,loc)):
        assert np.array_equal(x.tensors,y.tensors)
        assert np.array_equal(x.normal,y.normal)


@pytest.mark.parametrize('kind',['line','point'])
def test_unselected_intersection_is_rejected_instead_of_using_element_number(kind):
    source, parent = junction()
    with pytest.raises(ExtractionError,match='parent_plate'):
        extract(source,replace(location(parent,kind=kind),parent_plate=None))


def test_element_renumbering_and_reordering_do_not_select_a_different_plate():
    a,pa=junction(); b,pb=junction(renumber=True)
    for x,y in zip(extract(a,location(pa)),extract(b,location(pb))):
        np.testing.assert_allclose(x.tensors,y.tensors,rtol=1e-12)


def test_rigid_frame_rotation_and_translation_preserve_weld_relative_stress():
    a=np.deg2rad(37);b=np.deg2rad(23)
    ry=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    rz=np.array([[np.cos(b),-np.sin(b),0],[np.sin(b),np.cos(b),0],[0,0,1]])
    q=rz@ry
    original,p=junction(); rotated,rp=junction(rotation=q)
    loc=location(rp)
    loc=replace(loc,weld_direction=tuple(q@np.array([0,1,0])),
                readout=replace(loc.readout,direction=tuple(q@np.array([1,0,0]))))
    for x,y in zip(extract(original,location(p)),extract(rotated,loc)):
        sx=in_plane_components(x.tensors,x.normal,x.weld_direction)
        sy=in_plane_components(y.tensors,y.normal,y.weld_direction)
        np.testing.assert_allclose(sx,sy,rtol=1e-11,atol=1e-11)


@pytest.mark.parametrize('selected,match', [((1,20),'planar'),((404,),'not in the mesh'),
                                         ((20,30,10),'read-out side')])
def test_bad_parent_selection_fails_closed(selected,match):
    source,parent=junction()
    with pytest.raises(ExtractionError,match=match):
        extract(source,location(replace(parent,element_ids=selected)))


def test_opposite_readout_side_is_not_recovered_using_other_side_elements():
    source,parent=junction();loc=location(parent)
    with pytest.raises(ExtractionError,match='read-out side'):
        extract(source,replace(loc,readout=replace(loc.readout,direction=(-1,0,0))))


def test_side_flag_without_explicit_readout_direction_is_respected():
    source,parent=junction();loc=location(parent)
    # n x weld = -x, so side=-1 reads the positive-x parent patch.
    okay=replace(loc,readout=ReadOut('hotspot_a',thickness_mm=20,side=-1))
    assert extract(source,okay)[0].tensors[0,0] == pytest.approx(100.)
    with pytest.raises(ExtractionError,match='read-out side'):
        extract(source,replace(okay,readout=replace(okay.readout,side=1)))


def test_off_parent_probe_never_falls_back_to_another_element():
    source,parent=junction()
    with pytest.raises(ExtractionError):
        extract(source,location(parent,thickness=200))


def test_node_outside_parent_patch_is_rejected():
    source,parent=junction()
    with pytest.raises(ExtractionError,match='node 5'):
        extract(source,replace(location(parent),node_ids=(5,6)))


def test_upstream_mesh_identity_is_checked_when_bound():
    source,parent=junction()
    with pytest.raises(ExtractionError,match='mesh_id'):
        extract(source,location(replace(parent,mesh_id='different-mesh')))


def test_mixed_directors_do_not_silently_swap_top_and_bottom():
    source,parent=junction()
    normals=source.mesh.normals().copy();normals[1]*=-1
    source.mesh=replace(source.mesh,element_normals=normals)
    with pytest.raises(ExtractionError,match='top/bottom'):
        extract(source,location(parent))


def test_geometric_winding_is_not_used_to_override_consistent_physical_directors():
    source,parent=junction()
    normals=source.mesh.normals().copy()
    corners=source.mesh.element_corner_ids.copy();corners[1]=corners[1,::-1]
    source.mesh=replace(source.mesh,element_corner_ids=corners,element_normals=normals)
    points=extract(source,location(parent))
    assert points[0].tensors[0,0] == pytest.approx(100)
    assert points[2].tensors[0,0] == pytest.approx(-50)


def test_disconnected_planar_regions_are_not_one_parent_patch():
    from helpers import plate_mesh, source_from_fields, tensor_field
    source=source_from_fields(plate_mesh(4,1),[tensor_field(np.zeros((3,6)))])
    with pytest.raises(ExtractionError,match='connected'):
        ParentPlate((1,4),'element_average').bind(source)


def test_duplicate_shell_elements_are_not_a_parent_patch():
    source,parent=junction()
    corners=source.mesh.element_corner_ids.copy();corners[1]=corners[0]
    source.mesh=replace(source.mesh,element_corner_ids=corners)
    with pytest.raises(ExtractionError,match='duplicate'):
        parent.bind(source)


def test_lazy_producer_is_read_once_per_face_and_global_nodal_field_is_never_requested():
    source,parent=junction()
    class Lazy(StressSource):
        def __init__(self):
            self.mesh=source.mesh;self.case_labels=source.case_labels
            self.case_times=source.case_times;self.provenance=source.provenance
            self.tensor_completeness=source.tensor_completeness;self.calls=[]
        def element_stress(self,rows,surface):
            self.calls.append((tuple(rows),surface));return source.element_stress(rows,surface)
        def node_stress(self,rows,surface):
            raise AssertionError('unscoped nodal recovery must not be read')
    lazy=Lazy();extract(lazy,location(parent,kind='point'))
    assert lazy.calls == [((0,1),'top'),((0,1),'bottom')]


@pytest.mark.parametrize('bad', [(),(1,1),(True,),('1',),(1.5,),None])
def test_invalid_element_ids(bad):
    with pytest.raises(InputError): ParentPlate(bad,'element_average')


@pytest.mark.parametrize('kwargs',[{'nodal_recovery':'source_nodal'},{'name':''},{'mesh_id':''}])
def test_invalid_parent_configuration(kwargs):
    with pytest.raises(InputError): replace(ParentPlate((1,),'element_average'),**kwargs)


def test_parent_selection_roundtrip_and_legacy_definition_shape():
    _source,parent=junction()
    for kind in ('line','point'):
        loc=location(parent,kind=kind)
        assert location_from_dict(json.loads(json.dumps(loc.to_dict()))) == loc
        legacy=replace(loc,parent_plate=None)
        assert 'parent_plate' not in legacy.to_dict()
        assert location_from_dict(legacy.to_dict()) == legacy


@pytest.mark.parametrize('method,curve_name,normal_range',[('hotspot_a','D',200.),('hotspot_b','E',225.)])
def test_saved_definition_cli_npz_and_damage_preserve_extraction_identity(tmp_path,method,curve_name,normal_range):
    source,parent=junction();loc=location(parent,method=method)
    curve=get_curve(curve_name,'free_corrosion')
    st=TimeSeriesSettings(curve,20,2/SECONDS_PER_YEAR,1,(LoadBlock('history',source.case_labels),))
    definition=AnalysisDefinition('time_series',st,(loc,))
    definition_path=definition.save(tmp_path/'parent.json')
    stress_path=write_stress_npz(source,tmp_path/'source.npz')
    back=AnalysisDefinition.load(definition_path)
    result=back.run(read_stress_npz(stress_path))
    assert result.critical.damage == pytest.approx(1/curve.cycles_to_failure(normal_range),rel=1e-12)
    data=result_to_dict(result)
    assert data['provenance']['locations'][0]['parent_plate']==parent.to_dict()
    assert 'parent-only element average' in data['points'][0]['notes']
    assert main(['run',str(definition_path),'--stress',str(stress_path),'--out',str(tmp_path/'reports')]) == 0


def test_multidirectional_parent_plate_uses_same_selected_surface_tensor():
    from anyfatigue import MultidirectionalSettings
    source,parent=junction();loc=location(parent)
    curve=get_curve('D','free_corrosion')
    md=MultidirectionalSettings.uniform(curve,gamma=1,gamma_basis='test',curve_basis='test',step_deg=45)
    st=TimeSeriesSettings(curve,20,2/SECONDS_PER_YEAR,1,(LoadBlock('history',source.case_labels),),multidirectional=md)
    result=assess_time_series(source,[loc],st)
    assert result.critical.normal_to_weld_damage == pytest.approx(1/curve.cycles_to_failure(200),rel=1e-12)
    assert result.critical.damage >= result.critical.normal_to_weld_damage


def test_report_describes_actual_parent_recovery_not_poisoned_source_nodal_field():
    source,parent=junction();loc=location(parent)
    curve=get_curve('D','free_corrosion')
    st=TimeSeriesSettings(curve,20,2/SECONDS_PER_YEAR,1,(LoadBlock('history',source.case_labels),))
    result=assess_time_series(source,[loc],st)
    text=report_markdown(result)
    assert 'parent-only element average on explicitly selected patches' in text
    assert 'element IDs [20, 30]' in text
    assert 'junction-v1' in text
