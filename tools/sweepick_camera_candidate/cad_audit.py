#!/usr/bin/env python3
"""Reproduce STEP audit and candidate meshes. Requires cadquery==2.8.0.
All STEP coordinates are imported in mm; exported STL vertices are metres.
The seating/yaw choices below are proposals, not assembly constraints.
"""
import hashlib
import json
from pathlib import Path
import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Trsf

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / 'assets/sweepick_camera_candidate'
BOTTOM_DATUM = np.array([46.81250222347442, 36.97500175621825, 0.])
TOP_DATUM = np.array([-50.4179925918579, -1.199890136719, 15.1870002746582])
SHOULDER_MM = 223.712455749511
# Proposal: bottom long 112mm direction => robot X. top CAD Y => robot Z.
R_BOTTOM = np.array([[0.,1,0],[-1,0,0],[0,0,1]])
R_TOP = np.array([[1.,0,0],[0,0,-1],[0,1,0]])


def rigid(shape, rotation, translation):
    t = gp_Trsf()
    t.SetValues(*np.column_stack([rotation, translation]).ravel().tolist())
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped,t,True).Shape())


def bounds(shape):
    b=shape.BoundingBox()
    return [[getattr(b,a+'min'),getattr(b,a+'max')] for a in 'xyz']


def main():
    (ASSET/'meshes').mkdir(parents=True,exist_ok=True)
    report={'status':'CANDIDATE_NOT_CALIBRATED','units':'mm in STEP, m in STL',
            'bottom_datum_cad_mm':BOTTOM_DATUM.tolist(),
            'top_datum_cad_mm':TOP_DATUM.tolist(),
            'R_mount_from_bottom_cad':R_BOTTOM.tolist(),
            'R_top_link_from_top_cad':R_TOP.tolist(),
            'top_link_xyz_in_mount_m':[0,0,SHOULDER_MM/1000], 'parts':{}}
    shapes=[]
    for name,R,datum in [('MOUNTBOTTM2',R_BOTTOM,BOTTOM_DATUM),('cam_mount_top2',R_TOP,TOP_DATUM)]:
        path=ASSET/'cad'/f'{name}.step'
        raw=path.read_bytes()
        assert b'SI_UNIT(.MILLI.,.METRE.)' in raw
        s=cq.importers.importStep(str(path)).val()
        assert s.isValid()
        faces=[]
        for i,f in enumerate(s.Faces()):
            d={'index':i,'type':f.geomType(),'area_mm2':f.Area(),
               'center_mm':list(f.Center().toTuple()),'bounds_mm':bounds(f)}
            if d['type']=='PLANE': d['normal']=list(f.normalAt().toTuple())
            if d['type']=='CYLINDER':
                cy=BRepAdaptor_Surface(f.wrapped).Cylinder()
                d.update(radius_mm=cy.Radius(),axis=list(cy.Axis().Direction().Coord()),
                         axis_point_mm=list(cy.Location().Coord()))
            faces.append(d)
        (ASSET/f'{name}_faces.json').write_text(json.dumps(faces,indent=2)+'\n')
        local=rigid(s,R,-R@datum)
        for i,solid in enumerate(local.Solids()):
            filename=f'{name}_{i}_m.stl'
            cq.exporters.export(solid.scale(.001),str(ASSET/'meshes'/filename),
                                tolerance=.00005,angularTolerance=.05)
        report['parts'][name]={'sha256':hashlib.sha256(raw).hexdigest(),
            'valid':s.isValid(),'solids':len(s.Solids()),'bounds_cad_mm':bounds(s),
            'volume_mm3':s.Volume(),'normalized_bounds_mm':bounds(local),
            'solid_bounds_cad_mm':[bounds(x) for x in s.Solids()]}
        shapes.append(local)
    # A single Boolean at proposed shoulder seating; broad-phase volume is not a fit proof.
    seated=shapes[1].translate((0,0,SHOULDER_MM))
    report['proposed_assembly']={'bounds_mm':bounds(cq.Compound.makeCompound([shapes[0],seated])),
        'interference_volume_mm3':shapes[0].intersect(seated).Volume(),
        'socket_depth_mm':7.9980010986327,'tongue_height_mm':7.850036621094,
        'warning':'No mates/tolerances in STEP. Zero interference alone does not prove seating or yaw.'}
    (ASSET/'cad_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['proposed_assembly'],indent=2))

if __name__=='__main__': main()
