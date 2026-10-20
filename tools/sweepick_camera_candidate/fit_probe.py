#!/usr/bin/env python3
"""Slow exact BRep candidate seating probes. Prints JSON; no files overwritten."""
import json
import cadquery as cq
import numpy as np
from cad_audit import ASSET,BOTTOM_DATUM,TOP_DATUM,R_BOTTOM,R_TOP,SHOULDER_MM,rigid

def main():
    bottom=cq.importers.importStep(str(ASSET/'cad/MOUNTBOTTM2.step')).val()
    top=cq.importers.importStep(str(ASSET/'cad/cam_mount_top2.step')).val()
    b=rigid(bottom,R_BOTTOM,-R_BOTTOM@BOTTOM_DATUM)
    t=rigid(top,R_TOP,-R_TOP@TOP_DATUM)
    results=[]
    for yaw in (0,180):
        s=t.rotate((0,0,0),(0,0,1),yaw).translate((0,0,SHOULDER_MM))
        for dz in (0,-.15,.15):
            p=s.translate((0,0,dz))
            results.append(dict(top_yaw_deg=yaw,seating_delta_mm=dz,
                                interference_mm3=b.intersect(p).Volume(),minimum_distance_mm=b.distance(p)))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
