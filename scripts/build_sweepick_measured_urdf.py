#!/usr/bin/env python3
"""Build a separate SAPIEN SO-101 runtime URDF from the active P-gripper URDF.
Only replace its base_so101_v2.stl mesh references, keeping joint frames.
Never overwrite the existing SAPIEN/CuRobo source URDFs.
"""
import argparse
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "assets/embodiments/so101-pgripper"
SOURCE = ARM / "so101_pgripper.urdf"
TARGET = ARM / "so101_pgripper_sweepick_measured.urdf"
MESH = ARM / "assets/Base_SO101_sweepick_measured_mm.stl"


def dimensions_mm(data):
    if len(data) < 134:
        raise ValueError("STL data too short")
    n, = struct.unpack_from("<I", data, 80)
    if n == 0 or len(data) != 84 + n * 50:
        raise ValueError("Expected binary STL")
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for t in range(n):
        for vertex in range(3):
            xyz = struct.unpack_from("<fff", data, 84 + 50*t + 12 + 12*vertex)
            for a in range(3):
                lo[a] = min(lo[a], xyz[a])
                hi[a] = max(hi[a], xyz[a])
    dims = [hi[i]-lo[i] for i in range(3)]
    if not all(10 < d < 250 for d in dims):
        raise ValueError(f"STL dimensions expected in millimetres, got {dims}")
    return dims


def create(base_stl):
    base = Path(base_stl).expanduser().resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    original = ET.parse(SOURCE).getroot()
    count = 0
    for mesh in original.findall(".//mesh"):
        if mesh.get("filename") == "assets/base_so101_v2.stl":
            if mesh.get("scale") not in (None, "1 1 1", "1.0 1.0 1.0"):
                raise ValueError("Unexpected original mesh scale")
            mesh.set("filename", "assets/Base_SO101_sweepick_measured_mm.stl")
            mesh.set("scale", "0.001 0.001 0.001")
            count += 1
    if count != 2:
        raise ValueError(f"Expected 2 visual/collision base mesh references, got {count}")
    for mesh in original.findall(".//mesh"):
        if mesh.get("filename") == "assets/Base_SO101_sweepick_measured_mm.stl":
            continue
        if not (ARM / mesh.get("filename")).is_file():
            raise FileNotFoundError(f"Missing original mesh {mesh.get('filename')}")
    raw_stl = base.read_bytes()
    dims = dimensions_mm(raw_stl)
    ET.indent(original)
    raw_xml = ET.tostring(original, encoding="utf-8", xml_declaration=True)
    for path, data in ((MESH, raw_stl), (TARGET, raw_xml)):
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite an existing generated file: {path}")
    MESH.write_bytes(raw_stl)
    TARGET.write_bytes(raw_xml)
    print("Generated measured-only SAPIEN URDF:", TARGET)
    print("Modified base STL dimensions (mm):", dims)
    print("Original SAPIEN and CuRobo URDFs were not modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-stl", required=True, type=Path)
    create(parser.parse_args().base_stl)
