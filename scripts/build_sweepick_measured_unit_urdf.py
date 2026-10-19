#!/usr/bin/env python3
"""Build measured SO-101 SAPIEN URDF without non-unit collision mesh scaling.

Input Base_SO101__.stl vertices are in mm; bake the mm->m scale directly into
a new binary STL and keep the generated URDF mesh scales at 1 (or omitted).
This leaves source SO101 URDF, its kinematic joint origins, and all preexisting
generated measured_mm artifacts untouched.
"""
import argparse
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "assets/embodiments/so101-pgripper"
SOURCE = ARM / "so101_pgripper.urdf"
TARGET = ARM / "so101_pgripper_sweepick_unit.urdf"
MESH = ARM / "assets/Base_SO101_sweepick_unit_m.stl"
SOURCE_MESH = "assets/base_so101_v2.stl"
OUTPUT_MESH = "assets/Base_SO101_sweepick_unit_m.stl"


def convert_stl_mm_to_m(raw):
    """Return binary STL with triangle vertices multiplied by .001."""
    if len(raw) < 134:
        raise ValueError("STL too short")
    count, = struct.unpack_from("<I", raw, 80)
    if count <= 0 or len(raw) != 84 + 50 * count:
        raise ValueError("Expected nonempty binary STL (84 + 50*N bytes)")
    out = bytearray(raw)
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    for tri in range(count):
        for k in range(3):
            offset = 84 + 50 * tri + 12 + 12 * k
            xyz = struct.unpack_from("<fff", raw, offset)
            for j in range(3):
                low[j] = min(low[j], xyz[j])
                high[j] = max(high[j], xyz[j])
            struct.pack_into("<fff", out, offset, *[v * 0.001 for v in xyz])
    dims_mm = [high[i] - low[i] for i in range(3)]
    if not all(10 < d < 250 for d in dims_mm):
        raise ValueError(f"Base STL dimensions must be in mm, got {dims_mm}")
    return bytes(out), dims_mm


def create(base_stl):
    stl_path = Path(base_stl).expanduser().resolve()
    if not stl_path.is_file():
        raise FileNotFoundError(stl_path)
    converted, dims_mm = convert_stl_mm_to_m(stl_path.read_bytes())
    root = ET.parse(SOURCE).getroot()
    count = 0
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename == SOURCE_MESH:
            if mesh.get("scale") not in (None, "1 1 1", "1.0 1.0 1.0"):
                raise ValueError("Unexpected original base mesh scale")
            mesh.set("filename", OUTPUT_MESH)
            mesh.attrib.pop("scale", None)
            count += 1
        elif filename and not (ARM / filename).is_file():
            raise FileNotFoundError(f"Missing original link mesh {filename}")
    if count != 2:
        raise ValueError(f"Expected 2 visual/collision base mesh references, got {count}")
    ET.indent(root)
    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    # Idempotent; never silently overwrite changed generated assets.
    for path, data in ((MESH, converted), (TARGET, xml)):
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(
                f"Different generated asset exists: {path}. "
                "Inspect that generated file before explicitly removing it."
            )
    MESH.parent.mkdir(parents=True, exist_ok=True)
    MESH.write_bytes(converted)
    TARGET.write_bytes(xml)
    print(f"[SWEEPICK] Original STL mm extents: {dims_mm}")
    print(f"[SWEEPICK] Converted STL vertices to metres: {MESH}")
    print(f"[SWEEPICK] Unit-scale SAPIEN URDF: {TARGET}")
    print("[SWEEPICK] Existing joint frames, original URDF, CuRobo source unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-stl", type=Path, required=True)
    create(parser.parse_args().base_stl)
