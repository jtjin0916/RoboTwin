#!/usr/bin/env python3
"""Package generated measured SWEEPICK SO-101 URDF and all referenced meshes.

Runs on the teammate's SOURCE machine, where the generated unit STL/URDF
exist locally (they are intentionally not committed to RoboTwin Git).
The ZIP contains repository-relative paths for portable extraction.
"""
import argparse
import hashlib
import io
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ARM = Path("assets/embodiments/so101-pgripper")
MEASURED = Path("assets/embodiments/so101-pgripper-sweepick-measured")
GENERATED = ARM / "so101_pgripper_sweepick_unit.urdf"
UNIT_STL = ARM / "assets/Base_SO101_sweepick_unit_m.stl"
OTHER_URDFS = (
    ARM / "so101_pgripper.urdf",
    ARM / "so101_pgripper_calib.urdf",
)
CONFIG_FILES = (
    MEASURED / "config.yml",
    MEASURED / "curobo.yml",
)
SOURCE_FILES = (
    Path("scripts/build_sweepick_measured_unit_urdf.py"),
    Path("sweepick_table_frame.py"),
    Path("docs/dapier/SWEEPICK_TABLETOP_WORLD_Y_SMOKE.md"),
)


def _mesh_files(root, urdf_rel):
    urdf_file = root / urdf_rel
    try:
        tree = ET.parse(urdf_file)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF XML: {urdf_file}") from exc
    found = set()
    for mesh in tree.findall(".//mesh"):
        name = mesh.attrib.get("filename")
        if not name:
            raise ValueError(f"Mesh missing filename in {urdf_rel}")
        if "://" in name or name.startswith("/") or "\\" in name:
            raise ValueError(f"Nonportable URDF mesh reference: {name}")
        rel = Path(name)
        if ".." in rel.parts or rel.is_absolute():
            raise ValueError(f"Mesh path traversal in {urdf_rel}: {name}")
        found.add(ARM / rel)
    return found


def _readme():
    return """# SWEEPICK measured SO-101 handoff

Contents:
- assets/embodiments/so101-pgripper/so101_pgripper_sweepick_unit.urdf
- assets/embodiments/so101-pgripper/assets/Base_SO101_sweepick_unit_m.stl
- every mesh referenced by the generated SAPIEN URDF, source and CuRobo URDF
- assets/embodiments/so101-pgripper-sweepick-measured/config.yml
- assets/embodiments/so101-pgripper-sweepick-measured/curobo.yml
- source URDF and measured unit-STL conversion script
- SHA256SUMS.txt to verify transfer integrity

This is the CURRENT SIMULATION ARM model, not a single merged
TurtleBot3 + two SO-101 + OS30A URDF.
The main URDF is used for SAPIEN; measured config.yml locates both
arms and the static head camera. CuRobo's curobo.yml still refers to
the original so101_pgripper_calib.urdf, not the new measured-base URDF.
TurtleBot3 URDF and its meshes are external and NOT in this ZIP.
The head-camera extrinsic is PROVISIONAL until remeasured.
The TurtleBot chassis and profile obstacle shapes are NOT fully
modeled/validated in CuRobo. Never assume physical collision safety.

Extraction: unzip SWEEPICK_SO101_measured_handoff.zip -d /path/to/RoboTwin
Keep the directory structure! Open the generated URDF only after the
referenced STL mesh files have been extracted to their original paths.
The rest of RoboTwin code and the external TurtleBot3 assets must be
taken from the matching project revision.

See MANIFEST.txt for the exact included filenames and SHA256SUMS.txt
for file checksums (sha256sum -c SHA256SUMS.txt from extraction root).
"""


def package(root, output):
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    required = set((GENERATED, UNIT_STL, *OTHER_URDFS, *CONFIG_FILES, *SOURCE_FILES))
    if not (root / GENERATED).is_file() or not (root / UNIT_STL).is_file():
        raise FileNotFoundError(
            "Missing generated Sweepick unit-scale URDF/STL. On the machine "
            "where simulation worked, run:\n"
            "python scripts/build_sweepick_measured_unit_urdf.py "
            "--base-stl ~/sweepick_ros2_ws/src/sweepick_description/resources/Base_SO101__.stl"
        )
    for urdf in (GENERATED, *OTHER_URDFS):
        required.update(_mesh_files(root, urdf))
    for name in sorted(required, key=str):
        f = root / name
        if not f.is_file() or f.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty required handoff asset: {f}")

    # Do not modify any URDF, STL or project configuration.
    # Validate all inputs before creating the archive.
    if output.is_relative_to(root) and output in [root / p for p in required]:
        raise ValueError("Output path would overwrite a model input!")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Archive exists; refusing to overwrite: {output}")

    lines = []
    for name in sorted(required, key=str):
        data = (root / name).read_bytes()
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name.as_posix()}")
    manifest = "\n".join(sorted(str(x) for x in required)) + "\n"
    checksums = "\n".join(lines) + "\n"

    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for rel in sorted(required, key=str):
            zf.write(root / rel, arcname=rel.as_posix())
        zf.writestr("HANDOFF_README.md", _readme())
        zf.writestr("MANIFEST.txt", manifest)
        zf.writestr("SHA256SUMS.txt", checksums)

    with zipfile.ZipFile(output) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
    print(f"[SWEEPICK HANDOFF] {output}")
    print(f"[SWEEPICK HANDOFF] Model/config/source files: {len(required)}")
    print(f"[SWEEPICK HANDOFF] Bytes: {output.stat().st_size}")
    print("[SWEEPICK HANDOFF] Zip CRC check passed; SAPIEN loading is not tested.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT, help="RoboTwin worktree")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "SWEEPICK_SO101_measured_handoff.zip")
    ns = ap.parse_args()
    package(ns.root, ns.output)


if __name__ == "__main__":
    main()
