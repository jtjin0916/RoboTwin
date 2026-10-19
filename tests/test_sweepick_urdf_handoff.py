"""Standalone CPU tests of measured-URDF portable handoff packer."""
import importlib.util
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/package_sweepick_measured_urdf.py"
spec = importlib.util.spec_from_file_location("handoff_packer", SCRIPT)
packer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packer)


def _make_mock_project(root):
    refs = (
        (packer.GENERATED, "assets/Base_SO101_sweepick_unit_m.stl"),
        (packer.OTHER_URDFS[0], "assets/base_so101_v2.stl"),
        (packer.OTHER_URDFS[1], "assets_meter/gripper.stl"),
    )
    for rel, mesh in refs:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        node = ET.Element("robot", {"name": "mock"})
        link = ET.SubElement(node, "link", {"name": "base"})
        visual = ET.SubElement(link, "visual")
        geo = ET.SubElement(visual, "geometry")
        ET.SubElement(geo, "mesh", {"filename": mesh})
        path.write_bytes(ET.tostring(node))
        mesh_path = root / packer.ARM / mesh
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.write_bytes(b"test-mesh")
    for rel in (*packer.CONFIG_FILES, *packer.SOURCE_FILES):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mock\n")


def test_complete_archive_contains_generated_and_all_referenced_meshes(tmp_path):
    root = tmp_path / "repo"
    _make_mock_project(root)
    zip_path = tmp_path / "handoff.zip"
    packer.package(root, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.testzip() is None
        files = set(zf.namelist())
        assert packer.GENERATED.as_posix() in files
        assert packer.UNIT_STL.as_posix() in files
        assert "assets/embodiments/so101-pgripper/assets_meter/gripper.stl" in files
        assert "HANDOFF_README.md" in files
        assert "MANIFEST.txt" in files
        assert "SHA256SUMS.txt" in files
    with pytest.raises(FileExistsError):
        packer.package(root, zip_path)


def test_missing_mesh_fails_without_creating_archive(tmp_path):
    root = tmp_path / "repo"
    _make_mock_project(root)
    (root / packer.UNIT_STL).unlink()
    zip_path = tmp_path / "bad.zip"
    with pytest.raises(FileNotFoundError):
        packer.package(root, zip_path)
    assert not zip_path.exists()


def test_traversal_mesh_rejected(tmp_path):
    root = tmp_path / "repo"
    _make_mock_project(root)
    path = root / packer.GENERATED
    path.write_text(
        "<robot><link name='x'><visual><geometry>"
        "<mesh filename='../escape.stl'/>"
        "</geometry></visual></link></robot>"
    )
    with pytest.raises(ValueError, match="traversal"):
        packer.package(root, tmp_path / "bad.zip")
