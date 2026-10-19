"""Unit-scale STL and isolated randomization config tests (CPU only)."""
import importlib.util
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    path = ROOT / "scripts/build_sweepick_measured_unit_urdf.py"
    spec = importlib.util.spec_from_file_location("sweepick_unit_stl_builder", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stl_vertices_baked_from_mm_into_m():
    raw = bytearray(84 + 50)
    struct.pack_into("<I", raw, 80, 1)
    for k, vertex in enumerate(((10., 20., 30.), (50., 20., 30.), (10., 70., 90.))):
        struct.pack_into("<fff", raw, 84 + 12 + 12*k, *vertex)
    out, dims = _builder().convert_stl_mm_to_m(bytes(raw))
    assert dims == [40., 50., 60.]
    assert out[:96] == raw[:96]
    first = struct.unpack_from("<fff", out, 96)
    assert all(abs(x-y)<1e-6 for x,y in zip(first, (.01,.02,.03)))


def test_generated_urdf_target_needs_no_nonunit_mesh_scale():
    cfg = yaml.safe_load((ROOT/"assets/embodiments/so101-pgripper-sweepick-measured/config.yml").read_text())
    assert cfg["urdf_path"].endswith("/so101_pgripper_sweepick_unit.urdf")
    source = ET.parse(ROOT/"assets/embodiments/so101-pgripper/so101_pgripper.urdf")
    assert sum(mesh.get("filename") == "assets/base_so101_v2.stl"
               for mesh in source.findall(".//mesh")) == 2


def test_staged_randomization_profiles_keep_relative_table_z_constant():
    names = ("background", "light", "camera", "combined")
    profiles = {
        name: yaml.safe_load((
            ROOT/f"env_cfg/task_config/demo_so101_pgripper_sweepick_rand_{name}.yml"
        ).read_text()) for name in names
    }
    for name, cfg in profiles.items():
        rand = cfg["domain_randomization"]
        assert cfg["episode_num"] == 1
        assert cfg["max_seed_attempts"] == 5
        assert rand["random_table_height"] == 0
        assert rand["cluttered_table"] is False
        assert cfg["sweepick_crop_base"] is False
        assert cfg["save_path"].endswith(name)
    assert profiles["background"]["domain_randomization"]["random_background"] is True
    assert profiles["background"]["domain_randomization"]["clean_background_rate"] == 0
    assert profiles["light"]["domain_randomization"]["random_light"] is True
    assert profiles["camera"]["domain_randomization"]["random_head_camera_dis"] == 0.005
    all_on = profiles["combined"]["domain_randomization"]
    assert all_on["random_background"] and all_on["random_light"]
    assert all_on["random_head_camera_dis"] == 0.005
