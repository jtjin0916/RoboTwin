"""CPU-only smoke settings: camera jitter toggles and external texture checks."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def config(suffix):
    return yaml.safe_load((ROOT / "env_cfg/task_config" / f"{suffix}.yml").read_text())


def test_camera_on_off_profiles_keep_other_randomization_identical():
    on = config("demo_so101_pgripper_sweepick_rand_combined")
    off = config("demo_so101_pgripper_sweepick_rand_combined_no_camera")
    dr_on, dr_off = on["domain_randomization"], off["domain_randomization"]
    assert dr_on["random_head_camera_dis"] == 0.005
    assert dr_off["random_head_camera_dis"] == 0
    for key in ("random_background","random_light","random_table_height","cluttered_table"):
        assert dr_on[key] == dr_off[key]
    assert dr_on["random_background"] is True
    assert dr_on["random_light"] is True
    assert dr_on["random_table_height"] == 0
    assert dr_on["cluttered_table"] is False
    assert on["save_path"] != off["save_path"]


def test_independent_unit_scale_regression_dataset():
    baseline = config("demo_so101_pgripper_sweepick_measured_smoke")
    retest = config("demo_so101_pgripper_sweepick_unit_scale_retest")
    assert baseline["save_path"] != retest["save_path"]
    assert retest["domain_randomization"]["random_head_camera_dis"] == 0


def test_texture_selector_checks_external_png_assets_without_download():
    source = (ROOT / "envs/_base_task.py").read_text()
    assert "Missing RoboTwin background textures" in source
    assert 'directory_path.glob("*.png")' in source
