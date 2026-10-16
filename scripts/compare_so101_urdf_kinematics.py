from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


EMBODIMENT = Path(__file__).resolve().parents[1] / "assets/embodiments/so101-pgripper"
PROD = EMBODIMENT / "so101_pgripper.urdf"
CALIB = EMBODIMENT / "so101_pgripper_calib.urdf"

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper_frame_joint",
]


def vec3(text, default):
    if text is None:
        return np.array(default, dtype=float)

    return np.array(
        [float(v) for v in text.split()],
        dtype=float,
    )


def read_joint(root, name):
    joint = root.find(f"./joint[@name='{name}']")

    if joint is None:
        return None

    parent = joint.find("parent")
    child = joint.find("child")
    origin = joint.find("origin")
    axis = joint.find("axis")

    xyz = vec3(
        None if origin is None else origin.get("xyz"),
        [0.0, 0.0, 0.0],
    )

    rpy = vec3(
        None if origin is None else origin.get("rpy"),
        [0.0, 0.0, 0.0],
    )

    axis_v = vec3(
        None if axis is None else axis.get("xyz"),
        [1.0, 0.0, 0.0],
    )

    return {
        "type": joint.get("type"),
        "parent": None if parent is None else parent.get("link"),
        "child": None if child is None else child.get("link"),
        "xyz": xyz,
        "rpy": rpy,
        "axis": axis_v,
    }


prod_root = ET.parse(PROD).getroot()
calib_root = ET.parse(CALIB).getroot()

print("==========================================")
print("SO101 URDF KINEMATICS COMPARISON")
print("==========================================")
print("Production:", PROD)
print("Calibration:", CALIB)
print()

kinematics_ok = True

for name in JOINTS:

    a = read_joint(prod_root, name)
    b = read_joint(calib_root, name)

    print("------------------------------------------")
    print("JOINT:", name)
    print("------------------------------------------")

    if a is None or b is None:
        print("Production:", "FOUND" if a else "MISSING")
        print("Calibration:", "FOUND" if b else "MISSING")
        kinematics_ok = False
        continue

    print("type:")
    print("  production :", a["type"])
    print("  calibration:", b["type"])

    print("parent:")
    print("  production :", a["parent"])
    print("  calibration:", b["parent"])

    print("child:")
    print("  production :", a["child"])
    print("  calibration:", b["child"])

    print("origin xyz:")
    print("  production :", a["xyz"])
    print("  calibration:", b["xyz"])
    print(
        "  max diff   :",
        np.max(np.abs(a["xyz"] - b["xyz"]))
    )

    print("origin rpy:")
    print("  production :", a["rpy"])
    print("  calibration:", b["rpy"])
    print(
        "  max diff   :",
        np.max(np.abs(a["rpy"] - b["rpy"]))
    )

    print("axis:")
    print("  production :", a["axis"])
    print("  calibration:", b["axis"])
    print(
        "  max diff   :",
        np.max(np.abs(a["axis"] - b["axis"]))
    )

    parent_ok = a["parent"] == b["parent"]
    child_ok = a["child"] == b["child"]

    xyz_ok = np.allclose(
        a["xyz"],
        b["xyz"],
        atol=1e-10,
        rtol=0.0,
    )

    rpy_ok = np.allclose(
        a["rpy"],
        b["rpy"],
        atol=1e-10,
        rtol=0.0,
    )

    axis_ok = np.allclose(
        a["axis"],
        b["axis"],
        atol=1e-10,
        rtol=0.0,
    )

    joint_ok = (
        parent_ok
        and child_ok
        and xyz_ok
        and rpy_ok
        and axis_ok
    )

    if not joint_ok:
        kinematics_ok = False

    print(
        "KINEMATICS:",
        "MATCH" if joint_ok else "DIFFERENT"
    )

print()
print("==========================================")

if kinematics_ok:
    print("ALL KINEMATIC PARAMETERS MATCH")
else:
    print("KINEMATIC DIFFERENCE FOUND")

print("==========================================")
