"""
Visualize retargeted G1 Edu joint angles in MuJoCo.

Usage:
    python HST/retargeting/visualize_g1.py --loop
"""

import argparse
import os
import sys
import time
import tempfile
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# URDF preparation (robust XML edit, no regex)
# ---------------------------------------------------------------------------

def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print indentation for ElementTree output."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if not elem.tail or not elem.tail.strip():
            elem.tail = i


def _make_origin(xyz="0 0 0", rpy="0 0 0") -> ET.Element:
    o = ET.Element("origin")
    o.set("xyz", xyz)
    o.set("rpy", rpy)
    return o


def _make_link(name: str) -> ET.Element:
    l = ET.Element("link")
    l.set("name", name)
    return l


def _make_floating_joint(parent: str, child: str, name="floating_base_joint") -> ET.Element:
    j = ET.Element("joint")
    j.set("name", name)
    j.set("type", "floating")
    j.append(_make_origin())
    p = ET.Element("parent"); p.set("link", parent)
    c = ET.Element("child");  c.set("link", child)
    j.append(p)
    j.append(c)
    return j


def _replace_mesh_geoms_with_sphere(robot: ET.Element, radius: str = "0.03") -> None:
    """
    Replace any <geometry><mesh .../></geometry> with <geometry><sphere radius="..."/></geometry>
    for both visual and collision. Handles non-self-closing <mesh> too.
    """
    for geom in robot.iter("geometry"):
        # Find a mesh child anywhere directly under geometry
        mesh = None
        for child in list(geom):
            if child.tag == "mesh":
                mesh = child
                break
        if mesh is None:
            continue

        # Clear geometry children and replace with sphere
        for child in list(geom):
            geom.remove(child)
        sphere = ET.Element("sphere")
        sphere.set("radius", radius)
        geom.append(sphere)


def _resolve_mesh_path(urdf_dir: str, mesh_filename: str) -> str:
    """Resolve a URDF mesh filename to an absolute local path when possible."""
    # Minimal handling for package:// URIs: strip the scheme and treat as relative.
    if mesh_filename.startswith("package://"):
        mesh_filename = mesh_filename[len("package://"):]
    if os.path.isabs(mesh_filename):
        return mesh_filename
    return os.path.normpath(os.path.join(urdf_dir, mesh_filename))


def _find_missing_mesh_files(robot: ET.Element, urdf_dir: str):
    """Return unique missing mesh filenames referenced by the URDF."""
    missing = []
    seen = set()
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        resolved = _resolve_mesh_path(urdf_dir, filename)
        if not os.path.exists(resolved) and filename not in seen:
            missing.append(filename)
            seen.add(filename)
    return missing


def _ensure_joint_has_origin(robot: ET.Element) -> None:
    """
    Add <origin xyz="0 0 0" rpy="0 0 0"/> to any joint missing origin.
    Some URDF consumers assume identity; MuJoCo importer can be picky.
    """
    for joint in robot.findall("joint"):
        has_origin = any(ch.tag == "origin" for ch in list(joint))
        if not has_origin:
            # Put origin first for readability (common URDF style)
            joint.insert(0, _make_origin())


def _insert_world_and_floating_base(robot: ET.Element, base_link: str = "pelvis") -> None:
    """
    Add:
      <link name="world"/>
      <joint type="floating" parent=world child=pelvis .../>
    but ensure the floating joint comes AFTER the pelvis link definition.
    """
    # Collect top-level children (direct under <robot>)
    children = list(robot)

    # If world already exists, skip making it twice
    world_exists = any(ch.tag == "link" and ch.get("name") == "world" for ch in children)
    if not world_exists:
        # Insert <link name="world"/> near the top (before other links/joints)
        # Prefer: before the first <link> if possible, else at very start.
        first_link_idx = next((i for i, ch in enumerate(children) if ch.tag == "link"), 0)
        robot.insert(first_link_idx, _make_link("world"))
        children = list(robot)  # refresh

    # Find pelvis link index (must exist)
    pelvis_idx = next(
        (i for i, ch in enumerate(children) if ch.tag == "link" and ch.get("name") == base_link),
        None
    )
    if pelvis_idx is None:
        raise ValueError(f"Base link '{base_link}' not found in URDF.")

    # Avoid inserting if already exists
    fb_exists = any(ch.tag == "joint" and ch.get("name") == "floating_base_joint" for ch in children)
    if fb_exists:
        return

    # Insert floating joint *after pelvis link definition*
    robot.insert(pelvis_idx + 1, _make_floating_joint("world", base_link))


def prepare_urdf_for_mujoco(urdf_path: str, use_spheres: bool = False) -> str:
    """
    Returns path to a temp patched URDF file (written next to original so relative paths resolve).
    """
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    if robot.tag != "robot":
        raise ValueError("Not a valid URDF: root tag is not <robot>")

    # 1) Make joints more explicit
    _ensure_joint_has_origin(robot)

    # 2) Add floating base in a parser-friendly order
    _insert_world_and_floating_base(robot, base_link="pelvis")

    # 3) Optional debug geometry replacement
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    if not use_spheres:
        missing_meshes = _find_missing_mesh_files(robot, urdf_dir)
        if missing_meshes:
            preview = ", ".join(missing_meshes[:4])
            if len(missing_meshes) > 4:
                preview += ", ..."
            print(
                f"Warning: {len(missing_meshes)} mesh files are missing ({preview}). "
                "Falling back to spheres. Use --spheres to force this mode."
            )
            use_spheres = True
    if use_spheres:
        _replace_mesh_geoms_with_sphere(robot, radius="0.03")

    # Pretty print (optional)
    _indent_xml(robot)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", dir=urdf_dir, delete=False)
    tree.write(tmp.name, encoding="unicode", xml_declaration=True)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

def visualize(
    joints_path: str,
    root_path: str,
    urdf_path: str,
    fps: float = 10.0,
    loop: bool = False,
    use_spheres: bool = False,
):
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        raise RuntimeError("MuJoCo not installed. Run: pip install mujoco")

    joints = np.load(joints_path)  # (T, 53)
    root   = np.load(root_path)    # (T, 7): [tx, ty, tz, qw, qx, qy, qz]
    T, num_joints = joints.shape
    print(f"Motion: {T} frames @ {fps}fps, {num_joints} joints")
    if num_joints != 53:
        raise ValueError(f"Expected 53 joints, got {num_joints}")

    print(f"Loading URDF: {urdf_path}")
    tmp_urdf = prepare_urdf_for_mujoco(urdf_path, use_spheres=use_spheres)
    print(f"Patched URDF: {tmp_urdf}")

    import re as _re

    # Load URDF → MuJoCo, then save as MJCF so we can inject scene elements
    tmp_model = mujoco.MjModel.from_xml_path(tmp_urdf)
    mjcf_path = tmp_urdf.replace(".urdf", ".xml")
    mujoco.mj_saveLastXML(mjcf_path, tmp_model)
    os.unlink(tmp_urdf)

    with open(mjcf_path, "r") as f:
        mjcf = f.read()

    scene_xml = """
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>
"""
    floor_xml = (
        '    <geom name="floor" size="0 0 0.05" type="plane" '
        'material="groundplane" condim="3"/>\n'
        '    <light pos="0 0 3" dir="0 0 -1" directional="true" '
        'diffuse="0.8 0.8 0.8"/>\n'
        '    <light pos="2 2 3" dir="-1 -1 -2" diffuse="0.5 0.5 0.5"/>\n'
    )

    mjcf = _re.sub(
        r'(<mujoco model="[^"]*">)',
        r'\1' + scene_xml,
        mjcf,
    )
    mjcf = mjcf.replace("<worldbody>", "<worldbody>\n" + floor_xml)

    with open(mjcf_path, "w") as f:
        f.write(mjcf)

    try:
        model = mujoco.MjModel.from_xml_path(mjcf_path)
    finally:
        os.unlink(mjcf_path)

    data = mujoco.MjData(model)

    # qpos layout:  [0:3] translation, [3:7] quat wxyz, [7:60] 53 joints
    def set_frame(k: int):
        r = root[k]
        q = joints[k]
        data.qpos[0:3] = r[0:3]
        data.qpos[3:7] = r[3:7]
        data.qpos[7:7 + 53] = q
        mujoco.mj_forward(model, data)

    dt = 1.0 / fps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("pelvis").id
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0

        k = 0
        while viewer.is_running():
            t0 = time.time()
            set_frame(k)
            viewer.sync()

            k += 1
            if k >= T:
                if loop:
                    k = 0
                else:
                    break

            sleep_time = dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--joints", type=str, default="HST/legged_gym/data/ACCAD_advance_g1_10fps.npy")
    p.add_argument("--root", type=str, default="HST/legged_gym/data/ACCAD_advance_g1_10fps_root.npy")
    p.add_argument("--urdf", type=str, default="g1.urdf")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument(
        "--spheres",
        action="store_true",
        help="Replace mesh geometry with spheres (debug mode). Default keeps URDF meshes.",
    )
    args = p.parse_args()

    visualize(
        args.joints,
        args.root,
        args.urdf,
        fps=args.fps,
        loop=args.loop,
        use_spheres=args.spheres,
    )