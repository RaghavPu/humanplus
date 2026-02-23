"""
Visualize H1 robot motion from Isaac Gym training using MuJoCo.

Usage:
    pip install mujoco numpy
    python HST/visualize_mujoco.py --data_dir ~/Downloads/h1_viz
"""
import argparse
import re
import tempfile
import time
import shutil
import numpy as np
import mujoco
import mujoco.viewer
import os


def create_mujoco_scene(urdf_path):
    """Convert URDF to a MuJoCo scene with lights, floor, and free body."""
    with open(urdf_path, 'r') as f:
        content = f.read()

    # Uncomment floating base joint if commented
    content = content.replace(
        '<!-- <link name="world"></link>\n'
        '  <joint name="floating_base_joint" type="floating">\n'
        '    <parent link="world" />\n'
        '    <child link="pelvis" />\n'
        '  </joint> -->',
        '<link name="world"></link>\n'
        '  <joint name="floating_base_joint" type="floating">\n'
        '    <parent link="world" />\n'
        '    <child link="pelvis" />\n'
        '  </joint>'
    )

    # Remove collision elements that have no geometry (empty or only comments)
    content = re.sub(
        r'<collision>\s*(?:<!--[\s\S]*?-->)?\s*</collision>',
        '',
        content
    )

    # Save cleaned URDF
    urdf_dir = os.path.dirname(urdf_path)
    clean_urdf = os.path.join(urdf_dir, 'h1_mujoco_tmp.urdf')
    with open(clean_urdf, 'w') as f:
        f.write(content)

    # Load URDF and save as MJCF XML
    tmp_model = mujoco.MjModel.from_xml_path(clean_urdf)
    mjcf_path = os.path.join(urdf_dir, 'h1_mujoco_tmp.xml')
    mujoco.mj_saveLastXML(mjcf_path, tmp_model)

    # Read the generated MJCF and add lights, floor, skybox
    with open(mjcf_path, 'r') as f:
        mjcf = f.read()

    # Add visual and lighting settings after <mujoco>
    scene_additions = """
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
"""

    # Add floor geom inside worldbody
    floor_geom = '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" condim="3"/>\n    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>\n    <light pos="2 2 3" dir="-1 -1 -2" diffuse="0.5 0.5 0.5"/>\n'

    mjcf = mjcf.replace('<mujoco model=', scene_additions + '\n<mujoco model=', 1)
    # Actually, let's do this more carefully
    # Re-read and reconstruct
    with open(mjcf_path, 'r') as f:
        mjcf = f.read()

    # Insert scene settings after the opening <mujoco ...> tag
    mjcf = re.sub(
        r'(<mujoco model="[^"]*">)',
        r'\1' + scene_additions,
        mjcf
    )

    # Insert floor and lights as first children of <worldbody>
    mjcf = mjcf.replace(
        '<worldbody>',
        '<worldbody>\n' + floor_geom
    )

    # Write final scene
    with open(mjcf_path, 'w') as f:
        f.write(mjcf)

    os.remove(clean_urdf)
    return mjcf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/Downloads/h1_viz'),
                        help='Directory containing joint_positions.npy and root_states.npy')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--loop', action='store_true', help='Loop the playback')
    args = parser.parse_args()

    # Load motion data
    joint_positions = np.load(os.path.join(args.data_dir, 'joint_positions.npy'))
    root_states = np.load(os.path.join(args.data_dir, 'root_states.npy'))

    print(f"Joint data shape: {joint_positions.shape}")  # (num_steps, 19)
    print(f"Root data shape: {root_states.shape}")        # (num_steps, 13)
    num_steps = joint_positions.shape[0]

    # Load H1 URDF (cleaned for MuJoCo)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, 'legged_gym', 'resources', 'robots', 'h1', 'urdf', 'h1.urdf')

    scene_path = create_mujoco_scene(urdf_path)
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
    finally:
        os.remove(scene_path)  # Clean up temp file

    data = mujoco.MjData(model)

    # Print joint info
    print(f"\nMuJoCo model: {model.nq} qpos, {model.nv} qvel, {model.njnt} joints")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  Joint {i}: {name}, qpos_addr={model.jnt_qposadr[i]}")

    # Isaac Gym root_states: [pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)]
    # MuJoCo free joint qpos: [pos(3), quat_wxyz(4)]

    def set_state(step):
        root = root_states[step]
        joints = joint_positions[step]

        # Root position
        data.qpos[0:3] = root[0:3]

        # Quaternion: Isaac Gym (x,y,z,w) -> MuJoCo (w,x,y,z)
        data.qpos[3] = root[6]   # w
        data.qpos[4] = root[3]   # x
        data.qpos[5] = root[4]   # y
        data.qpos[6] = root[5]   # z

        # Joint positions (19 DOFs)
        data.qpos[7:7 + 19] = joints

        mujoco.mj_forward(model, data)

    # Launch interactive viewer
    dt = 0.02 / args.speed  # 50 Hz playback

    print(f"\nPlaying {num_steps} frames at {args.speed}x speed...")
    print("Controls: mouse drag to rotate, scroll to zoom, double-click to track")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            set_state(step)
            viewer.sync()
            time.sleep(dt)

            step += 1
            if step >= num_steps:
                if args.loop:
                    step = 0
                else:
                    print("Playback complete! Close the window to exit.")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
                    break


if __name__ == '__main__':
    main()
