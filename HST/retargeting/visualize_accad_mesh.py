"""
Render an ACCAD/AMASS SMPL-X clip to an MP4 video (no GUI window needed).

Requirements (install in .venv):
  source .venv/bin/activate
  pip install smplx trimesh torch numpy matplotlib

Usage:
  source .venv/bin/activate
  python HST/retargeting/visualize_accad_mesh.py \
      --clip HST/legged_gym/data/ACCAD/MartialArtsWalksTurns_c3d/E3_-_advance_stageii.npz \
      --model-dir models \
      --fps 10 \
      --out accad_motion.mp4
"""

import argparse
import os

import numpy as np
import torch


def load_clip(clip_path: str, target_fps: float):
    data = np.load(clip_path, allow_pickle=True)

    model_type = str(data["surface_model_type"])
    if model_type.lower() != "smplx":
        raise ValueError(f"Expected SMPL-X clip, got: {model_type}")

    src_fps = float(data["mocap_frame_rate"])
    step = max(1, int(round(src_fps / target_fps)))
    indices = np.arange(0, data["pose_body"].shape[0], step)

    pose_body = data["pose_body"][indices].astype(np.float32)
    pose_hand = data["pose_hand"][indices].astype(np.float32)
    root_orient = data["root_orient"][indices].astype(np.float32)
    trans = data["trans"][indices].astype(np.float32)
    gender = str(data["gender"]).lower()

    betas_raw = np.asarray(
        data.get("betas", np.zeros(10, dtype=np.float32)), dtype=np.float32
    ).ravel()
    num_betas = min(10, betas_raw.shape[0]) if betas_raw.size else 10
    betas = betas_raw[:num_betas] if num_betas > 0 else np.zeros(10, dtype=np.float32)

    return {
        "pose_body": pose_body,
        "left_hand": pose_hand[:, :45],
        "right_hand": pose_hand[:, 45:],
        "root_orient": root_orient,
        "trans": trans,
        "gender": gender,
        "betas": betas,
        "T": len(indices),
    }


def render_video(
    clip_path: str,
    model_dir: str,
    out_path: str = "accad_motion.mp4",
    fps: float = 10.0,
):
    import smplx as smplx_lib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    clip = load_clip(clip_path, fps)
    T = clip["T"]
    print(f"Clip: {clip_path}")
    print(f"Frames: {T} @ {fps}fps, gender: {clip['gender']}")

    betas_t = torch.from_numpy(clip["betas"][None, :]).float()
    body_model = smplx_lib.create(
        model_dir,
        model_type="smplx",
        gender=clip["gender"],
        ext="npz",
        use_pca=False,
        num_betas=clip["betas"].shape[0],
        flat_hand_mean=True,
        batch_size=1,
    )
    faces = body_model.faces

    print("Pre-computing mesh vertices...")
    all_verts = []
    with torch.no_grad():
        for k in range(T):
            out = body_model(
                global_orient=torch.from_numpy(clip["root_orient"][k : k + 1]).float(),
                body_pose=torch.from_numpy(clip["pose_body"][k : k + 1]).float(),
                left_hand_pose=torch.from_numpy(clip["left_hand"][k : k + 1]).float(),
                right_hand_pose=torch.from_numpy(clip["right_hand"][k : k + 1]).float(),
                transl=torch.from_numpy(clip["trans"][k : k + 1]).float(),
                betas=betas_t,
            )
            all_verts.append(out.vertices[0].cpu().numpy())
    print(f"Done. {T} frames ready.")

    # Compute global bounds for consistent camera
    all_v = np.concatenate(all_verts, axis=0)
    center = all_v.mean(axis=0)
    span = max(all_v.max(axis=0) - all_v.min(axis=0)) * 0.6

    # Downsample faces for faster rendering (every 4th face)
    face_step = 4
    faces_ds = faces[::face_step]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def draw_frame(k):
        ax.cla()
        ax.set_axis_off()

        verts = all_verts[k]
        polys = verts[faces_ds]

        mesh_col = Poly3DCollection(
            polys, alpha=0.9, edgecolor=(0.3, 0.3, 0.4, 0.15), linewidth=0.1
        )
        mesh_col.set_facecolor((0.5, 0.6, 0.95, 0.9))
        ax.add_collection3d(mesh_col)

        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.view_init(elev=10, azim=120)

        ax.set_title(f"Frame {k+1}/{T}", fontsize=10, pad=0)
        if k % 10 == 0:
            print(f"  Rendering frame {k+1}/{T}")

    print(f"Rendering {T} frames to {out_path}...")
    anim = FuncAnimation(fig, draw_frame, frames=T, interval=1000 / fps)

    writer = FFMpegWriter(fps=fps, metadata={"title": "ACCAD SMPL-X"}, bitrate=3000)
    anim.save(out_path, writer=writer)
    plt.close(fig)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip",
        type=str,
        default="HST/legged_gym/data/ACCAD/MartialArtsWalksTurns_c3d/E3_-_advance_stageii.npz",
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--out", type=str, default="accad_motion.mp4", help="Output video path"
    )
    args = parser.parse_args()

    render_video(
        clip_path=args.clip,
        model_dir=args.model_dir,
        out_path=args.out,
        fps=args.fps,
    )
