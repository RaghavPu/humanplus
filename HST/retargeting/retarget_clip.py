"""
Retarget an AMASS/ACCAD SMPL-X clip to G1 Edu joint angles.

Usage:
    python HST/retargeting/retarget_clip.py \
        --clip HST/legged_gym/data/ACCAD/MartialArtsWalksTurns_c3d/E3_-_advance_stageii.npz \
        --out  HST/legged_gym/data/ACCAD_advance_g1_10fps.npy \
        --fps  10

Outputs:
    <out>               (T, 53) float32  — retargeted G1 joint angles at target fps
    <out_stem>_root.npy (T, 7)  float32  — [trans(3), quat_wxyz(4)] for visualization
"""

import argparse
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from HST.retargeting.smplx_to_g1 import SMPLXToG1Retargeter


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (N, 3) → rotation matrix (N, 3, 3) via Rodrigues."""
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)  # (N, 1)
    safe = np.where(angle < 1e-8, np.ones_like(angle), angle)
    axis = aa / safe  # (N, 3)
    angle = angle[..., 0]  # (N,)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c

    R = np.stack([
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c,
    ], axis=-1).reshape(-1, 3, 3)

    # Fix near-zero rotations to identity
    small = np.linalg.norm(aa, axis=-1) < 1e-8
    R[small] = np.eye(3)
    return R


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (N, 3, 3) → quaternion wxyz (N, 4)."""
    N = R.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        m = R[i]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            q[i] = [0.25 / s, (m[2, 1] - m[1, 2]) * s,
                     (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                     (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                     0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    # Normalize
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.astype(np.float32)


# SMPL-X body frame → G1 URDF body frame rotation
# SMPL-X: X=left, Y=up, Z=forward   G1: X=forward, Y=left, Z=up
_R_G_S = np.array([
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.],
], dtype=np.float64)

_R_G_S_T = _R_G_S.T


def smplx_root_to_g1_quat(root_orient_aa: np.ndarray) -> np.ndarray:
    """
    Convert SMPL-X root_orient (axis-angle, N×3) to MuJoCo G1 quaternion (wxyz, N×4).

    SMPL-X root_orient rotates from body frame (Y-up, faces +Z)
    to world frame (Z-up). For the G1 (Z-up, faces +X):
      R_world_g1 = R_world_smplx @ R_G_S^T
    """
    R_world_smplx = _aa_to_rotmat(root_orient_aa)  # (N, 3, 3)
    R_world_g1 = R_world_smplx @ _R_G_S_T          # (N, 3, 3)
    return _rotmat_to_quat_wxyz(R_world_g1)         # (N, 4) wxyz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def retarget_clip(clip_path: str, out_path: str, target_fps: int = 10):
    print(f"Loading: {clip_path}")
    data = np.load(clip_path, allow_pickle=True)

    # Validate
    gender = str(data['gender'])
    model_type = str(data['surface_model_type'])
    src_fps = float(data['mocap_frame_rate'])
    T_full = data['pose_body'].shape[0]

    print(f"  gender={gender}, model={model_type}, fps={src_fps}, frames={T_full}")
    assert model_type == 'smplx', f"Expected smplx, got {model_type}"

    # Downsample
    step = int(round(src_fps / target_fps))
    indices = np.arange(0, T_full, step)
    T = len(indices)
    print(f"  Downsampling {src_fps}fps → {target_fps}fps: {T_full} → {T} frames (step={step})")

    pose_body  = data['pose_body'][indices]    # (T, 63)
    pose_hand  = data['pose_hand'][indices]    # (T, 90) — left(45) + right(45)
    root_orient = data['root_orient'][indices] # (T, 3)
    trans       = data['trans'][indices]       # (T, 3)

    # Split hands
    left_hand  = pose_hand[:, :45]   # (T, 45)
    right_hand = pose_hand[:, 45:]   # (T, 45)

    # Run retargeter
    retargeter = SMPLXToG1Retargeter(urdf_path='g1.urdf', device='cpu')

    q = retargeter(
        body_pose       = torch.from_numpy(pose_body).float(),
        left_hand_pose  = torch.from_numpy(left_hand).float(),
        right_hand_pose = torch.from_numpy(right_hand).float(),
    ).numpy()   # (T, 53)

    print(f"  Output shape: {q.shape}, range: [{q.min():.3f}, {q.max():.3f}]")

    # Ensure .npy extension
    if not out_path.endswith('.npy'):
        out_path += '.npy'

    # Save joint angles
    np.save(out_path, q.astype(np.float32))
    print(f"  Saved: {out_path}")

    # Save root states for visualization: [trans(3), quat_wxyz(4)]
    root_quat = smplx_root_to_g1_quat(root_orient)  # (T, 4)
    root_states = np.concatenate([trans.astype(np.float32), root_quat], axis=-1)  # (T, 7)

    root_path = out_path.replace('.npy', '_root.npy')
    np.save(root_path, root_states)
    print(f"  Saved: {root_path}")

    return q, root_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', type=str,
        default='HST/legged_gym/data/ACCAD/MartialArtsWalksTurns_c3d/E3_-_advance_stageii.npz',
        help='Path to AMASS SMPL-X .npz clip')
    parser.add_argument('--out', type=str,
        default='HST/legged_gym/data/ACCAD_advance_g1_10fps.npy',
        help='Output path for retargeted joint angles')
    parser.add_argument('--fps', type=int, default=10,
        help='Target fps for output sequence')
    args = parser.parse_args()

    retarget_clip(args.clip, args.out, args.fps)
