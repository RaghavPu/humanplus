"""
Deterministic SMPL-X (AMASS) -> Unitree G1 EDU + Inspire hands retargeter (53 DOF)

- Body: 29 joints
- Hands: 12 + 12 joints (Inspire)
Total: 53 targets per frame.

Input:
  AMASS / SMPL-X "body_pose" axis-angle for joints 1..21 (pelvis excluded):
    body_pose: (B, 63) or (B, 21, 3)
  Optional hand poses:
    left_hand_pose/right_hand_pose: (B, 45) or (B, 15, 3)

Output:
  q: (B, 53) in JOINT_NAMES order.

Notes:
- Deterministic: no randomness, purely analytic.
- Uses URDF axes + limits to pick correct signs for hinge joints and Inspire fingers.
- Uses swing-twist sequential decomposition for 3-DoF groups (hip, shoulder, waist, wrist).
- Still not "perfect URDF kinematic-frame alignment" (needs per-joint A_i calibration),
  but robust enough for HumanPlus-A style conditioning and generating training targets.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch


# ============================================================================
# 0) Output joint order (53)
# ============================================================================

JOINT_NAMES: List[str] = [
    # left leg (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # right leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # left hand (12)
    "left_thumb_1_joint", "left_thumb_2_joint", "left_thumb_3_joint", "left_thumb_4_joint",
    "left_index_1_joint", "left_index_2_joint",
    "left_middle_1_joint", "left_middle_2_joint",
    "left_ring_1_joint", "left_ring_2_joint",
    "left_little_1_joint", "left_little_2_joint",
    # right hand (12)
    "right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint",
    "right_index_1_joint", "right_index_2_joint",
    "right_middle_1_joint", "right_middle_2_joint",
    "right_ring_1_joint", "right_ring_2_joint",
    "right_little_1_joint", "right_little_2_joint",
]

NUM_BODY = 29
NUM_HAND = 12
NUM_TOTAL = 53


# ============================================================================
# 1) SMPL-X -> G1 global basis swap
# ============================================================================
# SMPL-X convention: Y-up, +Z forward, +X left
# G1 URDF convention: Z-up, +X forward, +Y left
#
# Map SMPL basis vectors to G1:
#   SMPL +X (left)    -> G1 +Y (left)
#   SMPL +Y (up)      -> G1 +Z (up)
#   SMPL +Z (forward) -> G1 +X (forward)
R_G_S = torch.tensor(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)
R_G_S_T = R_G_S.T


def conjugate_global_basis(R: torch.Tensor) -> torch.Tensor:
    """
    Conjugate rotation matrices from SMPL global basis to G1 global basis.
    R_g1 = R_G_S * R_smpl * R_G_S^T
    """
    return R_G_S.to(R.device) @ R @ R_G_S_T.to(R.device)


# ============================================================================
# 2) URDF parsing (axes + limits)
# ============================================================================

@dataclass(frozen=True)
class JointInfo:
    name: str
    axis: torch.Tensor      # (3,) as floats, in joint frame (URDF axis)
    limit_lower: float
    limit_upper: float


def _parse_urdf_joint_info(urdf_path: str) -> Dict[str, JointInfo]:
    """
    Parse URDF and return JointInfo for every joint that has axis+limits.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joint_info: Dict[str, JointInfo] = {}

    for j in root.findall("joint"):
        name = j.attrib.get("name", "")
        axis_el = j.find("axis")
        lim_el = j.find("limit")
        if axis_el is None or lim_el is None:
            continue
        if "xyz" not in axis_el.attrib:
            continue
        xyz = [float(x) for x in axis_el.attrib["xyz"].split()]
        axis = torch.tensor(xyz, dtype=torch.float32)
        lower = float(lim_el.attrib.get("lower", "0.0"))
        upper = float(lim_el.attrib.get("upper", "0.0"))
        joint_info[name] = JointInfo(name=name, axis=axis, limit_lower=lower, limit_upper=upper)

    return joint_info


# ============================================================================
# 3) Math: axis-angle, rotmat <-> quat, swing-twist angle about axis
# ============================================================================

def axis_angle_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues axis-angle -> rotation matrices.
    aa: (..., 3) or (..., N, 3)
    returns: (..., 3, 3) or (..., N, 3, 3)
    """
    shape = aa.shape
    aa_flat = aa.reshape(-1, 3)

    angle = aa_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa_flat / angle
    angle = angle.squeeze(-1)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = torch.cos(angle), torch.sin(angle)
    t = 1.0 - c

    R = torch.stack(
        [
            t * x * x + c,     t * x * y - s * z, t * x * z + s * y,
            t * x * y + s * z, t * y * y + c,     t * y * z - s * x,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    return R.reshape(*shape[:-1], 3, 3)


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (w,x,y,z), robust.
    R: (...,3,3)
    """
    # Based on standard numerically stable conversion
    device = R.device
    dtype = R.dtype
    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    trace = r00 + r11 + r22
    qw = torch.empty_like(trace)
    qx = torch.empty_like(trace)
    qy = torch.empty_like(trace)
    qz = torch.empty_like(trace)

    # masks
    m0 = trace > 0.0
    m1 = (r00 >= r11) & (r00 >= r22) & (~m0)
    m2 = (r11 > r00) & (r11 >= r22) & (~m0)
    m3 = (~m0) & (~m1) & (~m2)

    # trace > 0
    t = torch.sqrt(trace[m0] + 1.0) * 2.0
    qw[m0] = 0.25 * t
    qx[m0] = (r21[m0] - r12[m0]) / t
    qy[m0] = (r02[m0] - r20[m0]) / t
    qz[m0] = (r10[m0] - r01[m0]) / t

    # r00 largest
    t = torch.sqrt(1.0 + r00[m1] - r11[m1] - r22[m1]) * 2.0
    qw[m1] = (r21[m1] - r12[m1]) / t
    qx[m1] = 0.25 * t
    qy[m1] = (r01[m1] + r10[m1]) / t
    qz[m1] = (r02[m1] + r20[m1]) / t

    # r11 largest
    t = torch.sqrt(1.0 + r11[m2] - r00[m2] - r22[m2]) * 2.0
    qw[m2] = (r02[m2] - r20[m2]) / t
    qx[m2] = (r01[m2] + r10[m2]) / t
    qy[m2] = 0.25 * t
    qz[m2] = (r12[m2] + r21[m2]) / t

    # r22 largest
    t = torch.sqrt(1.0 + r22[m3] - r00[m3] - r11[m3]) * 2.0
    qw[m3] = (r10[m3] - r01[m3]) / t
    qx[m3] = (r02[m3] + r20[m3]) / t
    qy[m3] = (r12[m3] + r21[m3]) / t
    qz[m3] = 0.25 * t

    q = torch.stack([qw, qx, qy, qz], dim=-1).to(device=device, dtype=dtype)
    # normalize
    q = q / (q.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    return q


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply (wxyz)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def twist_angle_about_axis(R: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Extract the twist angle (rotation) of R about a given axis.
    R: (...,3,3)
    axis: (3,) in the same frame as R
    Returns angle in [-pi, pi], deterministic.
    """
    axis = axis.to(R.device, R.dtype)
    axis = axis / axis.norm().clamp(min=1e-8)

    q = rotmat_to_quat(R)  # (...,4) wxyz
    v = q[..., 1:]         # (...,3)
    w = q[..., 0]          # (...)

    # project v onto axis
    proj = (v * axis).sum(dim=-1, keepdim=True) * axis  # (...,3)
    q_twist = torch.cat([w.unsqueeze(-1), proj], dim=-1)  # (...,4)
    q_twist = q_twist / q_twist.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # angle = 2*atan2(|v|, w), sign from dot(v, axis)
    vw = q_twist[..., 0]
    vv = q_twist[..., 1:]
    sin_half = vv.norm(dim=-1).clamp(min=1e-8)
    angle = 2.0 * torch.atan2(sin_half, vw.clamp(min=-1.0, max=1.0))
    sign = torch.sign((vv * axis).sum(dim=-1))
    angle = angle * sign

    # wrap to [-pi, pi]
    angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return angle


def rot_axis_angle(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Build rotation matrix R(axis, theta).
    axis: (3,)
    theta: (...,)
    returns (...,3,3)
    """
    axis = axis / axis.norm().clamp(min=1e-8)
    aa = axis.view(1, 3).to(theta.device, theta.dtype) * theta.view(-1, 1)
    return axis_angle_to_rotmat(aa).reshape(*theta.shape, 3, 3)


def sequential_decompose(R: torch.Tensor, axes: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Deterministically decompose R into sequential hinge rotations about axes[0], axes[1], ...
    using twist extraction at each step:
      theta1 = twist_angle(R, a1)
      R1 = R(a1, theta1)
      R_rem = R1^T * R
      theta2 = twist_angle(R_rem, a2)
      ...
    R: (B,3,3)
    axes: list of (3,)
    Returns list of angles [theta1, theta2, ...], each (B,)
    """
    B = R.shape[0]
    R_rem = R
    thetas: List[torch.Tensor] = []
    for a in axes:
        th = twist_angle_about_axis(R_rem, a)
        thetas.append(th)
        R_th = rot_axis_angle(a.to(R.device, R.dtype), th)  # (B,3,3)
        R_rem = torch.transpose(R_th, -1, -2) @ R_rem
    return thetas


# ============================================================================
# 4) Shoulder "stand-down" offset (stable)
# ============================================================================
def _rot_between_vecs(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    v1 = v1 / (v1.norm() + eps)
    v2 = v2 / (v2.norm() + eps)
    dot = torch.dot(v1, v2).clamp(-1.0, 1.0)

    if dot > 1.0 - eps:
        return torch.eye(3, dtype=v1.dtype)

    if dot < -1.0 + eps:
        helper = torch.tensor([1.0, 0.0, 0.0], dtype=v1.dtype)
        if torch.abs(v1[0]) > 0.9:
            helper = torch.tensor([0.0, 1.0, 0.0], dtype=v1.dtype)
        axis = torch.linalg.cross(v1, helper)
        axis = axis / (axis.norm() + eps)
        return axis_angle_to_rotmat(axis * torch.pi)

    cross = torch.linalg.cross(v1, v2)
    K = torch.tensor(
        [
            [0.0, -cross[2].item(), cross[1].item()],
            [cross[2].item(), 0.0, -cross[0].item()],
            [-cross[1].item(), cross[0].item(), 0.0],
        ],
        dtype=v1.dtype,
    )
    I = torch.eye(3, dtype=v1.dtype)
    return I + K + (K @ K) / (1.0 + dot)


# Rest-pose upper arm directions from SMPLX_NEUTRAL (your constants)
_REST_DIR_L = torch.tensor([0.9498, -0.2697, -0.1587], dtype=torch.float32)
_REST_DIR_R = torch.tensor([-0.9865, -0.1328, -0.0963], dtype=torch.float32)
_DOWN_SMPLX = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)

_R_STAND_L = _rot_between_vecs(_REST_DIR_L, _DOWN_SMPLX)
_R_STAND_R = _rot_between_vecs(_REST_DIR_R, _DOWN_SMPLX)

# conjugate into G1 basis and transpose
_R_STAND_L_G1_T = R_G_S @ _R_STAND_L.T @ R_G_S_T
_R_STAND_R_G1_T = R_G_S @ _R_STAND_R.T @ R_G_S_T


# ============================================================================
# 5) SMPL-X joint indexing assumptions (AMASS SMPL-X body_pose excludes pelvis)
# ============================================================================
# body_rotmats[:, i] corresponds to SMPL-X joint (i+1):
# 0:L_hip, 1:R_hip, 2:spine1, 3:L_knee, 4:R_knee, 5:spine2,
# 6:L_ankle, 7:R_ankle, 8:spine3, 9:L_foot, 10:R_foot,
# 11:neck, 12:L_collar, 13:R_collar, 14:head,
# 15:L_shoulder, 16:R_shoulder, 17:L_elbow, 18:R_elbow,
# 19:L_wrist, 20:R_wrist

# hand_pose rotmats order (15):
# index 0-2, middle 3-5, pinky 6-8, ring 9-11, thumb 12-14


# ============================================================================
# 6) Retargeter implementation
# ============================================================================

class SMPLXToG1Retargeter:
    def __init__(self, urdf_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.joint_info = _parse_urdf_joint_info(urdf_path)

        # sanity: ensure all output joints exist in URDF joint_info
        missing = [n for n in JOINT_NAMES if n not in self.joint_info]
        if len(missing) > 0:
            raise ValueError(f"URDF missing joint info for: {missing[:10]}{'...' if len(missing)>10 else ''}")

        # precompute axis and limits in output order
        self.axes = torch.stack([self.joint_info[n].axis for n in JOINT_NAMES], dim=0).to(self.device)
        self.lower = torch.tensor([self.joint_info[n].limit_lower for n in JOINT_NAMES], dtype=torch.float32, device=self.device)
        self.upper = torch.tensor([self.joint_info[n].limit_upper for n in JOINT_NAMES], dtype=torch.float32, device=self.device)

        # Inspire hands are 0..max style. We'll enforce q>=0 for those joints only.
        self.hand_mask = torch.zeros(NUM_TOTAL, dtype=torch.bool, device=self.device)
        self.hand_mask[29:] = True

        # Axes for groups from URDF (these are in joint frame and are ±unit x/y/z)
        # We'll use them directly as twist axes in the group frame.
        self._setup_group_axes()

    def _setup_group_axes(self):
        # Hip group: pitch, roll, yaw
        self.hip_axes_L = [
            self.joint_info["left_hip_pitch_joint"].axis,
            self.joint_info["left_hip_roll_joint"].axis,
            self.joint_info["left_hip_yaw_joint"].axis,
        ]
        self.hip_axes_R = [
            self.joint_info["right_hip_pitch_joint"].axis,
            self.joint_info["right_hip_roll_joint"].axis,
            self.joint_info["right_hip_yaw_joint"].axis,
        ]

        # Ankle group: pitch, roll
        self.ankle_axes_L = [
            self.joint_info["left_ankle_pitch_joint"].axis,
            self.joint_info["left_ankle_roll_joint"].axis,
        ]
        self.ankle_axes_R = [
            self.joint_info["right_ankle_pitch_joint"].axis,
            self.joint_info["right_ankle_roll_joint"].axis,
        ]

        # Waist group: yaw, roll, pitch
        self.waist_axes = [
            self.joint_info["waist_yaw_joint"].axis,
            self.joint_info["waist_roll_joint"].axis,
            self.joint_info["waist_pitch_joint"].axis,
        ]

        # Shoulder group: pitch, roll, yaw
        self.shoulder_axes_L = [
            self.joint_info["left_shoulder_pitch_joint"].axis,
            self.joint_info["left_shoulder_roll_joint"].axis,
            self.joint_info["left_shoulder_yaw_joint"].axis,
        ]
        self.shoulder_axes_R = [
            self.joint_info["right_shoulder_pitch_joint"].axis,
            self.joint_info["right_shoulder_roll_joint"].axis,
            self.joint_info["right_shoulder_yaw_joint"].axis,
        ]

        # Wrist group: roll, pitch, yaw
        self.wrist_axes_L = [
            self.joint_info["left_wrist_roll_joint"].axis,
            self.joint_info["left_wrist_pitch_joint"].axis,
            self.joint_info["left_wrist_yaw_joint"].axis,
        ]
        self.wrist_axes_R = [
            self.joint_info["right_wrist_roll_joint"].axis,
            self.joint_info["right_wrist_pitch_joint"].axis,
            self.joint_info["right_wrist_yaw_joint"].axis,
        ]

        # Single-axis (knee, elbow) from URDF
        self.knee_axis_L = self.joint_info["left_knee_joint"].axis
        self.knee_axis_R = self.joint_info["right_knee_joint"].axis
        self.elbow_axis_L = self.joint_info["left_elbow_joint"].axis
        self.elbow_axis_R = self.joint_info["right_elbow_joint"].axis

    def _clamp(self, q: torch.Tensor) -> torch.Tensor:
        q = torch.max(q, self.lower.view(1, -1))
        q = torch.min(q, self.upper.view(1, -1))
        # Inspire: clamp negatives to 0
        q[:, 29:] = torch.clamp(q[:, 29:], min=0.0)
        return q

    def _retarget_body(self, body_rotmats: torch.Tensor) -> torch.Tensor:
        """
        body_rotmats: (B,21,3,3) local SMPL-X rotations for joints 1..21 (pelvis excluded).
        Returns: (B,29) body joint targets.
        """
        B = body_rotmats.shape[0]
        device = body_rotmats.device
        q = torch.zeros(B, NUM_BODY, device=device, dtype=torch.float32)

        # global basis swap on each local rot (approx; good for conditioning)
        R = conjugate_global_basis(body_rotmats)

        # --- Left leg ---
        # SMPL L_hip: R[:,0]
        hip_L = R[:, 0]
        th = sequential_decompose(hip_L, [a.to(device) for a in self.hip_axes_L])
        q[:, 0], q[:, 1], q[:, 2] = th[0], th[1], th[2]

        # knee: SMPL L_knee is R[:,3]
        q[:, 3] = twist_angle_about_axis(R[:, 3], self.knee_axis_L.to(device))

        # ankle: SMPL L_ankle is R[:,6]
        ankle_L = R[:, 6]
        th = sequential_decompose(ankle_L, [a.to(device) for a in self.ankle_axes_L])
        q[:, 4], q[:, 5] = th[0], th[1]

        # --- Right leg ---
        hip_R = R[:, 1]
        th = sequential_decompose(hip_R, [a.to(device) for a in self.hip_axes_R])
        q[:, 6], q[:, 7], q[:, 8] = th[0], th[1], th[2]

        q[:, 9] = twist_angle_about_axis(R[:, 4], self.knee_axis_R.to(device))

        ankle_R = R[:, 7]
        th = sequential_decompose(ankle_R, [a.to(device) for a in self.ankle_axes_R])
        q[:, 10], q[:, 11] = th[0], th[1]

        # --- Waist: spine1 @ spine2 @ spine3 ---
        R_spine = R[:, 2] @ R[:, 5] @ R[:, 8]
        th = sequential_decompose(R_spine, [a.to(device) for a in self.waist_axes])
        q[:, 12], q[:, 13], q[:, 14] = th[0], th[1], th[2]

        # --- Left shoulder ---
        # Combine collar (idx 12) + shoulder (idx 15) in SMPL-X frame,
        # basis-swap to G1 torso frame, then compose with T-pose base.
        R_s_L = body_rotmats[:, 12] @ body_rotmats[:, 15]
        R_g1_L = R_G_S.to(device) @ R_s_L @ R_G_S_T.to(device)
        R_tpose_L = rot_axis_angle(
            self.joint_info["left_shoulder_roll_joint"].axis.to(device),
            torch.tensor([math.pi / 2], device=device),
        ).expand(B, -1, -1)
        R_total_L = R_g1_L @ R_tpose_L
        th = sequential_decompose(R_total_L, [a.to(device) for a in self.shoulder_axes_L])
        q[:, 15], q[:, 16], q[:, 17] = th[0], th[1], th[2]

        # --- Right shoulder ---
        R_s_R = body_rotmats[:, 13] @ body_rotmats[:, 16]
        R_g1_R = R_G_S.to(device) @ R_s_R @ R_G_S_T.to(device)
        R_tpose_R = rot_axis_angle(
            self.joint_info["right_shoulder_roll_joint"].axis.to(device),
            torch.tensor([-math.pi / 2], device=device),
        ).expand(B, -1, -1)
        R_total_R = R_g1_R @ R_tpose_R
        th = sequential_decompose(R_total_R, [a.to(device) for a in self.shoulder_axes_R])
        q[:, 22], q[:, 23], q[:, 24] = th[0], th[1], th[2]

        # --- Left elbow ---
        # Frame change: SMPL-X arm frame -> G1 arm frame using actual shoulder
        M_L = R_total_L.transpose(-1, -2) @ R_G_S.to(device) @ R_s_L
        R_e_g1_L = M_L @ body_rotmats[:, 17] @ M_L.transpose(-1, -2)
        delta_L = twist_angle_about_axis(R_e_g1_L, self.elbow_axis_L.to(device))
        q[:, 18] = math.pi / 2 + delta_L

        # --- Right elbow ---
        M_R = R_total_R.transpose(-1, -2) @ R_G_S.to(device) @ R_s_R
        R_e_g1_R = M_R @ body_rotmats[:, 18] @ M_R.transpose(-1, -2)
        delta_R = twist_angle_about_axis(R_e_g1_R, self.elbow_axis_R.to(device))
        q[:, 25] = math.pi / 2 + delta_R

        # Clamp using URDF body limits (from output vectors)
        q_full = torch.zeros(B, NUM_TOTAL, device=device)
        q_full[:, :NUM_BODY] = q
        q_full = self._clamp(q_full)
        return q_full[:, :NUM_BODY]

    def _retarget_hand(self, hand_rotmats: torch.Tensor, side: str) -> torch.Tensor:
        """TODO: rewrite hand retargeting."""
        B = hand_rotmats.shape[0]
        return torch.zeros(B, 12, device=hand_rotmats.device)

    def __call__(
        self,
        body_pose: torch.Tensor,
        left_hand_pose: Optional[torch.Tensor] = None,
        right_hand_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          body_pose: (B,63) or (B,21,3) axis-angle for SMPL-X joints 1..21 (pelvis excluded)
          left_hand_pose/right_hand_pose: (B,45) or (B,15,3) axis-angle for 15 hand joints
        Returns:
          q: (B,53)
        """
        body_pose = body_pose.to(self.device)
        B = body_pose.shape[0]

        if body_pose.ndim == 2:
            body_pose = body_pose.reshape(B, 21, 3)

        body_rotmats = axis_angle_to_rotmat(body_pose)  # (B,21,3,3)
        q_body = self._retarget_body(body_rotmats)       # (B,29)

        if left_hand_pose is not None:
            lh = left_hand_pose.to(self.device)
            if lh.ndim == 2:
                lh = lh.reshape(B, 15, 3)
            lh_R = axis_angle_to_rotmat(lh)
            q_lh = self._retarget_hand(lh_R, "left")
        else:
            q_lh = torch.zeros(B, 12, device=self.device)

        if right_hand_pose is not None:
            rh = right_hand_pose.to(self.device)
            if rh.ndim == 2:
                rh = rh.reshape(B, 15, 3)
            rh_R = axis_angle_to_rotmat(rh)
            q_rh = self._retarget_hand(rh_R, "right")
        else:
            q_rh = torch.zeros(B, 12, device=self.device)

        q = torch.cat([q_body, q_lh, q_rh], dim=-1)  # (B,53)
        # Final clamp (belt & suspenders)
        q_full = torch.zeros(B, NUM_TOTAL, device=self.device)
        q_full[:, :] = q
        q_full = self._clamp(q_full)
        return q_full

    def joint_names(self) -> List[str]:
        return JOINT_NAMES


# ============================================================================
# 7) Minimal usage example
# ============================================================================

if __name__ == "__main__":
    # Example:
    #   urdf_path = "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
    #   retarget = SMPLXToG1Retargeter(urdf_path, device="cpu")
    #   body_pose = torch.zeros(1, 63)  # neutral
    #   q = retarget(body_pose)         # (1,53)
    #   print(q)
    pass