"""
isaacgym.torch_utils  ── Isaac Lab compatibility shim
=====================================================
All functions are implemented in pure PyTorch with NO dependency on the Isaac
Gym C-extension.  They are therefore importable and runnable with only PyTorch
installed.

Quaternion convention: **xyzw** (same as Isaac Gym Preview Release).
Note: Isaac Lab / Isaac Sim uses **wxyz** – you must reorder when calling
Isaac Lab APIs.  Helper `xyzw_to_wxyz` / `wxyz_to_xyzw` are provided.

Usage (drop-in replacement for the old import):
    from isaacgym.torch_utils import *      # unchanged import line
    # or
    from isaacgym import torch_utils
"""

from __future__ import annotations
import math
import torch
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Tensor helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_torch(x, dtype: torch.dtype = torch.float, device: str = "cpu",
             requires_grad: bool = False) -> torch.Tensor:
    """Convert list / ndarray / scalar → tensor.  Mirrors isaacgym.torch_utils.to_torch."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def torch_rand_float(lower: float, upper: float, shape, device) -> torch.Tensor:
    """Uniform random float tensor in [lower, upper)."""
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def tensor_clamp(t: torch.Tensor, min_t: torch.Tensor,
                 max_t: torch.Tensor) -> torch.Tensor:
    """Element-wise clamp with tensor bounds (like np.clip but for tensors)."""
    return torch.max(torch.min(t, max_t), min_t)


def scale(x: torch.Tensor, lower: torch.Tensor,
          upper: torch.Tensor) -> torch.Tensor:
    """Scale x from [-1, 1] → [lower, upper].
    Inverse of unscale().
    """
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x: torch.Tensor, lower: torch.Tensor,
            upper: torch.Tensor) -> torch.Tensor:
    """Scale x from [lower, upper] → [-1, 1].
    Inverse of scale().
    """
    return (2.0 * x - upper - lower) / (upper - lower)


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion helpers  (xyzw convention, matching Isaac Gym)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """L2-normalise along the last dimension."""
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (xyzw).
    Shapes: (..., 4) × (..., 4) → (..., 4)
    """
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (inverse for unit quaternion) in xyzw.  Negates x,y,z."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def quat_unit(a: torch.Tensor) -> torch.Tensor:
    """Normalise quaternion."""
    return normalize(a)


def quat_from_angle_axis(angle: torch.Tensor,
                          axis: torch.Tensor) -> torch.Tensor:
    """Build xyzw quaternion from rotation angle (rad) and axis vector (..., 3)."""
    theta = (angle / 2.0).unsqueeze(-1)          # (..., 1)
    axis = normalize(axis)
    xyz = axis * torch.sin(theta)
    w   = torch.cos(theta)
    return torch.cat([xyz, w], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3-D vector v by quaternion q (xyzw).
    Shapes: (..., 4), (..., 3) → (..., 3)
    """
    # sandwich product: q * [v, 0] * q^-1
    shape = q.shape
    q_w = q[..., 3:4]
    q_xyz = q[..., :3]
    # formula: v' = v + 2*w*(q_xyz × v) + 2*(q_xyz × (q_xyz × v))
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the inverse of q (xyzw)."""
    return quat_rotate(quat_conjugate(q), v)


# Isaac Gym uses 'quat_apply' as an alias for quat_rotate
quat_apply = quat_rotate


def get_euler_xyz(q: torch.Tensor):
    """Return (roll, pitch, yaw) tuple from xyzw quaternion batch.
    Each element has shape (...,).
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor,
                         yaw: torch.Tensor) -> torch.Tensor:
    """Build xyzw quaternion from roll/pitch/yaw (rad) tensors of shape (...)."""
    cy = torch.cos(yaw  * 0.5);  sy = torch.sin(yaw  * 0.5)
    cp = torch.cos(pitch* 0.5);  sp = torch.sin(pitch* 0.5)
    cr = torch.cos(roll * 0.5);  sr = torch.sin(roll * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return torch.stack([x, y, z, w], dim=-1)


def get_basis_vector(q: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotate a basis vector b by quaternion q (xyzw)."""
    return quat_rotate(q, b)


# ─────────────────────────────────────────────────────────────────────────────
# Convention conversion helpers (Isaac Gym xyzw ↔ Isaac Lab wxyz)
# ─────────────────────────────────────────────────────────────────────────────

def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert xyzw → wxyz (Isaac Gym → Isaac Lab/Isaac Sim convention)."""
    return torch.cat([q[..., 3:4], q[..., :3]], dim=-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Convert wxyz → xyzw (Isaac Lab/Isaac Sim → Isaac Gym convention)."""
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_heading_and_up(torso_rotation, inv_start_rot, to_target,
                            vec0, vec1, up_idx: int):
    num_envs = torso_rotation.shape[0]
    target_dirs = normalize(to_target)
    torso_quat  = quat_mul(torso_rotation, inv_start_rot)
    up_vec      = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
    heading_vec = get_basis_vector(torso_quat, vec0).view(num_envs, 3)
    up_proj     = up_vec[:, up_idx]
    heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3),
                              target_dirs.view(num_envs, 3, 1)).view(num_envs)
    return torso_quat, up_proj, heading_proj, up_vec, heading_vec


def compute_rot(torso_quat, velocity, ang_velocity, targets, torso_positions):
    vel_loc    = quat_rotate_inverse(torso_quat, velocity)
    angvel_loc = quat_rotate_inverse(torso_quat, ang_velocity)
    roll, pitch, yaw = get_euler_xyz(torso_quat)
    walk_target_angle = torch.atan2(targets[:, 2] - torso_positions[:, 2],
                                    targets[:, 0] - torso_positions[:, 0])
    angle_to_target = walk_target_angle - yaw
    return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target


@torch.jit.script
def quat_axis(q: torch.Tensor, axis: int = 0) -> torch.Tensor:
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


# ─────────────────────────────────────────────────────────────────────────────
# Expose all public names for  `from isaacgym.torch_utils import *`
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "to_torch", "torch_rand_float", "tensor_clamp", "scale", "unscale",
    "normalize",
    "quat_mul", "quat_conjugate", "quat_unit", "quat_from_angle_axis",
    "quat_rotate", "quat_rotate_inverse", "quat_apply",
    "get_euler_xyz", "quat_from_euler_xyz",
    "get_basis_vector",
    "compute_heading_and_up", "compute_rot", "quat_axis",
    "xyzw_to_wxyz", "wxyz_to_xyzw",
]
