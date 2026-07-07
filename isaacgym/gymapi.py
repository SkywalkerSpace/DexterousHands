"""
isaacgym.gymapi  ── Isaac Lab compatibility shim
================================================
Provides pure-Python replacements for every data type and constant that source
code accesses via  `from isaacgym import gymapi`.

Nothing here requires the Isaac Gym C-extension.  The classes are plain Python
dataclasses / simple objects, intentionally matching the attribute layout that
Isaac Gym code expects.

Isaac Lab migration notes are embedded as inline comments where the concept has
a direct counterpart.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Simulation backend constants
# ─────────────────────────────────────────────────────────────────────────────

SIM_PHYSX = 0          # Isaac Lab only supports PhysX; Flex is removed
SIM_FLEX  = 1          # Unsupported in Isaac Lab

# ─────────────────────────────────────────────────────────────────────────────
# DOF drive mode constants
# Isaac Lab equivalent: isaaclab.actuators.* (ImplicitActuator, DCMotor …)
# ─────────────────────────────────────────────────────────────────────────────

DOF_MODE_NONE   = 0    # No drive
DOF_MODE_EFFORT = 1    # Torque / force control
DOF_MODE_VEL    = 2    # Velocity control
DOF_MODE_POS    = 3    # Position control  (most common)

# ─────────────────────────────────────────────────────────────────────────────
# Mesh / visual / collision constants
# ─────────────────────────────────────────────────────────────────────────────

MESH_VISUAL           = 0
MESH_COLLISION        = 1
MESH_NONE             = 2

# Mesh normal computation mode
COMPUTE_PER_VERTEX    = 0
COMPUTE_PER_FACE      = 1
COMPUTE_SMOOTH        = 2
COMPUTE_NONE          = 3

# ─────────────────────────────────────────────────────────────────────────────
# Domain / index scope constants
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_SIM   = 0       # global (simulation-wide) index
DOMAIN_ENV   = 1       # per-environment index
DOMAIN_ACTOR = 2       # per-actor index

# ─────────────────────────────────────────────────────────────────────────────
# Force / torque application space
# Isaac Lab equivalent: articulation.set_external_force_and_torque(is_global=...)
# ─────────────────────────────────────────────────────────────────────────────

ENV_SPACE    = 0       # environment (local) frame
LOCAL_SPACE  = 1       # body-local frame
GLOBAL_SPACE = 2       # world frame

# ─────────────────────────────────────────────────────────────────────────────
# Camera image types
# Isaac Lab equivalent: omni.isaac.sensor.Camera output keys
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_COLOR          = 0   # RGBA  uint8
IMAGE_DEPTH          = 1   # depth float32
IMAGE_SEGMENTATION   = 2
IMAGE_OPTICAL_FLOW   = 3

# ─────────────────────────────────────────────────────────────────────────────
# Up-axis constants
# ─────────────────────────────────────────────────────────────────────────────

class UpAxis:
    UP_AXIS_Y = 1
    UP_AXIS_Z = 2

UP_AXIS_Y = UpAxis.UP_AXIS_Y
UP_AXIS_Z = UpAxis.UP_AXIS_Z


# ─────────────────────────────────────────────────────────────────────────────
# Vec3
# Isaac Lab equivalent: just use torch tensors / np arrays
# ─────────────────────────────────────────────────────────────────────────────

class Vec3:
    """3-D vector matching gymapi.Vec3 attribute layout (x, y, z)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __repr__(self) -> str:
        return f"Vec3({self.x}, {self.y}, {self.z})"


# ─────────────────────────────────────────────────────────────────────────────
# Quat  (x, y, z, w)  – Isaac Gym xyzw convention
# Isaac Lab uses wxyz; convert with gymapi_quat_to_lab() below
# ─────────────────────────────────────────────────────────────────────────────

class Quat:
    """Quaternion in xyzw convention (matching Isaac Gym)."""

    __slots__ = ("x", "y", "z", "w")

    def __init__(self, x: float = 0.0, y: float = 0.0,
                 z: float = 0.0, w: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def from_euler_zyx(self, z: float, y: float, x: float) -> "Quat":
        """Set from ZYX intrinsic Euler angles (radians) and return self."""
        cz, sz = math.cos(z / 2), math.sin(z / 2)
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        cx, sx = math.cos(x / 2), math.sin(x / 2)

        self.w = cx * cy * cz + sx * sy * sz
        self.x = sx * cy * cz - cx * sy * sz
        self.y = cx * sy * cz + sx * cy * sz
        self.z = cx * cy * sz - sx * sy * cz
        return self

    def __repr__(self) -> str:
        return f"Quat(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, w={self.w:.4f})"

    def to_list_xyzw(self):
        return [self.x, self.y, self.z, self.w]

    def to_list_wxyz(self):
        """Isaac Lab / Isaac Sim convention."""
        return [self.w, self.x, self.y, self.z]


# ─────────────────────────────────────────────────────────────────────────────
# Transform
# Isaac Lab equivalent: isaaclab.utils.math.combine_frame_transforms / pose tensors
# ─────────────────────────────────────────────────────────────────────────────

class Transform:
    """Rigid-body pose: position (Vec3) + orientation (Quat)."""

    def __init__(self):
        self.p = Vec3()
        self.r = Quat()

    def __repr__(self) -> str:
        return f"Transform(p={self.p}, r={self.r})"

    def to_list(self) -> list:
        """[px,py,pz, qx,qy,qz,qw] – format used in root_state_tensor."""
        return [self.p.x, self.p.y, self.p.z,
                self.r.x, self.r.y, self.r.z, self.r.w]


# ─────────────────────────────────────────────────────────────────────────────
# PlaneParams
# Isaac Lab equivalent: isaaclab.scene.GroundPlaneCfg
# ─────────────────────────────────────────────────────────────────────────────

class PlaneParams:
    def __init__(self):
        self.normal           = Vec3(0.0, 0.0, 1.0)
        self.distance         = 0.0
        self.static_friction  = 1.0
        self.dynamic_friction = 1.0
        self.restitution      = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# VhacdParams
# Isaac Lab equivalent: V-HACD params are set via USD collision approximation
# ─────────────────────────────────────────────────────────────────────────────

class VhacdParams:
    def __init__(self):
        self.resolution         = 300_000
        self.max_convex_hulls   = 1024
        self.max_num_vertices   = 64
        self.concavity          = 0.001
        self.alpha              = 0.05
        self.beta               = 0.05
        self.convex_hull_approximation = True
        self.mode               = 0     # 0 = voxel, 1 = tetrahedra
        self.plane_downsampling = 4
        self.hull_downsampling  = 4
        self.pca                = 0
        self.min_volume_per_convex_hull = 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# AssetOptions
# Isaac Lab equivalent: ArticulationCfg / RigidObjectCfg spawn options;
#   most flags translate to USD prim properties set by the spawner.
# ─────────────────────────────────────────────────────────────────────────────

class AssetOptions:
    """
    Matches every attribute set by bidexhands / dexteroushandenvs tasks.

    Isaac Lab migration map
    -----------------------
    fix_base_link          → ArticulationCfg(fix_base=True)
    disable_gravity        → RigidBodyPropertiesCfg(disable_gravity=True)
    flip_visual_attachments→ USD mesh transform at import time
    collapse_fixed_joints  → ArticulationRootPropertiesCfg(fix_base=True) +
                             merge_fixed_joints in the URDF→USD converter
    angular_damping /
    linear_damping         → RigidBodyPropertiesCfg(angular_damping=...,
                                                     linear_damping=...)
    use_physx_armature     → ArticulationRootPropertiesCfg(armature=...)
    default_dof_drive_mode → set per-joint in ActuatorCfg
    density                → MassPropertiesCfg(density=...)
    vhacd_enabled/params   → CollisionPropertiesCfg / MeshCollisionPropertiesCfg
    """
    def __init__(self):
        self.flip_visual_attachments  = False
        self.fix_base_link            = False
        self.collapse_fixed_joints    = False
        self.disable_gravity          = False
        self.thickness                = 0.0
        self.angular_damping          = 0.0
        self.linear_damping           = 0.0
        self.use_physx_armature       = False
        self.default_dof_drive_mode   = DOF_MODE_NONE
        self.density                  = 1000.0
        self.use_mesh_materials       = False
        self.mesh_normal_mode         = COMPUTE_PER_VERTEX
        self.override_com             = False
        self.override_inertia         = False
        self.vhacd_enabled            = False
        self.vhacd_params             = VhacdParams()
        self.armature                 = 0.0
        self.max_angular_velocity     = 1000.0
        self.max_linear_velocity      = 1000.0
        self.convex_decomposition_from_submeshes = False
        self.tendon_limit_stiffness   = 0.0
        self.min_particle_mass        = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DofProperties  (structured array proxy)
# Isaac Lab equivalent: per-joint limits in ArticulationCfg
# ─────────────────────────────────────────────────────────────────────────────

class DofProperties:
    """
    Proxy for the structured numpy-array returned by
    gym.get_asset_dof_properties().  Supports dict-style field access:
        props['lower'][i], props['upper'][i], props['stiffness'][i] …
    """
    FIELDS = ("lower", "upper", "stiffness", "damping", "friction",
              "velocity", "effort", "armature", "driveMode", "hasLimits")

    def __init__(self, n: int):
        self._data = {f: [0.0] * n for f in self.FIELDS}
        self._data["hasLimits"] = [False] * n
        self._data["driveMode"] = [DOF_MODE_NONE] * n

    def __getitem__(self, key: str):
        return self._data[key]

    def __setitem__(self, key: str, value):
        self._data[key] = value

    def __len__(self):
        return len(self._data["lower"])


# ─────────────────────────────────────────────────────────────────────────────
# RigidShapeProperties
# Isaac Lab equivalent: RigidBodyPropertiesCfg / CollisionPropertiesCfg
# ─────────────────────────────────────────────────────────────────────────────

class RigidShapeProperties:
    def __init__(self):
        self.friction     = 1.0
        self.restitution  = 0.0
        self.rolling_friction  = 0.0
        self.torsion_friction  = 0.0
        self.compliance   = 0.0
        self.thickness    = 0.0
        self.filter_data  = None


# ─────────────────────────────────────────────────────────────────────────────
# CameraProperties
# Isaac Lab equivalent: CameraCfg (omni.isaac.sensor)
# ─────────────────────────────────────────────────────────────────────────────

class CameraProperties:
    """
    Isaac Lab equivalent: isaaclab.sensors.CameraCfg
      width, height  → CameraCfg.width / .height
      horizontal_fov → CameraCfg.focal_length (computed from fov + sensor size)
      enable_tensors → always True in Isaac Lab GPU pipeline
    """
    def __init__(self):
        self.width                   = 256
        self.height                  = 256
        self.enable_tensors          = True
        self.horizontal_fov          = 90.0
        self.near_plane              = 0.1
        self.far_plane               = 100.0
        self.supersampling_horizontal = 1
        self.supersampling_vertical  = 1
        self.use_collision_geometry  = False


# ─────────────────────────────────────────────────────────────────────────────
# PhysxParams / SimParams
# Isaac Lab equivalent: SimulationCfg in isaaclab.sim
# ─────────────────────────────────────────────────────────────────────────────

class PhysxParams:
    """
    Isaac Lab equivalent: SimulationCfg.physx
      solver_type          → 0=PGS, 1=TGS  (Isaac Lab: solver_type)
      num_position_iters   → SimulationCfg.physx.solver_position_iteration_count
      num_velocity_iters   → SimulationCfg.physx.solver_velocity_iteration_count
      use_gpu              → SimulationCfg.use_gpu_pipeline
    """
    def __init__(self):
        self.num_threads                    = 4
        self.solver_type                    = 1    # 0=PGS, 1=TGS
        self.use_gpu                        = True
        self.num_position_iterations        = 8
        self.num_velocity_iterations        = 0
        self.contact_offset                 = 0.002
        self.rest_offset                    = 0.0
        self.bounce_threshold_velocity      = 0.2
        self.max_depenetration_velocity     = 1000.0
        self.default_buffer_size_multiplier = 5.0
        self.max_gpu_contact_pairs          = 1 << 23
        self.num_subscenes                  = 0
        self.contact_collection             = 0


class SimParams:
    """
    Isaac Lab equivalent: SimulationCfg (isaaclab.sim.SimulationCfg)
      dt                → SimulationCfg.dt
      substeps          → SimulationCfg.render_interval (approx)
      gravity           → SimulationCfg.gravity
      use_gpu_pipeline  → SimulationCfg.use_gpu_pipeline
      up_axis           → always Z in Isaac Lab
    """
    def __init__(self):
        self.dt               = 1.0 / 60.0
        self.substeps         = 2
        self.up_axis          = UP_AXIS_Z
        self.use_gpu_pipeline = True
        self.gravity          = Vec3(0.0, 0.0, -9.81)
        self.physx            = PhysxParams()


# ─────────────────────────────────────────────────────────────────────────────
# TendonProperties  (used by shadow hand)
# Isaac Lab: tendons not natively supported; approximate with constraints/gains
# ─────────────────────────────────────────────────────────────────────────────

class TendonProperties:
    def __init__(self):
        self.stiffness       = 0.0
        self.damping         = 0.0
        self.limit_stiffness = 0.0
        self.lower           = -1e6
        self.upper           =  1e6
        self.rest_length     = 0.0
        self.offset          = 0.0
        self.free_motion     = True


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight handle types  (opaque ids, integer-backed)
# ─────────────────────────────────────────────────────────────────────────────

class _Handle(int):
    """Base opaque handle (just an integer with a class label)."""
    pass

class AssetHandle(_Handle):   pass
class EnvHandle(_Handle):     pass
class ActorHandle(_Handle):   pass
class AttractorHandle(_Handle): pass
