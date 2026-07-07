"""
isaacgym.gymutil  ── Isaac Lab compatibility shim
=================================================
Provides no-op / argparse-based stubs for:
  - gymutil.parse_arguments()
  - gymutil.parse_sim_config()
  - gymutil.get_property_setter_map / get_property_getter_map
  - gymutil.get_default_setter_args / apply_random_samples
  - gymutil.check_buckets / generate_random_samples

Isaac Lab migration notes
-------------------------
parse_arguments()    → replaced by Hydra / isaaclab.envs.utils.parse_env_cfg
parse_sim_config()   → SimulationCfg is built from the Hydra config dict
randomization utils  → isaaclab.envs.mdp.randomizations.*
"""

from __future__ import annotations
import argparse
import sys
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# Isaac Lab replacement: isaaclab.utils.parse_env_cfg  +  Hydra @hydra.main
# ─────────────────────────────────────────────────────────────────────────────

def parse_arguments(description: str = "RL Policy",
                    headless: bool = False,
                    no_graphics: bool = False,
                    custom_parameters: Optional[list] = None) -> argparse.Namespace:
    """
    Minimal drop-in for gymutil.parse_arguments().

    Isaac Lab replacement:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--headless",    action="store_true", default=headless)
    parser.add_argument("--no_graphics", action="store_true", default=no_graphics)
    parser.add_argument("--device",      type=str, default="cuda:0")
    parser.add_argument("--physics_engine", type=str, default="physx")
    parser.add_argument("--num_envs",    type=int,  default=None)
    parser.add_argument("--sim_device",  type=str,  default="cuda:0")
    parser.add_argument("--compute_device_id", type=int, default=0)
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--flex",        action="store_true")
    parser.add_argument("--physx",       action="store_true")
    parser.add_argument("--pipeline",    type=str, default="gpu")
    parser.add_argument("--subscenes",   type=int, default=0)
    parser.add_argument("--slices",      type=int, default=None)

    if custom_parameters:
        for param in custom_parameters:
            name  = param.get("name", "")
            ptype = param.get("type", str)
            default = param.get("default", None)
            help_str = param.get("help", "")
            if name.startswith("--"):
                parser.add_argument(name, type=ptype, default=default, help=help_str)

    # Use parse_known_args so pytest / hydra extra args don't break things
    args, _ = parser.parse_known_args()
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Sim config parsing
# Isaac Lab replacement: SimulationCfg(**cfg["sim"])
# ─────────────────────────────────────────────────────────────────────────────

def parse_sim_config(cfg: dict, sim_params) -> None:
    """
    Apply entries from the yaml-loaded cfg["sim"] dict onto a SimParams object
    in-place.  Mirrors gymutil.parse_sim_config behaviour.

    Isaac Lab replacement:
        sim_cfg = SimulationCfg(dt=cfg["dt"], gravity=(0,0,-9.81), ...)
    """
    if not cfg:
        return

    # top-level dt / substeps / up_axis
    if "dt" in cfg:
        sim_params.dt = float(cfg["dt"])
    if "substeps" in cfg:
        sim_params.substeps = int(cfg["substeps"])
    if "up_axis" in cfg:
        val = cfg["up_axis"]
        if isinstance(val, str):
            from isaacgym.gymapi import UP_AXIS_Z, UP_AXIS_Y
            sim_params.up_axis = UP_AXIS_Z if val.lower() == "z" else UP_AXIS_Y
        else:
            sim_params.up_axis = int(val)
    if "use_gpu_pipeline" in cfg:
        sim_params.use_gpu_pipeline = bool(cfg["use_gpu_pipeline"])
    if "gravity" in cfg:
        from isaacgym.gymapi import Vec3
        g = cfg["gravity"]
        sim_params.gravity = Vec3(g[0], g[1], g[2])

    # physx sub-section
    physx_cfg = cfg.get("physx", {})
    px = sim_params.physx
    _field_map = {
        "num_threads":                    ("num_threads", int),
        "solver_type":                    ("solver_type", int),
        "use_gpu":                        ("use_gpu", bool),
        "num_position_iterations":        ("num_position_iterations", int),
        "num_velocity_iterations":        ("num_velocity_iterations", int),
        "contact_offset":                 ("contact_offset", float),
        "rest_offset":                    ("rest_offset", float),
        "bounce_threshold_velocity":      ("bounce_threshold_velocity", float),
        "max_depenetration_velocity":     ("max_depenetration_velocity", float),
        "default_buffer_size_multiplier": ("default_buffer_size_multiplier", float),
        "max_gpu_contact_pairs":          ("max_gpu_contact_pairs", int),
        "num_subscenes":                  ("num_subscenes", int),
        "contact_collection":             ("contact_collection", int),
    }
    for yaml_key, (attr, cast) in _field_map.items():
        if yaml_key in physx_cfg:
            setattr(px, attr, cast(physx_cfg[yaml_key]))


# ─────────────────────────────────────────────────────────────────────────────
# Domain-randomization utilities
# Isaac Lab replacement: isaaclab.envs.mdp.randomizations.*
# These stubs keep existing randomization calls from crashing at import time.
# ─────────────────────────────────────────────────────────────────────────────

def get_property_setter_map(gym) -> Dict[str, Any]:
    """
    Returns a map of property-name → setter callable.
    Isaac Lab replacement: event-manager randomization terms.
    """
    return {
        "dof_properties":   _noop_setter,
        "rigid_body_properties": _noop_setter,
        "rigid_shape_properties": _noop_setter,
        "actor_root_state": _noop_setter,
        "dof_state":        _noop_setter,
        "joint_state":      _noop_setter,
    }


def get_property_getter_map(gym) -> Dict[str, Any]:
    """
    Returns a map of property-name → getter callable.
    Isaac Lab replacement: event-manager randomization terms.
    """
    return {
        "dof_properties":   _noop_getter,
        "rigid_body_properties": _noop_getter,
        "rigid_shape_properties": _noop_getter,
        "actor_root_state": _noop_getter,
        "dof_state":        _noop_getter,
        "joint_state":      _noop_getter,
    }


def get_default_setter_args(gym) -> Dict[str, Any]:
    """Returns default argument dicts for each property setter."""
    return {}


def apply_random_samples(prop, og_prop, attr, lo, hi, distrib,
                          op, sched_val, sched_scaling):
    """
    Apply one randomization sample.  No-op stub.
    Isaac Lab replacement: EventTermCfg with func=randomize_*
    """
    pass


def check_buckets(gym, envs, prop_name) -> bool:
    """Validate that aggregate buckets are sized correctly.  Returns True."""
    return True


def generate_random_samples(attr, lo, hi, n, distrib):
    """Generate n random samples in [lo, hi].  Returns list of floats."""
    import random
    if distrib == "uniform":
        return [random.uniform(lo, hi) for _ in range(n)]
    elif distrib == "gaussian":
        import statistics
        mu  = (lo + hi) / 2.0
        std = (hi - lo) / 6.0
        return [max(lo, min(hi, random.gauss(mu, std))) for _ in range(n)]
    return [0.0] * n


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _noop_setter(*args, **kwargs):
    pass


def _noop_getter(*args, **kwargs):
    return None
