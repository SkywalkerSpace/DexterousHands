"""
isaacgym.gymtorch  ── Isaac Lab compatibility shim
===================================================
In Isaac Gym, gymtorch.wrap_tensor() converted a raw PhysX GPU pointer into a
PyTorch tensor, and unwrap_tensor() reversed the conversion so the gym C-API
could consume it.

In Isaac Lab the physics state is already exposed as plain PyTorch tensors, so
both operations are pure identity functions.  Source code that calls
    self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    ...
    gymtorch.unwrap_tensor(self.dof_state)
will continue to compile and run without modification.
"""

import torch


def wrap_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Isaac Gym → Isaac Lab migration:
      Old: gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
      New: articulation.data.joint_pos / articulation.data.joint_vel
           (the acquire + wrap pattern is replaced by direct attribute access)

    This shim returns the tensor unchanged so existing code paths keep working
    when you supply a pre-built tensor from Isaac Lab state dicts.
    """
    return tensor


def unwrap_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Isaac Gym → Isaac Lab migration:
      Old: gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(buf), ...)
      New: articulation.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
           (indexed set calls are replaced by write_*_to_sim helpers)

    This shim returns the tensor unchanged.
    """
    return tensor
