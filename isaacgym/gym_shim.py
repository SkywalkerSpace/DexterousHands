"""
isaacgym.gym_shim  ── Isaac Lab compatibility shim  (self.gym 替代层)
====================================================================
GymShim 是整个兼容层最核心的类。它替代 Isaac Gym 中的 gym 句柄对象，
把源代码里的 self.gym.xxx() 调用路由到 Isaac Lab 等效 API。

设计原则
--------
1. acquire_* / refresh_*  →  全部 no-op，Isaac Lab 状态张量实时可用
2. wrap_tensor / unwrap_tensor  →  identity（见 gymtorch.py）
3. set_*_indexed  →  路由到 Isaac Lab Articulation.write_*_to_sim()
4. create_*/load_asset  →  存储配置，在 Isaac Lab scene.InteractiveScene 层完成实例化
5. apply_rigid_body_force_tensors  →  RigidObject.set_external_force_and_torque()

使用方式（在 BaseTask 子类的 __init__ 里）
------------------------------------------
    from isaacgym.gym_shim import GymShim

    self.gym = GymShim()
    # 注册 Isaac Lab 已实例化的仿真对象（在 _setup_scene 后调用）
    self.gym.register(
        robot=self.robot,          # Articulation
        obj=self.object,           # RigidObject
        goal=self.goal,            # RigidObject（可选）
        sim=self.sim,              # SimulationContext
    )
    # acquire 照常调用，返回预先拼好的视图张量
    dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
    self.dof_state   = gymtorch.wrap_tensor(dof_state_tensor)   # identity
"""

from __future__ import annotations
import torch
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────────────────────────────────────

class _SimDummy:
    """占位 sim 句柄；Isaac Lab 不需要显式 sim 对象。"""
    pass


class _AssetRecord:
    __slots__ = ("root", "file", "options", "handle")
    def __init__(self, root, file, options, handle):
        self.root    = root
        self.file    = file
        self.options = options
        self.handle  = handle          # 整数 id


class _EnvRecord:
    __slots__ = ("handle", "actors")
    def __init__(self, handle):
        self.handle = handle
        self.actors: list = []


# ─────────────────────────────────────────────────────────────────────────────
# GymShim
# ─────────────────────────────────────────────────────────────────────────────

class GymShim:
    """
    Isaac Gym gym 句柄的兼容层。

    工作流程
    --------
    Step 1  创建：self.gym = GymShim()
    Step 2  注册 Isaac Lab 仿真对象（场景初始化后）：self.gym.register(...)
    Step 3  源码里的 acquire / refresh / set_indexed 等照常调用
    """

    def __init__(self):
        # Isaac Lab 仿真对象引用（注册后填充）
        self._robot:      Optional[Any] = None   # Articulation
        self._objects:    Dict[str, Any] = {}    # name → RigidObject / Articulation
        self._sim_ctx:    Optional[Any] = None   # SimulationContext

        # 内部资产 / 环境注册表
        self._assets:  List[_AssetRecord] = []
        self._envs:    List[_EnvRecord]   = []

        # 状态张量缓存（acquire 后由 _rebuild_* 填充）
        self._root_state_tensor:  Optional[torch.Tensor] = None
        self._dof_state_tensor:   Optional[torch.Tensor] = None
        self._rb_state_tensor:    Optional[torch.Tensor] = None
        self._dof_force_tensor:   Optional[torch.Tensor] = None
        self._contact_force_tensor: Optional[torch.Tensor] = None

    # ─────────────────────────────────────────────────────────────────────────
    # 注册 Isaac Lab 对象
    # ─────────────────────────────────────────────────────────────────────────

    def register(self, sim=None, robot=None, **objects):
        """
        注册 Isaac Lab 仿真对象，使 GymShim 能构建兼容张量视图。

        参数
        ----
        sim    : SimulationContext（可省略，仅作引用保存）
        robot  : 主 Articulation（对应原 allegrohand / shadow_hand 等）
        **objects : 其他 RigidObject / Articulation，键名任意
                    e.g.  obj=self.object, goal=self.goal

        调用时机：scene._setup_scene() 完成之后，第一次 acquire 之前。
        """
        if sim is not None:
            self._sim_ctx = sim
        if robot is not None:
            self._robot = robot
        self._objects.update(objects)

    # ─────────────────────────────────────────────────────────────────────────
    # sim 创建 / 场景初始化（Isaac Lab 框架接管，这里返回占位对象）
    # ─────────────────────────────────────────────────────────────────────────

    def create_sim(self, compute_device: int, graphics_device: int,
                   physics_engine, params) -> _SimDummy:
        """
        Isaac Lab 替代：
            sim_cfg = SimulationCfg(dt=..., use_gpu_pipeline=True)
            self.sim = SimulationContext(sim_cfg)
        框架会在 ManagerBasedEnv.__init__ 中自动完成，无需手动调用。
        """
        return _SimDummy()

    def destroy_sim(self, sim):
        pass

    def add_ground_plane(self, sim, plane_params):
        """
        Isaac Lab 替代：
            from isaaclab.scene import GroundPlaneCfg
            cfg.scene.ground = GroundPlaneCfg(...)
        在 InteractiveSceneCfg 里声明，不需要手动调用。
        """
        pass

    # 旧式接口别名
    def add_ground(self, sim, plane_params):
        return self.add_ground_plane(sim, plane_params)

    # ─────────────────────────────────────────────────────────────────────────
    # 环境 / Actor 创建（Isaac Lab 框架接管）
    # ─────────────────────────────────────────────────────────────────────────

    def create_env(self, sim, lower, upper, num_per_row) -> int:
        """
        Isaac Lab 替代：
            scene = InteractiveScene(cfg)
            scene.clone_environments(copy_from_source=False)
        返回环境 index 作为 handle。
        """
        handle = len(self._envs)
        self._envs.append(_EnvRecord(handle))
        return handle

    def load_asset(self, sim, asset_root: str, asset_file: str,
                   options=None) -> int:
        """
        Isaac Lab 替代：
            from isaaclab.assets import ArticulationCfg
            cfg = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/robot",
                spawn=UsdFileCfg(usd_path=f"{asset_root}/{asset_file}"),
                ...
            )
        URDF → USD 使用 isaaclab/lab_tasks/... 中的转换器离线完成。
        返回整数 asset handle。
        """
        handle = len(self._assets)
        self._assets.append(_AssetRecord(asset_root, asset_file, options, handle))
        return handle

    def create_actor(self, env_handle: int, asset_handle: int, pose,
                     name: str, env_idx: int, collision_filter: int = 1,
                     segmentation_id: int = 0) -> int:
        """
        Isaac Lab 替代：
            # Actor 由 Articulation / RigidObject 在 _setup_scene 中实例化
            self.robot = Articulation(cfg=robot_cfg)
        返回 actor index（env 内部顺序）。
        """
        if 0 <= env_handle < len(self._envs):
            actor_idx = len(self._envs[env_handle].actors)
            self._envs[env_handle].actors.append(
                {"asset": asset_handle, "name": name, "env": env_idx}
            )
            return actor_idx
        return 0

    def set_light_parameters(self, *args, **kwargs):
        pass

    def set_camera_location(self, *args, **kwargs):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # acquire_*  →  返回状态张量（Isaac Lab 直接从 .data.* 属性读取）
    # ─────────────────────────────────────────────────────────────────────────
    #
    # 接口保持不变，内部从 Isaac Lab Articulation / RigidObject 拼装张量。
    # 调用 register() 后才能得到真实数据；未注册时返回空张量，不崩溃。
    # ─────────────────────────────────────────────────────────────────────────

    def acquire_actor_root_state_tensor(self, sim) -> torch.Tensor:
        """
        Isaac Lab 替代：
            # 单个 Articulation
            root_state = self.robot.data.root_state_w          # (N, 13)
            # 多个 actor 拼接（与原始 flat [N_actors*N_envs, 13] 一致）
            root_state = torch.cat([
                self.robot.data.root_state_w,
                self.object.data.root_state_w,
            ], dim=0)
        """
        self._root_state_tensor = self._build_root_state()
        return self._root_state_tensor

    def acquire_dof_state_tensor(self, sim) -> torch.Tensor:
        """
        Isaac Lab 替代：
            # shape: (N_envs, N_dof, 2)  → .view(N_envs, -1, 2)
            joint_pos = self.robot.data.joint_pos   # (N_envs, N_dof)
            joint_vel = self.robot.data.joint_vel   # (N_envs, N_dof)
            dof_state = torch.stack([joint_pos, joint_vel], dim=-1)
        """
        self._dof_state_tensor = self._build_dof_state()
        return self._dof_state_tensor

    def acquire_rigid_body_state_tensor(self, sim) -> torch.Tensor:
        """
        Isaac Lab 替代：
            body_state = self.robot.data.body_state_w  # (N_envs, N_bodies, 13)
        flatten 成 (N_envs * N_bodies, 13) 以兼容原 .view(num_envs, -1, 13) 模式。
        """
        self._rb_state_tensor = self._build_rigid_body_state()
        return self._rb_state_tensor

    def acquire_dof_force_tensor(self, sim) -> torch.Tensor:
        """
        Isaac Lab 替代：
            applied_torque = self.robot.data.applied_torque  # (N_envs, N_dof)
        """
        self._dof_force_tensor = self._build_dof_force()
        return self._dof_force_tensor

    def acquire_force_sensor_tensor(self, sim) -> torch.Tensor:
        """
        Isaac Lab 替代：
            from isaaclab.sensors import ContactSensor
            net_force = self.contact_sensor.data.net_forces_w  # (N_envs, N_sensors, 3)
        """
        return torch.zeros(max(len(self._envs), 1), 6)

    def acquire_net_contact_force_tensor(self, sim) -> torch.Tensor:
        """
        Isaac Lab 替代：
            contact_force = self.contact_sensor.data.net_forces_w
        """
        self._contact_force_tensor = torch.zeros(max(len(self._envs), 1), 3)
        return self._contact_force_tensor

    # ─────────────────────────────────────────────────────────────────────────
    # refresh_*  →  全部 no-op（Isaac Lab 每 step 自动刷新）
    # ─────────────────────────────────────────────────────────────────────────

    def refresh_actor_root_state_tensor(self, sim):   pass
    def refresh_dof_state_tensor(self, sim):          pass
    def refresh_rigid_body_state_tensor(self, sim):   pass
    def refresh_force_sensor_tensor(self, sim):       pass
    def refresh_dof_force_tensor(self, sim):          pass
    def refresh_net_contact_force_tensor(self, sim):  pass
    def refresh_jacobian_tensors(self, sim):          pass
    def refresh_mass_matrix_tensors(self, sim):       pass

    # ─────────────────────────────────────────────────────────────────────────
    # set_*_indexed  →  路由到 Isaac Lab write_*_to_sim()
    # ─────────────────────────────────────────────────────────────────────────

    def set_actor_root_state_tensor_indexed(self, sim,
                                             root_tensor: torch.Tensor,
                                             index_tensor: torch.Tensor,
                                             count: int):
        """
        Isaac Lab 替代：
            env_ids = index_tensor[:count]
            # Articulation
            self.robot.write_root_state_to_sim(root_tensor[env_ids], env_ids=env_ids)
            # RigidObject
            self.object.write_root_state_to_sim(root_tensor[env_ids], env_ids=env_ids)

        注意：Isaac Lab 的 root_state 格式 = [pos(3), quat_wxyz(4), vel(3), ang_vel(3)]
              Isaac Gym 格式               = [pos(3), quat_xyzw(4), vel(3), ang_vel(3)]
        四元数分量顺序需在调用前转换：
            state_lab = state_gym.clone()
            state_lab[:, 3:7] = state_gym[:, [6,3,4,5]]  # xyzw → wxyz
        """
        if self._robot is None:
            return
        env_ids = index_tensor[:count].long()
        state   = root_tensor.view(-1, 13)
        # xyzw → wxyz 转换（Isaac Gym → Isaac Lab quat 约定）
        state_lab = state.clone()
        state_lab[:, 3:7] = state[:, [6, 3, 4, 5]]
        try:
            self._robot.write_root_state_to_sim(state_lab[env_ids], env_ids=env_ids)
            for obj in self._objects.values():
                if hasattr(obj, "write_root_state_to_sim"):
                    obj.write_root_state_to_sim(state_lab[env_ids], env_ids=env_ids)
        except Exception as e:
            import warnings
            warnings.warn(f"[GymShim] set_actor_root_state_tensor_indexed: {e}")

    def set_dof_state_tensor_indexed(self, sim,
                                      dof_tensor: torch.Tensor,
                                      index_tensor: torch.Tensor,
                                      count: int):
        """
        Isaac Lab 替代：
            env_ids  = index_tensor[:count]
            joint_pos = dof_tensor.view(N_envs, N_dof, 2)[env_ids, :, 0]
            joint_vel = dof_tensor.view(N_envs, N_dof, 2)[env_ids, :, 1]
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel,
                                                 joint_ids=None, env_ids=env_ids)
        """
        if self._robot is None:
            return
        env_ids   = index_tensor[:count].long()
        n_dof     = self._robot.num_joints if hasattr(self._robot, "num_joints") else \
                    dof_tensor.shape[0] // max(len(self._envs), 1)
        dof_view  = dof_tensor.view(-1, n_dof, 2)
        joint_pos = dof_view[env_ids, :, 0]
        joint_vel = dof_view[env_ids, :, 1]
        try:
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel,
                                                   joint_ids=None, env_ids=env_ids)
        except Exception as e:
            import warnings
            warnings.warn(f"[GymShim] set_dof_state_tensor_indexed: {e}")

    def set_dof_position_target_tensor_indexed(self, sim,
                                                target_tensor: torch.Tensor,
                                                index_tensor: torch.Tensor,
                                                count: int):
        """
        Isaac Lab 替代：
            env_ids = index_tensor[:count]
            self.robot.set_joint_position_target(targets[env_ids],
                                                  joint_ids=None, env_ids=env_ids)
        """
        if self._robot is None:
            return
        env_ids = index_tensor[:count].long()
        try:
            self._robot.set_joint_position_target(
                target_tensor[env_ids], joint_ids=None, env_ids=env_ids)
        except Exception as e:
            import warnings
            warnings.warn(f"[GymShim] set_dof_position_target_tensor_indexed: {e}")

    def set_dof_velocity_target_tensor_indexed(self, sim,
                                                target_tensor: torch.Tensor,
                                                index_tensor: torch.Tensor,
                                                count: int):
        """
        Isaac Lab 替代：
            self.robot.set_joint_velocity_target(targets[env_ids], env_ids=env_ids)
        """
        if self._robot is None:
            return
        env_ids = index_tensor[:count].long()
        try:
            self._robot.set_joint_velocity_target(
                target_tensor[env_ids], joint_ids=None, env_ids=env_ids)
        except Exception as e:
            import warnings
            warnings.warn(f"[GymShim] set_dof_velocity_target_tensor_indexed: {e}")

    def apply_rigid_body_force_tensors(self, sim,
                                        force_tensor: torch.Tensor,
                                        torque_tensor: Optional[torch.Tensor],
                                        space: int = 0):
        """
        Isaac Lab 替代：
            from isaacgym.gymapi import ENV_SPACE, GLOBAL_SPACE
            is_global = (space == GLOBAL_SPACE)
            # 外力要在 pre_physics_step 里调用 write_*，Isaac Lab 以 body index 为单位
            self.robot.set_external_force_and_torque(
                forces  = force_tensor.view(N_envs, N_bodies, 3),
                torques = torque_tensor.view(N_envs, N_bodies, 3),
                body_ids = None,
                env_ids  = None,
            )
        """
        if self._robot is None:
            return
        n_envs = max(len(self._envs), 1)
        try:
            n_bodies = force_tensor.shape[0] // n_envs
            forces   = force_tensor.view(n_envs, n_bodies, 3)
            torques  = (torque_tensor.view(n_envs, n_bodies, 3)
                        if torque_tensor is not None
                        else torch.zeros_like(forces))
            self._robot.set_external_force_and_torque(forces, torques)
        except Exception as e:
            import warnings
            warnings.warn(f"[GymShim] apply_rigid_body_force_tensors: {e}")

    # 非索引版本（整体更新）
    def set_actor_root_state_tensor(self, sim, root_tensor: torch.Tensor):
        self.set_actor_root_state_tensor_indexed(
            sim, root_tensor,
            torch.arange(len(self._envs), device=root_tensor.device),
            len(self._envs))

    def set_dof_state_tensor(self, sim, dof_tensor: torch.Tensor):
        self.set_dof_state_tensor_indexed(
            sim, dof_tensor,
            torch.arange(len(self._envs), device=dof_tensor.device),
            len(self._envs))

    # ─────────────────────────────────────────────────────────────────────────
    # asset / actor 属性查询
    # ─────────────────────────────────────────────────────────────────────────

    def get_asset_dof_count(self, asset_handle: int) -> int:
        """
        Isaac Lab 替代：
            self.robot.num_joints
        """
        if self._robot is not None and hasattr(self._robot, "num_joints"):
            return self._robot.num_joints
        return 0

    def get_asset_rigid_body_count(self, asset_handle: int) -> int:
        """
        Isaac Lab 替代：
            self.robot.num_bodies
        """
        if self._robot is not None and hasattr(self._robot, "num_bodies"):
            return self._robot.num_bodies
        return 0

    def get_asset_dof_properties(self, asset_handle: int):
        """
        Isaac Lab 替代：
            limits = self.robot.data.joint_limits   # (N_dof, 2)
            stiffness = ...   # from ActuatorCfg
        """
        from isaacgym.gymapi import DofProperties
        n = self.get_asset_dof_count(asset_handle)
        props = DofProperties(n)
        if self._robot is not None:
            try:
                limits = self._robot.data.joint_limits   # (N_dof, 2)
                for i in range(n):
                    props["lower"][i]    = limits[i, 0].item()
                    props["upper"][i]    = limits[i, 1].item()
                    props["hasLimits"][i] = True
            except Exception:
                pass
        return props

    def get_asset_rigid_body_names(self, asset_handle: int) -> List[str]:
        """Isaac Lab 替代：self.robot.body_names"""
        if self._robot is not None and hasattr(self._robot, "body_names"):
            return list(self._robot.body_names)
        return []

    def get_asset_dof_names(self, asset_handle: int) -> List[str]:
        """Isaac Lab 替代：self.robot.joint_names"""
        if self._robot is not None and hasattr(self._robot, "joint_names"):
            return list(self._robot.joint_names)
        return []

    def find_asset_rigid_body_index(self, asset_handle: int,
                                     name: str) -> int:
        """Isaac Lab 替代：self.robot.find_bodies([name])[0]"""
        if self._robot is not None:
            try:
                idx = self._robot.find_bodies([name])
                return idx[0] if idx else -1
            except Exception:
                pass
        return -1

    def find_asset_dof_index(self, asset_handle: int, name: str) -> int:
        """Isaac Lab 替代：self.robot.find_joints([name])[0]"""
        if self._robot is not None:
            try:
                idx = self._robot.find_joints([name])
                return idx[0] if idx else -1
            except Exception:
                pass
        return -1

    def get_actor_dof_properties(self, env_handle, actor_handle):
        return self.get_asset_dof_properties(actor_handle)

    def set_actor_dof_properties(self, env_handle, actor_handle, props):
        """
        Isaac Lab 替代：在 ActuatorCfg 里静态声明；运行时修改需 write_joint_*_to_sim。
        此处为 no-op stub。
        """
        pass

    def get_actor_rigid_body_names(self, env_handle, actor_handle):
        return self.get_asset_rigid_body_names(actor_handle)

    def get_actor_dof_names(self, env_handle, actor_handle):
        return self.get_asset_dof_names(actor_handle)

    def get_actor_rigid_body_index(self, env_handle, actor_handle, name: str):
        return self.find_asset_rigid_body_index(actor_handle, name)

    def get_asset_tendon_count(self, asset_handle: int) -> int:
        """Isaac Lab 目前不原生支持肌腱；返回 0。"""
        return 0

    def get_actor_tendon_properties(self, env_handle, actor_handle):
        return []

    def set_actor_tendon_properties(self, env_handle, actor_handle, props):
        pass

    def get_asset_rigid_shape_count(self, asset_handle: int) -> int:
        return 0

    def set_actor_rigid_shape_properties(self, env_handle, actor_handle, props):
        pass

    def get_actor_rigid_shape_properties(self, env_handle, actor_handle):
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Camera（简化 stub）
    # Isaac Lab 替代：isaaclab.sensors.CameraCfg + TiledCamera
    # ─────────────────────────────────────────────────────────────────────────

    def create_camera_sensor(self, env_handle, props):
        return 0

    def set_camera_transform(self, cam_handle, env_handle, transform):
        pass

    def get_camera_image(self, sim, env_handle, cam_handle, image_type):
        return None

    def start_access_image_tensors(self, sim):
        pass

    def end_access_image_tensors(self, sim):
        pass

    def get_camera_image_gpu_tensor(self, sim, env_handle,
                                     cam_handle, image_type):
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Viewer / 渲染（headless 场景下全部 no-op）
    # ─────────────────────────────────────────────────────────────────────────

    def create_viewer(self, sim, props):
        return None

    def destroy_viewer(self, viewer):
        pass

    def viewer_is_open(self, viewer) -> bool:
        return False

    def step_graphics(self, sim):
        pass

    def draw_lines(self, *args, **kwargs):
        pass

    def clear_lines(self, viewer):
        pass

    def subscribe_viewer_keyboard_event(self, *args, **kwargs):
        pass

    def query_viewer_has_closed(self, viewer) -> bool:
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Attractor（模拟末端执行器目标）
    # Isaac Lab 替代：DifferentialIKController / OperationalSpaceController
    # ─────────────────────────────────────────────────────────────────────────

    def create_rigid_body_attractor(self, env_handle, props):
        return 0

    def set_attractor_properties(self, env_handle, attractor_handle, props):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # 仿真步进（在 BaseTask 中通常由框架调用）
    # ─────────────────────────────────────────────────────────────────────────

    def simulate(self, sim):
        """
        Isaac Lab 替代（在 ManagerBasedEnv / DirectRLEnv 里自动调用）：
            self.sim.step()
        """
        if self._sim_ctx is not None and hasattr(self._sim_ctx, "step"):
            self._sim_ctx.step()

    def fetch_results(self, sim, wait_for_gpu: bool = True):
        pass

    def sync_frame_time(self, sim):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # 内部：从 Isaac Lab 对象构建兼容张量
    # ─────────────────────────────────────────────────────────────────────────

    def _build_root_state(self) -> torch.Tensor:
        """
        构建 (N_actors * N_envs, 13) root state 张量。
        格式：[pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)]
        Isaac Lab 使用 wxyz，这里转换成 xyzw 以保持源码兼容。
        """
        parts = []
        all_actors = ([self._robot] if self._robot is not None else []) + \
                     list(self._objects.values())
        for actor in all_actors:
            if actor is None:
                continue
            try:
                s = actor.data.root_state_w.clone()   # (N_envs, 13)
                # wxyz → xyzw  （Isaac Lab → Isaac Gym quat 约定）
                s[:, 3:7] = s[:, [4, 5, 6, 3]]
                parts.append(s)
            except Exception:
                pass
        if parts:
            return torch.cat(parts, dim=0)
        n = max(len(self._envs), 1)
        return torch.zeros(n, 13)

    def _build_dof_state(self) -> torch.Tensor:
        """
        构建 (N_envs * N_dof, 2) dof state 张量 [pos, vel]。
        源码通常做：dof_state.view(num_envs, -1, 2)[:, :n_dof]
        """
        if self._robot is not None:
            try:
                pos = self._robot.data.joint_pos   # (N_envs, N_dof)
                vel = self._robot.data.joint_vel
                return torch.stack([pos, vel], dim=-1).view(-1, 2)
            except Exception:
                pass
        n = max(len(self._envs), 1)
        return torch.zeros(n, 2)

    def _build_rigid_body_state(self) -> torch.Tensor:
        """
        构建 (N_envs * N_bodies, 13) rigid body state 张量。
        源码通常做：rb_state.view(num_envs, -1, 13)
        """
        if self._robot is not None:
            try:
                s = self._robot.data.body_state_w.clone()  # (N_envs, N_bodies, 13)
                # wxyz → xyzw
                s[..., 3:7] = s[..., [4, 5, 6, 3]]
                return s.view(-1, 13)
            except Exception:
                pass
        n = max(len(self._envs), 1)
        return torch.zeros(n, 13)

    def _build_dof_force(self) -> torch.Tensor:
        if self._robot is not None:
            try:
                return self._robot.data.applied_torque.view(-1)
            except Exception:
                pass
        return torch.zeros(max(len(self._envs), 1))

    # ─────────────────────────────────────────────────────────────────────────
    # 调试
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        r = self._robot.__class__.__name__ if self._robot else "None"
        return (f"GymShim(robot={r}, "
                f"n_envs={len(self._envs)}, n_assets={len(self._assets)}, "
                f"objects={list(self._objects.keys())})")
