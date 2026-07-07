"""
base_task_compat.py  ──  BaseTask / VecTask Isaac Lab 兼容基类
=============================================================
源代码通常继承：
    class ShadowHandKettle(BaseTask):  ...

Isaac Lab 的对应基类是 DirectRLEnv 或 ManagerBasedRLEnv。
本模块提供一个过渡基类 IsaacLabBaseTask，它：

  1. 继承 DirectRLEnv（需要 Isaac Lab 已安装）
  2. 在 __init__ 里实例化 GymShim 并赋给 self.gym
  3. 把 Isaac Gym 的 VecTask 接口映射到 DirectRLEnv 的回调钩子
  4. 保持 self.num_envs / self.device / self.dt 等属性不变

迁移步骤（最小改动）
--------------------
原代码：
    from bidexhands.tasks.hand_base.base_task import BaseTask
    class ShadowHandKettle(BaseTask):
        def __init__(self, cfg, sim_params, physics_engine, device_type,
                     device_id, headless):
            super().__init__(cfg=cfg, sim_params=sim_params, ...)

只需改两行：
    # 1) 修改 import
    from base_task_compat import IsaacLabBaseTask as BaseTask
    # 2) __init__ 签名不变，内部逻辑不变
"""

from __future__ import annotations
import torch
from typing import Any, Dict, Optional, Tuple

# ── 尝试导入 Isaac Lab；若未安装则降级为纯 Python 桩 ──────────────────────
try:
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.sim import SimulationContext
    _ISAACLAB_AVAILABLE = True
except ImportError:
    _ISAACLAB_AVAILABLE = False
    DirectRLEnv = object          # 降级：用 object 作基类

from isaacgym.gym_shim import GymShim
from isaacgym import gymapi


# ─────────────────────────────────────────────────────────────────────────────
# 配置解析辅助
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sim_device(device_type: str, device_id: int) -> str:
    """将 Isaac Gym 的 (device_type, device_id) 转换为 torch device 字符串。"""
    if device_type in ("cuda", "gpu"):
        return f"cuda:{device_id}"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# IsaacLabBaseTask
# ─────────────────────────────────────────────────────────────────────────────

class IsaacLabBaseTask(DirectRLEnv):
    """
    VecTask / BaseTask → DirectRLEnv 过渡基类。

    子类只需：
    1. 实现 _create_envs(num_envs, spacing, num_per_row)
    2. 实现 compute_observations() → 填 self.obs_buf
    3. 实现 compute_reward()       → 填 self.rew_buf, self.reset_buf
    4. 实现 pre_physics_step(actions)
    5. 实现 post_physics_step()

    以下属性由本基类负责初始化，子类直接使用：
        self.gym        GymShim 实例（替代 isaacgym gym 句柄）
        self.sim        SimulationContext（或 None if not available）
        self.device     torch.device 字符串
        self.num_envs   int
        self.dt         float（仿真步长）
        self.obs_buf    Tensor (num_envs, num_obs)
        self.rew_buf    Tensor (num_envs,)
        self.reset_buf  Tensor (num_envs,)  1=需要 reset
        self.progress_buf Tensor (num_envs,)
    """

    # ── 子类可覆盖的类属性 ──────────────────────────────────────────────────
    num_obs:     int = 1
    num_actions: int = 1

    def __init__(self,
                 cfg:            Dict[str, Any],
                 sim_params:     Optional[gymapi.SimParams] = None,
                 physics_engine: int = gymapi.SIM_PHYSX,
                 device_type:    str = "cuda",
                 device_id:      int = 0,
                 headless:       bool = True):

        # ── 基础属性 ──────────────────────────────────────────────────────
        self.cfg            = cfg
        self.sim_params     = sim_params or gymapi.SimParams()
        self.physics_engine = physics_engine
        self.headless       = headless

        self.device = _parse_sim_device(device_type, device_id)
        self.num_envs    = int(cfg.get("env", {}).get("numEnvs", 1))
        self.num_obs     = int(cfg.get("env", {}).get("numObservations",
                                                       self.__class__.num_obs))
        self.num_actions = int(cfg.get("env", {}).get("numActions",
                                                       self.__class__.num_actions))
        self.dt          = self.sim_params.dt

        # ── GymShim（self.gym）──────────────────────────────────────────
        self.gym = GymShim()
        self.sim = self.gym.create_sim(
            device_id, device_id, physics_engine, self.sim_params)

        # ── 公共张量 ─────────────────────────────────────────────────────
        self.obs_buf      = torch.zeros(
            self.num_envs, self.num_obs,     device=self.device)
        self.rew_buf      = torch.zeros(
            self.num_envs,                   device=self.device)
        self.reset_buf    = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device)
        self.progress_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.randomize_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.extras: Dict[str, Any] = {}

        # ── 场景初始化（由子类实现）────────────────────────────────────
        spacing     = float(cfg.get("env", {}).get("envSpacing", 1.5))
        num_per_row = int(cfg.get("env",  {}).get("numEnvsPerRow",
                                                   max(1, int(self.num_envs ** 0.5))))
        self._create_envs(self.num_envs, spacing, num_per_row)

        # Isaac Lab DirectRLEnv 需要 cfg 对象；若框架可用则走正式路径
        if _ISAACLAB_AVAILABLE:
            lab_cfg = self._build_lab_cfg()
            super().__init__(cfg=lab_cfg)
        # else: 仅作 Python 层初始化（用于单元测试 / 离线脚本）

    # ─────────────────────────────────────────────────────────────────────────
    # 子类必须实现的接口（与原 VecTask 相同）
    # ─────────────────────────────────────────────────────────────────────────

    def _create_envs(self, num_envs: int, spacing: float, num_per_row: int):
        """
        创建场景资产、env handles、actor handles。
        在 Isaac Lab 中：声明 InteractiveSceneCfg 并在 _setup_scene() 里实例化。
        """
        raise NotImplementedError

    def compute_observations(self):
        """填写 self.obs_buf。"""
        raise NotImplementedError

    def compute_reward(self, actions: torch.Tensor):
        """填写 self.rew_buf 和 self.reset_buf。"""
        raise NotImplementedError

    def pre_physics_step(self, actions: torch.Tensor):
        """在每个仿真步之前应用动作。"""
        raise NotImplementedError

    def post_physics_step(self):
        """仿真步之后更新状态、调用 compute_observations / compute_reward。"""
        raise NotImplementedError

    # ─────────────────────────────────────────────────────────────────────────
    # DirectRLEnv 回调（把 Isaac Lab 调用路由到 VecTask 风格方法）
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        """
        Isaac Lab 钩子。子类在 _create_envs 中已完成资产配置，
        这里只需 clone 环境并刷新张量。
        """
        if _ISAACLAB_AVAILABLE and hasattr(self, "scene"):
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor):
        self.pre_physics_step(actions)

    def _apply_action(self):
        """Isaac Lab 在此帧调用；动作已在 _pre_physics_step 中处理。"""
        pass

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        self.post_physics_step()
        return {"policy": self.obs_buf}

    def _get_rewards(self) -> torch.Tensor:
        return self.rew_buf

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # terminated / truncated
        return self.reset_buf.bool(), torch.zeros_like(self.reset_buf, dtype=torch.bool)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.reset_buf[env_ids]    = 0
        self.progress_buf[env_ids] = 0

    # ─────────────────────────────────────────────────────────────────────────
    # 辅助
    # ─────────────────────────────────────────────────────────────────────────

    def _build_lab_cfg(self):
        """
        构造最小化的 DirectRLEnvCfg 供 super().__init__ 使用。
        完整迁移时应在子类里声明完整的 EnvCfg dataclass。
        """
        if not _ISAACLAB_AVAILABLE:
            return None
        cfg = DirectRLEnvCfg()
        cfg.num_envs        = self.num_envs
        cfg.observation_space = self.num_obs
        cfg.action_space    = self.num_actions
        cfg.episode_length_s = float(
            self.cfg.get("env", {}).get("episodeLength", 600)) * self.dt
        return cfg

    def allocate_buffers(self):
        """VecTask 兼容：重新分配张量（num_envs 变化时调用）。"""
        self.obs_buf      = torch.zeros(self.num_envs, self.num_obs,     device=self.device)
        self.rew_buf      = torch.zeros(self.num_envs,                   device=self.device)
        self.reset_buf    = torch.ones( self.num_envs, dtype=torch.long, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def step(self, actions: torch.Tensor):
        """
        VecTask 风格的 step 接口（供非 Isaac Lab 脚本直接调用）。
        Isaac Lab 框架会调用 _pre_physics_step / _get_observations 等；
        本方法仅在离线测试时使用。
        """
        self.pre_physics_step(actions)
        if self._sim_ctx_available():
            self.sim_ctx.step()
        self.post_physics_step()
        return (self.obs_buf, self.rew_buf,
                self.reset_buf, self.extras)

    def reset(self):
        """VecTask 风格的 reset，返回初始观测。"""
        self.reset_buf[:] = 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_idx(env_ids)
        self.compute_observations()
        return self.obs_buf

    def _sim_ctx_available(self) -> bool:
        return (self.gym._sim_ctx is not None and
                hasattr(self.gym._sim_ctx, "step"))


# ─────────────────────────────────────────────────────────────────────────────
# 向后兼容别名
# ─────────────────────────────────────────────────────────────────────────────
BaseTask = IsaacLabBaseTask
VecTask  = IsaacLabBaseTask
