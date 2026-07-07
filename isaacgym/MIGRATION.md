# Isaac Gym → Isaac Lab 兼容层迁移指南

## 目录结构

```
project/
├── isaacgym/                   ← ★ 本兼容包，覆盖已安装的 Isaac Gym
│   ├── __init__.py
│   ├── gymapi.py               # 数据类型 / 常量（纯 Python）
│   ├── gymtorch.py             # wrap/unwrap_tensor 恒等函数
│   ├── gymutil.py              # parse_arguments / parse_sim_config 等
│   ├── torch_utils.py          # 全部四元数 / 向量数学（纯 PyTorch）
│   └── gym_shim.py             # GymShim：self.gym 替代对象
├── base_task_compat.py         # ★ IsaacLabBaseTask（替换 VecTask/BaseTask）
├── bidexhands/
│   └── tasks/
│       ├── shadow_hand_kettle.py   ← 源码，改动 ≤ 3 行
│       └── hand_base/
│           ├── base_task.py        ← 只改一行 import
│           ├── meta_vec_task.py    ← 只改 import
│           └── torch_jit_utils.py  ← 不改（from isaacgym.torch_utils import *）
└── ...
```

---

## 文件逐一改动说明

### 1. `bidexhands/tasks/hand_base/base_task.py`  ── 改 1 行

```python
# ─── 原代码 ───────────────────────────────────────────────────
from isaacgym import gymapi
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples
# class BaseTask(...):  ...  不变

# ─── 只需额外加一行 import，把 VecTask 基类换掉 ───────────────
from base_task_compat import IsaacLabBaseTask as VecTask  # ← 新增
# 其余代码不变
```

### 2. `bidexhands/tasks/hand_base/meta_vec_task.py`  ── 改 2 行

```python
# 原
from isaacgym import gymtorch
# 新（不变，因为 isaacgym/ 兼容包已接管）：无需修改
```

### 3. `bidexhands/tasks/shadow_hand_kettle.py`  ── 改 1 行

```python
# 原
from bidexhands.tasks.hand_base.base_task import BaseTask
# 新
from base_task_compat import IsaacLabBaseTask as BaseTask  # ← 替换
```

`_create_envs`、`pre_physics_step`、`post_physics_step`、`compute_reward`
等方法**内部逻辑完全不动**。

### 4. `bidexhands/utils/torch_jit_utils.py`  ── **不改**

```python
from isaacgym.torch_utils import *  # 已由 isaacgym/torch_utils.py 接管
```

### 5. `bidexhands/utils/config.py`  ── **不改**

```python
from isaacgym import gymapi
from isaacgym import gymutil
gymutil.parse_sim_config(cfg["sim"], sim_params)  # 已由 gymutil.py 接管
```

---

## self.gym 注册流程（在 `_create_envs` 末尾添加）

```python
def _create_envs(self, num_envs, spacing, num_per_row):
    # ── 原有 Isaac Gym 代码（加载资产、创建 env/actor 等）──────
    # 这些调用已经由 GymShim 接管，运行不会报错
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, asset_options)
    ...
    for i in range(num_envs):
        env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
        hand_handle = self.gym.create_actor(env_ptr, self.hand_asset, ...)
        ...

    # ── ★ 新增：注册 Isaac Lab 对象（Isaac Lab 初始化完成后）──────
    self.gym.register(
        robot  = self.robot,    # Articulation（Isaac Lab 实例）
        obj    = self.object,   # RigidObject
        goal   = self.goal,     # RigidObject（可选）
        sim    = self.sim,
    )
```

---

## 关键 API 对照表

| Isaac Gym（源码中） | Isaac Lab 等效 | 兼容层处理方式 |
|---|---|---|
| `gymtorch.wrap_tensor(t)` | 直接使用张量 | **identity，不改代码** |
| `gymtorch.unwrap_tensor(t)` | 直接使用张量 | **identity，不改代码** |
| `gym.acquire_dof_state_tensor(sim)` | `articulation.data.joint_pos/vel` | GymShim 自动拼装 |
| `gym.acquire_rigid_body_state_tensor(sim)` | `articulation.data.body_state_w` | GymShim 自动拼装 |
| `gym.acquire_actor_root_state_tensor(sim)` | `articulation.data.root_state_w` | GymShim 自动拼装 |
| `gym.refresh_*_tensor(sim)` | 自动刷新 | **no-op，不改代码** |
| `gym.set_actor_root_state_tensor_indexed(...)` | `articulation.write_root_state_to_sim(...)` | GymShim 路由 |
| `gym.set_dof_state_tensor_indexed(...)` | `articulation.write_joint_state_to_sim(...)` | GymShim 路由 |
| `gym.set_dof_position_target_tensor_indexed(...)` | `articulation.set_joint_position_target(...)` | GymShim 路由 |
| `gym.apply_rigid_body_force_tensors(...)` | `articulation.set_external_force_and_torque(...)` | GymShim 路由 |
| `gym.simulate(sim)` | `sim_ctx.step()` | GymShim 路由 |
| `gym.fetch_results(sim, True)` | 自动同步 | **no-op** |
| `gymapi.Vec3 / Quat / Transform` | 纯 Python 数据类 | **gymapi.py 完整实现** |
| `gymapi.AssetOptions` | `ArticulationCfg / RigidObjectCfg` | **gymapi.py 存储配置** |
| `from isaacgym.torch_utils import *` | `isaaclab.utils.math.*` | **torch_utils.py 纯 PyTorch 实现** |

---

## 四元数约定差异（！重要）

| | 顺序 |
|---|---|
| Isaac Gym / 本兼容层 | **xyzw** |
| Isaac Lab / Isaac Sim | **wxyz** |

GymShim 在 `_build_root_state()` 和 `set_actor_root_state_tensor_indexed()` 
里自动做转换，**源代码里不需要手动转换**。

---

## 肌腱（Tendon）支持

Shadow Hand 使用肌腱。Isaac Lab 目前不原生支持肌腱建模，有两种替代方案：

1. **等效刚度增益**：在 `ActuatorCfg` 里把耦合关节的 stiffness / damping 配成与肌腱力矩等效的值。
2. **外力注入**：在 `pre_physics_step` 里用 `apply_rigid_body_force_tensors` 
   手动计算肌腱力并注入（GymShim 已支持此调用）。

---

## 不需要改动的文件

- `bidexhands/utils/torch_jit_utils.py`（`from isaacgym.torch_utils import *`）
- `bidexhands/utils/config.py`（`gymutil.parse_sim_config`）
- `bidexhands/tasks/hand_base/meta_vec_task.py`（`from isaacgym import gymtorch`）
- 所有任务文件中的数学计算、奖励函数、观测计算部分

---

## 最终需要完整重写的部分（超出兼容层能力）

| 模块 | 原因 | Isaac Lab 替代 |
|---|---|---|
| URDF/MJCF 资产加载 | USD 格式不同 | `isaaclab_assets` 里用 `UsdFileCfg` / 离线转换器 |
| 渲染 / Viewer | Isaac Sim AppLauncher | `AppLauncher(headless=True)` |
| domain randomization | event manager 结构不同 | `EventTermCfg` + `randomize_*` |
| 相机传感器 | `TiledCamera` / `CameraCfg` | `isaaclab.sensors.CameraCfg` |
| force sensor | `ContactSensorCfg` | `isaaclab.sensors.ContactSensor` |
