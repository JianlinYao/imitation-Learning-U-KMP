# Robot Imitation Learning: KMP and Uncertainty-Aware Control

二维示教轨迹学习、约束适配与可复现实验。保留原论文的 **DMP、ProMP、GP、KMP、U-KMP** 方法体系，以及**轨迹复现、缺失补全、障碍适配**三个任务。

新增的 `imitation_learning/` 是统一运行入口。`Imitationlearning-KMP/` 和 `2Dletters/` 中原有 Notebook 保留作历史研究记录；其中部分旧评测单元包含占位实现、不同实验协议或结果过滤，不作为新版评测入口。

> 新版结果由程序重新计算，不表示已经复现论文原表格中的数字。KMP 与 U-KMP 的规划均值可以相同；U-KMP 的不确定性感知控制效果在单独的 LQR 实验中评估。项目目前是二维数值实验，没有实机安全验证。

## 快速开始

在仓库根目录运行，使用 Python 3.10 或以上版本：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux 激活环境：

```bash
source .venv/bin/activate
```

随后安装并运行：

```bash
python -m pip install -e .
python -m imitation_learning inspect
python -m imitation_learning run --config configs/quick.json --output outputs/quick
```

打开 `outputs/quick/dashboard.html` 即可交互查看结果，无需联网或前端依赖。可以选择字母、实验和区间方法，播放轨迹，查看逐案例指标与全部汇总。`quick.json` 只运行 G 的一个测试折，是快速验证，不是完整五折结果。

也可以通过本地 HTTP 查看：

```bash
python -m imitation_learning serve outputs/quick --port 8000
```

浏览器访问 `http://127.0.0.1:8000/dashboard.html`。服务只绑定本机回环地址。

## 没有真实机械臂：运行 MuJoCo 物理仿真

v0.4 支持在**同一个页面下拉切换字母**。如果已经生成了 `outputs/robot-all/A` 到 `H` 的仿真结果，直接合并已有日志：

```bash
python -m imitation_learning robot-view outputs/robot-all
```

然后打开 `outputs/robot-all/dashboard.html`。选择字母时，机械臂回放、误差指标、通过状态及 CSV/报告链接会同步更新。合并已有日志不需要重新运行物理仿真，也不需要安装 MuJoCo。

如果尚未生成全部字母，可以一次运行 A–H 并自动生成选择页面：

```bash
python -m pip install -e ".[simulation]"
python -m imitation_learning simulate outputs/full --all-letters --method KMP --output outputs/robot-switch
```

Windows 打开合并页：

```powershell
start .\outputs\robot-switch\dashboard.html
```

输出会逐个显示 `Robot case 1/8` 等进度。也可用 `--letters A B` 只运行指定字母。请使用新的仿真输出目录；页面只列出实际存在的案例。旧的单字母页面不会通过刷新自动获得其他字母，需要打开新生成的合并页。

v0.3 新增带关节惯性、阻尼、力矩限幅和反馈的**两关节教学机械臂**。在 Windows / Linux 的 Python 3.12 环境中安装可选依赖：

```bash
python -m pip install -e ".[simulation]"
python -m imitation_learning simulate outputs/full --letter A --method KMP --output outputs/robot-a
```

Windows PowerShell 打开结果：

```powershell
start .\outputs\robot-a\dashboard.html
```

如果还没有完整学习结果，可以直接使用仓库内保存的 G 案例：

```bash
python -m imitation_learning simulate examples/validated_run --letter G --method KMP --output outputs/robot-g
```

页面显示模拟机械臂的运动、参考与实际末端路径、跟踪误差及力矩饱和情况。仿真无须显示器或 OpenGL 窗口；播放使用本地浏览器。无需 ROS 即可运行此教学模型。

输入是一次已保存的轨迹复现案例，不是重新训练。坐标按训练参考统一缩放成米，逆运动学得到关节轨迹，再按速度/加速度限制重新定时；默认至少 8 秒。MuJoCo 实际积分产生后续状态，固定 PD 发送限幅力矩。失败结果也保留，未达到跟踪标准时退出码为 1。

这一新增阶段是**教学模型的动力学仿真**，没有对应真实厂商机械臂，也没有验证实机或 U-KMP 的不确定性感知控制效果。实现、输出单位、通过标准和实机迁移条件见 [机械臂仿真说明](docs/ROBOT_SIMULATION.md)。

## 完整实验

```bash
python -m imitation_learning run --config configs/benchmark.json --output outputs/full
```

默认在 A—H 的 **105 条示教**上，按完整轨迹进行字母内五折划分；每条轨迹只在一个测试折出现。五种方法、三个任务共 **1,575 条评测记录**。每一方法/折只训练一次，三个任务共用同一训练结果。

例如只测试 G、H，或只比较核方法：

```bash
python -m imitation_learning run --letters G H --output outputs/gh
python -m imitation_learning run --letters G --methods GP KMP U-KMP --output outputs/kernel
```

输出目录必须为空，程序不会覆盖已有实验。更改参数后使用新目录。配置文件可设置基函数数量、核长度、GMM 分量数、参考点数、遮挡范围、观测数量、障碍半径和修复次数。

## 输出与指标

| 输出 | 内容 |
|---|---|
| `dashboard.html` | 自包含离线实验查看器 |
| `REPORT.md` | 新运行结果及解释 |
| `metrics.csv` | 每条测试示教、每个方法和任务的实际指标与状态 |
| `summary.csv` | 逐任务/方法的均值、样本标准差、有效数、失败数 |
| `fold_metrics.csv` | 逐字母/折/方法的均值；用于检查折间稳定性 |
| `control_ablation.csv` | 固定 Q 与协方差感知 LQR 的独立数值对比 |
| `effective_config.json` | 本次运行的完整有效配置 |
| `manifest.json` | 输入/源码哈希、环境版本、随机种子、计数和时间 |
| `splits.json` | 每折原始示教编号、训练集归一化参数、GMM 训练信息 |
| `predictions/` | 每次预测的 NPZ：时间、真值、位置/速度、边际协方差、遮挡标记 |
| `figures/` | 根据本次结果生成的轨迹图与误差图 |

统一计算坐标 RMSE、欧氏终点误差、位置三阶时间导数的平均范数（Jerk）、缺失区间 RMSE，以及训练、适配和预测耗时。概率方法另外记录名义 95% 区间的实际覆盖率。DMP 的协方差相关指标为空，不编造区间。

避障还记录整条折线路径的最小间距、碰撞、是否满足安全余量和是否到达目标。绕行可能增加对原示教的 RMSE，因此不能只靠 RMSE 判断避障好坏。

## 算法与评测约定

- 所有方法使用同一训练/测试划分、坐标尺度和时间轴。训练集拟合共享的空间归一化参数。
- KMP 的 GMM/GMR 参考均值与完整位置/速度协方差在每个训练折内重新生成。新版不读取现有的 `ref*.mat`。
- DMP 从所有训练示教联合拟合强迫项，使用正确的弹簧阻尼符号与数值积分；它作为只接收起终点的基线，不假装具备避障耦合或片段条件化。
- ProMP 学习基函数权重的高斯分布；GP 使用全部示教。GP 对同时间输入的重复观测进行精确压缩，训练计算和计时真实执行。
- GP/ProMP/KMP 使用学习到的联合轨迹分布接收位置或位置/速度观测。KMP 的条件化实现采用高斯后验更新，具体区别见方法说明。
- 缺失补全默认遮挡测试轨迹的中间 20%；先遮挡，再分别平滑两侧可见片段。被遮挡位置及其导数不进入输入。
- 障碍由训练参考轨迹定义；所有方法面对相同障碍。适配方法使用同一几何途经点政策与有界修复预算。失败也保存，不过滤差结果。
- KMP 与 U-KMP 规划层使用同一公式和参数。U-KMP 另通过预测协方差构造 LQR 状态代价；控制实验比较同一参考、同一系统、同一输入代价和同一外部扰动。

详细实现、公式、单位和局限见 [方法与实验协议](docs/METHODOLOGY.md)；与原论文的对应关系见 [论文实现对照](docs/DISSERTATION_MAPPING.md)。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖核导数、DMP 强迫项、GP 重复观测等价性、概率条件化、KMP 外推方差、LQR 增益、整段碰撞检测、遮挡泄漏、折划分、指标与保存预测的一致性，以及固定配置的重复运行。

GitHub Actions 在 Python 3.10 和 3.12 上运行测试及快速实验，另在 Windows / Ubuntu 的 Python 3.12 上检查 MuJoCo 仿真。CI 产物包含查看器和结果表。原学习部分的 **16 项测试**已通过，安装仿真依赖后共 **24 项测试**通过；已完成 **1,575 条学习全量评测，零运行错误**，以及 **40 个代表路径的机械臂仿真检查**。学习结果见 [已验证示例](examples/validated_run/README.md)，检查范围与限制见 [验证记录](docs/VALIDATION.md)。

![G 的障碍适配示例](examples/validated_run/figures/trajectory_obstacle.png)

## 原始资料与署名

本项目围绕硕士论文 *Uncertainty-Aware KMP for Robot Imitation Learning: A Comparative Study with DMP, ProMP and GPR* 展开。论文原文件未在本次代码完善中修改。

- Huang et al. (2019), [Kernelized Movement Primitives](https://arxiv.org/abs/1708.08638).
- Silvério et al. (2019), [Uncertainty-Aware Imitation Learning using Kernelized Movement Primitives](https://arxiv.org/abs/1903.02114).
- 原作者的 [robInfLib-matlab](https://github.com/yanlongtu/robInfLib-matlab) 提供 KMP 与不确定性感知示例。

数据及已有研究代码的来源说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。本次完善没有将已有算法表述为新提出的算法。
