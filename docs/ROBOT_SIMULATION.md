> v0.5: for a complete restartable Windows demo, double-click `run_demo.bat`, or run `python -m imitation_learning demo`. The resulting `outputs/demo/robot/dashboard.html` contains all letters/methods, with shared per-letter durations and actual task-space LQR / covariance-aware LQR torque control. Original single-case commands below default to fixed joint PD. See [research improvements](RESEARCH_IMPROVEMENTS.md) for controller comparisons, force pulses and boundaries.

# 机械臂动力学仿真

此阶段将已学习的二维路径交给 MuJoCo 中的教学用两关节平面机械臂。它验证坐标映射、逆运动学、时间参数化、限幅力矩控制和实际模拟状态记录。它不是实机验证，也不是经过辨识的工业机械臂数字孪生。

## 同一页面切换字母（v0.4）

已有按字母保存的结果，例如 `outputs/robot-all/A`、`B` 到 `H`，只需：

```bash
python -m imitation_learning robot-view outputs/robot-all
```

Windows 打开：`start .\outputs\robot-all\dashboard.html`。此命令读取保存的 `manifest.json` 和完整 `robot_trace.npz`，验证误差与指标一致，然后生成自包含的合并页面；不需要重新执行动力学，也不需要 MuJoCo 依赖。单个案例的原始日志、报告和页面保持不变。

没有全部字母的仿真结果时，在安装可选依赖后一次生成：

```bash
python -m imitation_learning simulate outputs/full --all-letters --method KMP --output outputs/robot-switch
```

打开 `outputs/robot-switch/dashboard.html`，在页面的“字母”下拉框选择 A–H。`--letters A B` 可选择部分字母；`--letter B` 继续表示原来的单字母运行。三个选择参数互斥。

切换会暂停播放、回到该案例起点，并同步更新全部指标、通过/失败提示、时间范围和文件下载链接。方法下拉框只提供该字母实际保存的方法；只有一种方法时禁用。切换字母时优先保留同名方法，没有则自动选择一个可用方法。

合并页面只列出实际找到的案例。缺少案例、同一字母/方法有重复结果、缺少日志/报告，或保存的误差与完整位置日志不一致时会明确报错。通过完整运行但未达跟踪阈值的案例仍显示，失败状态不会被删除。

`robot-view --output outputs/selector.html` 可指定 HTML 文件位置，默认重建源目录下的 `dashboard.html`。CSV 和报告通过相对链接访问，分享时请保留案例子目录。不同 Windows 盘符间输出时使用文件绝对链接，建议将合并页与案例保存在同一目录树中。

批量仿真会先检查所选字母是否存在于学习结果，然后逐个输出开始和完成信息，最后生成合并页。每个字母仍只取所保存的第 0 折代表案例。仿真输出目录必须为空；读取已有结果时使用 `robot-view`。

## 安装和运行

推荐 Python 3.12。MuJoCo 通过官方 Python 包安装；本项目把它作为可选依赖，不改变普通学习实验所需的安装方式。

```bash
python -m pip install -e ".[simulation]"
python -m imitation_learning simulate outputs/full --letter A --method KMP --output outputs/robot-a
```

`outputs/full` 需包含已完成学习实验生成的 `examples.json`。如果只有快速实验，使用其目录并选择 G；也可直接使用仓库保存的 G 案例：

```bash
python -m imitation_learning simulate examples/validated_run --letter G --method KMP --output outputs/robot-g
```

Windows 打开：`start .\outputs\robot-g\dashboard.html`。其他系统直接用浏览器打开该文件，或运行 `python -m imitation_learning serve outputs/robot-g --port 8000`。

通过 `--method DMP/ProMP/GP/KMP/U-KMP` 选择方法；`--duration 12` 设置最短执行时长；`--width 0.12` 设置训练参考最长轴的工作区尺寸（米）。输出目录必须为空。程序不连接任何硬件，物理计算可在无显示器的环境执行。

## 输入和坐标

- 取 `examples.json` 中相应字母的 `reproduction` 案例：第 0 折、第一个保存的测试示教。不是完整五折实机/机器人仿真评测。
- 使用该折 `training_mean` 的包围框中心作为原始坐标原点，最长轴映射为默认 0.16 m。两个轴使用同一缩放，保持形状比例。
- 机器人工作区中心默认 `(0.40, 0.0)` m，高度固定为 0.30 m。相同字母/折的所有方法使用同一映射，不按各方法自己的预测范围缩放。
- 输入原始时间仅用于定义路径相位；执行相位采用五次时间变换 `s=10u^3-15u^4+6u^5`，位置用 PCHIP 插值，得到低起止速度的参考。
- 解析逆运动学使用连续的正肘角支路。不可达点、关节超限和非有限数据直接失败，不将路径强行裁剪到工作区。

## 教学模型与控制

| 项目 | 数值 / 约定 |
|---|---|
| 运动形式 | 两个竖直旋转轴；末端在水平面运动 |
| 连杆长度 | 0.32 m、0.28 m |
| 连杆质量 | 0.6 kg、0.4 kg；胶囊几何计算惯性 |
| 关节位置范围 | 肩部 `[-2.7,2.7]` rad；肘部 `[0.1,3.0]` rad |
| 关节阻尼 / armature | 0.05 N·m·s/rad / 0.01 kg·m² |
| 默认物理步长 | 0.002 s（500 Hz） |
| 参考速度 / 加速度预算 | 实际阈值的 80%，为跟踪暂态留余量 |
| 实际速度 / 加速度阈值 | 1 rad/s、2 rad/s² |
| 控制 | 相同固定关节 PD：`tau = Kp(q_ref-q)+Kd(qd_ref-qd)` |
| Kp / Kd | `[120,80]` / `[12,8]`，按 rad 与 SI 力矩定义 |
| 力矩限制 | 每个关节 ±5 N·m |
| 默认跟踪阈值 | 全部物理采样中的最大末端欧氏误差不超过 5 mm |

MuJoCo 模型保存在 `imitation_learning/robot_arm.xml`。初始关节配置为路径第一个点，初始速度为零；到达该点之前的接近运动没有被模拟。仅在初始化时设置关节位置，后续每个状态均来自 `mj_step`。

时间参数化反复延长执行时长，直到离散参考关节速度和加速度满足预算；默认至少 8 秒，最多 60 秒。达不到预算时拒绝运行。不能根据所有方法相同的最短时长，推断它们实际执行时长也相同。

每次积分后重新执行前向计算，记录与当前时间对齐的末端 site 状态；避免将上一物理步的缓存位置误记为当前测量。记录的力矩对应刚结束的时间区间。

## 输出与判定

| 文件 | 内容 |
|---|---|
| `dashboard.html` | 自包含回放；最多 1,201 帧，仅用于展示 |
| `robot_trace.csv` | 所有物理采样点的参考、实际关节、末端、力矩、误差和接触 |
| `robot_trace.npz` | 同一完整数组，便于数值复核 |
| `manifest.json` | 案例编号、版本、配置、坐标映射、时长、指标及输入/模型/源码哈希 |
| `model.xml` | 本次运行采用的精确 MuJoCo 模型 |
| `tracking.png` | 路径和全采样跟踪误差图 |
| `REPORT.md` | 通过状态、数值结果及范围解释 |

`target/tip/elbow` 单位为 m；`q/q_ref` 为 rad；`qd/qd_ref` 为 rad/s；`qdd_ref` 为 rad/s²；`torque` 为 N·m；时间为 s。CSV 的 `_0/_1` 分别代表 x/y 或肩部/肘部。平面高度为固定 0.30 m。

跟踪 RMSE 是 `sqrt(mean(||tip-target||²))`，是二维欧氏误差 RMSE；它与学习评测中的逐坐标 RMSE 定义不同。实际加速度由完整采样的关节速度差分计算。所有指标使用完整物理日志，不使用浏览器抽样帧。

`tracking_passed` 要求同时满足：最大末端误差阈值、无接触采样、关节位置范围，以及实际速度和加速度限制。只满足参考速度限制不代表实际运动也满足限制。力矩饱和比例单独报告。

未通过判定仍保存报告、完整日志和回放，CLI 返回 1。该默认通过标准只是教学仿真的预设准则，不是硬件安全认证。

## 验证和实机迁移

```bash
python -m unittest discover -s tests -p test_robot.py -v
```

新增 8 项测试覆盖：共享坐标映射、时间参数化、不可达/超限输入、模型与解析运动学一致性、限幅力矩对实际轨迹的影响、完整日志和指标一致性、失败保留和确定性。未安装 MuJoCo 时，4 项动力学测试跳过；安装可选依赖后均执行。

v0.4 额外检查旧日志合并、UTF-8 路径、不同案例的指标/链接、重复/损坏日志和真实 A/B 预测的批量仿真。浏览器交互测试用明确标注的合成 UI 夹具检查字母/方法切换、播放重置、失败提示清除、下载目标和手机布局；夹具不作为物理实验结果。

浏览器检查需安装 Playwright，使用系统 Chrome/Chromium（或设置 `ROBOT_BROWSER_EXECUTABLE`），在 `ROBOT_BROWSER_TESTS=1` 环境下运行 `python -m unittest discover -s tests -p test_robot_browser.py -v`。未选择此项时跳过浏览器测试，普通安装不添加浏览器依赖。

对完整学习输出中的 A–H、五种方法各取一个代表案例，可额外执行 40 个物理仿真，并保存每个案例的完整 NPZ 日志、汇总和失败记录：

```bash
python scripts/validate_robot.py --run outputs/full --output outputs/robot-validation
```

这不是 1,575 条全量交叉验证；它只检查所保存的代表路径在教学模型上的执行情况。已验证的 40 案例汇总见 [仿真检查记录](../examples/robot_validation/README.md)。

实机迁移时，需选择真实型号、使用厂商模型/驱动、标定工具和坐标系、检查整条机械臂的碰撞及限位、采集真实反馈。此教学模型不包含实际硬件参数误差、传感器噪声、通信延时、接触写字或障碍场景。

此处五种学习方法均使用同一固定 PD。特别是 KMP 与 U-KMP 的规划一致时，它们的此处结果也应一致。原二维 LQR 不在此模型中被冒充为机械臂控制器；要验证不确定性感知控制，需要独立设计匹配机械臂动力学和可用接口的控制对比实验。

参考官方资料：[MuJoCo Python 安装与 API](https://mujoco.readthedocs.io/en/stable/python.html)、[模型与执行器 XML](https://mujoco.readthedocs.io/en/stable/XMLreference.html)。
