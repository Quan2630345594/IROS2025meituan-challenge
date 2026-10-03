> **代码范围说明：本仓库代码只包括 solution 的两部分内容：①结构化导航指令解析；②单目 RGB + LiDAR BEV 融合感知。不包含完整的导航决策、路径规划、运动控制或比赛部署系统。**

# IROS2025 The 3rd Low-Altitude Economy Intelligent Flight Management Challenge

本项目将自然语言导航指令转换为有序目标步骤，并探索利用相机语义与激光雷达几何建立统一的鸟瞰图（BEV）特征。两个模块可独立使用，目前尚未形成端到端导航闭环。

指令解析模块包含结构化推理、数据构建、LoRA 训练入口与评估工具。融合感知模块包含几何预处理、可学习网络原型和 ROS1 几何诊断；**感知网络尚未训练，没有可直接使用的检测权重或解码后的目标位置。**

## 目录结构

```text
catkin_ws/scripts/
├── language_parse/
│   ├── parse_goal_direction.py   结构化 JSON 推理
│   ├── contract.py / schema.json 词表、输出契约与校验
│   ├── system_prompt.txt         解析规则
│   ├── build_dataset.py          数据构建与分组划分
│   ├── check_dataset.py          数据一致性检查
│   ├── train_lora.py / training/  LoRA 训练入口与配置
│   ├── evaluate.py               模型对比与离线评分
│   └── data/ / tests/            数据与回归测试
└── goal_detection/
    ├── bev/geometry.py           栅格化、坐标变换与投影
    ├── bev/model.py              图像 FPN 与 BEV 融合网络
    ├── bev/cli.py                离线预处理与前向验证
    ├── config/bev.json           网格、类别与通道配置
    ├── scripts/bev_features_node.py  ROS1 几何诊断
    ├── launch/decision.launch    ROS 启动入口
    ├── docs/design.md            架构与训练设计
    └── tests/                    几何、CLI 与网络测试
```

## 第一部分：结构化导航指令解析

### 技术实现

解析器默认通过 Ollama 调用 `qwen2.5vl:7b`，使用系统规则、训练集 few-shot 示例和 JSON Schema 约束输出。推理采用 temperature=0、单次请求，生成后严格检查字段、词表和状态一致性；也支持具有 JSON Schema 输出能力的 OpenAI 兼容服务。

| 字段 | 约定 |
| --- | --- |
| `direction` | `front`、`back`、`left`、`right` |
| `object` | `trash`、`bench`、`billboard`、`tree`、`tractor trailer`、`barrel`、`fire hydrant`、`traffic cone` |
| `target_gap_m` | 最终位置与目标物体的间隔，单位米；未指定时为 `null` |
| `status` | `complete` 或 `needs_clarification` |
| `issues` | 缺失信息、未知类别或歧义的问题代码与说明 |

例如“向前走到树那里，右转后直走到消防栓”的目标输出为：

```json
{
  "status": "complete",
  "steps": [
    {"direction": "front", "object": "tree", "target_gap_m": null},
    {"direction": "right", "object": "fire hydrant", "target_gap_m": null}
  ],
  "issues": []
}
```

规则处理多步顺序、中英文别名、`at/to`、否定与纠正，并区分移动方向与物体之间的位置关系。转向后到达目标前的直行属于该转向步骤；到达目标后的新直行段才产生新的 `front`。明确的目标间隔统一换算为米，例如 50 cm → 0.5 m；行进路程不会被当作目标间隔。

缺少方向或目标、存在未解决的选择、间隔无单位或冲突时返回 `needs_clarification`。其中可以保留确定步骤供展示，但上层必须先澄清，不能直接执行部分结果。Schema 校验保证输出契约，语义是否忠于原指令仍需标注数据评估。

### 推理使用

解析、数据处理和离线测试仅依赖 Python 3.8+ 标准库。模型推理需要已安装并启动的 Ollama 服务及对应模型。从仓库根目录执行：

```bash
cd catkin_ws/scripts/language_parse
ollama pull qwen2.5vl:7b
python3 parse_goal_direction.py '向前走到树那里，右转后直走到消防栓'
python3 parse_goal_direction.py '向前走到距离树50厘米的位置'
```

使用 `--model` 或 `LANGUAGE_PARSE_MODEL` 切换模型，使用 `--base-url` 或 `OLLAMA_BASE_URL` 配置 Ollama 地址，默认地址为 `http://localhost:11434`。成功退出码为 `0`，待澄清为 `2`，请求或输出格式失败为 `1`。

### 数据、LoRA 训练与评估

当前数据共 241 条，train / validation / test 分别为 185 / 23 / 33 条，使用随机种子 42，按改写族隔离。同一路线的中英文改写与不同间隔样本归为同族，以减少跨集合泄漏。标签来自规则筛选和手写模板，属于小规模起始数据，正式训练前仍需人工语义审查。

在 `language_parse` 目录检查数据与离线逻辑：

```bash
python3 check_dataset.py
python3 -m unittest discover -s tests -v
# 修改规则或别名后重建，再运行检查
python3 build_dataset.py
```

LoRA 基座为 `Qwen/Qwen2.5-VL-7B-Instruct`，采用 rank=8、alpha=32、学习率 1e-4、最多 3 轮，仅训练语言部分 LoRA，冻结视觉模块与对齐模块。训练只监督最终 assistant JSON，按验证 loss 选择最佳 checkpoint，并执行早停。配置见 `training/lora_config.json`。

在独立 Linux/CUDA 训练环境执行；PyTorch/CUDA 需按训练机配置：

```bash
cd catkin_ws/scripts/language_parse
python3 -m pip install -r requirements-training.txt
python3 check_dataset.py
python3 train_lora.py         # 检查并打印参数
python3 train_lora.py --run   # 实际启动训练
```

训练后可合并适配器，再用支持该模型和 JSON Schema 输出的 vLLM 服务部署。Ollama 模型包不能直接作为训练权重，适配器也不能假定可直接导入 Ollama。以下相对路径需替换为实际 checkpoint 和合并结果；vLLM 建议安装在独立服务环境：

```bash
swift export --adapters output/qwen2.5vl-navigation/checkpoint-best --merge_lora true
vllm serve output/qwen2.5vl-navigation/merged --served-model-name navigation-lora --max-model-len 8192
python3 parse_goal_direction.py '向前走到树那里' --backend openai --base-url http://localhost:8000/v1 --model navigation-lora --no-few-shot
```

评估入口提供Schema 基座和 LoRA + Schema 两组对比：

```bash
python3 evaluate.py --mode schema --model qwen2.5vl:7b
python3 evaluate.py --mode lora-schema --backend openai --base-url http://localhost:8000/v1 --model navigation-lora
```

## 第二部分：单目 RGB + LiDAR BEV 融合感知

### 技术实现

核心流程为“LiDAR 米制网格 → 图像特征提取 → 标定投影与特征提升 → BEV 融合 → 多任务输出头”。

1. **点云编码**：将点变换到机器人坐标系，按 xy 划分柱状网格，统计 `log(1+点数)`、平均高度、最大高度和最小高度，形成四通道特征。当前为几何统计，尚非学习式 PointPillars PFN。
2. **图像编码**：轻量 CNN 与自顶向下 FPN 生成 P2/P3/P4，默认通道数 64；当前用 P2 做特征提升。
3. **投影与提升**：通过 LiDAR → camera optical 外参和相机内参建立对应，过滤相机后方及视野外点，用像素深度筛选同像素更远点。双线性采样图像特征，再按 BEV 单元均值聚合；参考实现支持 batch size=1。
4. **门控融合**：由 LiDAR、相机特征及有效掩码生成 sigmoid 门控，计算 `LiDAR + gate × Camera × mask`，再卷积细化；无相机对应的区域不注入图像特征。
5. **输出头**：目标类别热图、8 通道 box 回归、BEV 语义、占用状态及图像 P2 类别 logits。box 参数为 xy 偏移、中心 z、长宽高的对数和朝向 sin/cos。

默认网格为 x∈[0,50) m、y∈[-25,25) m、分辨率 0.1 m，形状为 500×500。机器人坐标为 x 前、y 左、z 上；相机 optical 坐标为 x 右、y 下、z 前。

网络尚无训练权重、目标分配、loss 和 box 解码，时序融合与完整 2D box head 尚未实现。稀疏提升只能覆盖有可靠点云对应的区域，没有 LiDAR 返回的小目标无法通过该路径建立对应；像素深度筛选也不能解决全部遮挡。空网格不能直接解释为可通行区域。

### 离线使用

从仓库根目录进入模块。几何预处理需要 NumPy，网络前向还需与环境匹配的 PyTorch：

```bash
cd catkin_ws/scripts/goal_detection
python3 -m pip install numpy
python3 -m unittest discover -s tests -v
# 需要网络前向时安装完整依赖
python3 -m pip install -r requirements-bev.txt
```

自行准备 `frame.npz`，包含以下数组，禁止使用 object dtype：

| 键 | 格式与要求 |
| --- | --- |
| `image_rgb` | 去畸变后的 `H×W×3` RGB uint8 图像 |
| `points_lidar` | `N×3` 或 `N×4`，前三列 xyz，单位米 |
| `K` | 与该去畸变图像对应的 `3×3` 内参 |
| `T_camera_lidar` | `4×4`，LiDAR → camera optical |
| `T_robot_lidar` | `4×4`，LiDAR → 机器人 |

```bash
# 保存 pillar 特征、原始点索引、像素坐标与 BEV 行列索引
python3 -m bev.cli --input frame.npz --output outputs/features.npz
# 额外保存随机初始化网络的原始任务头、融合特征与有效掩码
python3 -m bev.cli --input frame.npz --output outputs/raw_heads.npz --forward
```

`--forward` 仅验证张量形状与数据流，不产生可用检测结果。可通过 `--config config/custom_bev.json` 指定自定义配置；默认网格较大，CPU 验证可以缩小网格范围。

### ROS1 几何诊断

使用 Linux / ROS Noetic。代码位于 `catkin_ws/scripts`，catkin 不会自动将其识别为 `src` 包，需要链接到 ROS 工作空间。从仓库根目录运行：

```bash
mkdir -p catkin_ws/src
ln -s ../scripts/goal_detection catkin_ws/src/goal_detection
cd catkin_ws
rosdep install --from-paths src --ignore-src -r -y
chmod +x src/goal_detection/scripts/bev_features_node.py
catkin_make
source devel/setup.bash
roslaunch goal_detection decision.launch \
  image_topic:=/magv/camera/image_compressed/compressed \
  cloud_topic:=/magv/scan/3d \
  camera_info_topic:=/magv/camera/camera_info \
  robot_frame:=base_link fixed_frame:=odom resolution:=0.1 sync_slop:=0.05
```

输入为 `CompressedImage`、含 xyz 的 `PointCloud2` 和与图像尺寸及 optical frame 匹配的 `CameraInfo`，同时提供连通的 TF。节点以 K 为新内参去畸变，仅支持 `plumb_bob`；已校正图像应提供对应 K 和 D=0。图像、点云时间戳必须非零且使用相同时间基准。双时间 TF 查询补偿帧间自车运动，不替代扫描内逐点 deskew。

rosbag 回放时可先执行 `rosparam set use_sim_time true`，再执行 `rosbag play --clock your.bag`，并确保提供所需 TF 与 CameraInfo。

使用 `rqt_image_view` 查看诊断话题：

| 话题 | 内容 |
| --- | --- |
| `/bev_features_node/pillar_log_count` | 每格 log(1+点数) |
| `/bev_features_node/pillar_mean_height` | 平均高度，米 |
| `/bev_features_node/camera_observed_mask` | 有相机对应为 255，否则为 0 |
| `/bev_features_node/projection_overlay` | 去畸变图像与 LiDAR 投影叠加 |

`decision.launch` 目前仅启动几何诊断，不运行学习式检测或机器人控制，也不发布 `/detected_positions`。ROS 网格范围固定为上述默认范围，分辨率通过启动参数调整；`config/bev.json` 仅用于离线 CLI。

更多用法见 [融合感知说明](catkin_ws/scripts/goal_detection/README.md)，架构、loss 与训练计划见 [设计文档](catkin_ws/scripts/goal_detection/docs/design.md)。

## 接入完整 solution

解析模块输出有序语言目标，感知模块当前输出几何诊断或未训练网络的原始张量。两者尚无目标匹配与导航接口，解析的 8 类物体与感知配置的 14 类目标也不完全对应。接入完整系统需统一类别、完成感知标注与训练、实现检测解码及带时间戳和坐标系的目标发布，并由上层处理澄清、目标选择、规划与控制。

离线测试覆盖解析契约、数据隔离和计分逻辑，以及几何变换、投影、散射聚合与输出形状；缺少 PyTorch 时网络测试会跳过。模拟服务测试不能作为模型准确率，随机网络前向不能作为感知效果。实际训练、真实服务集成及 ROS 运行仍需在目标环境验证。
