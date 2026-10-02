# 单目 Camera–LiDAR BEV 系统设计与实施契约

本文区分已经落地的原型和下一阶段训练系统。以单相机、前向 50m、横向 50m、近距离小型基础设施为任务边界。以下张量统一 NCHW，BEV 行=y、列=x；尺寸和耗时是设计预算，不是已测性能。

## 网络结构和张量

```text
RGB → stride 4/8/16 CNN → FPN P2/P3/P4 ─→ image semantic logits
                              │              └→ 后续 2D box head
                       P2 sampled at LiDAR projections
                              │
                     valid-mask + mean BEV scatter
                              ↓
LiDAR → robot transform → pillar stats → encoder → gated residual fusion
                                                   │
                              ┌────────────────────┼─────────────────┐
                         Center heatmap/box      semantic logits    occupancy logits
```

默认 B=1、图像 720×1280、C=64、BEV 0.1m：

| 模块 | 张量 | 作用 |
|---|---|---|
| RGB | 1×3×720×1280 | [0,1] RGB；原型不采用 ImageNet normalization |
| P2 | 1×64×180×320 | stride 4 保留小物体纹理；承接语义与 lifting |
| P3 | 1×64×90×160 | 中尺度上下文 |
| P4 | 1×64×45×80 | 大尺度上下文，向上融合 |
| LiDAR points | N×3（生产版 N×4） | 米制几何；生产版可加入强度 |
| pillar statistics | 1×4×500×500 | log count、mean/max/min z |
| LiDAR feature | 1×64×500×500 | 两层保持分辨率的卷积 |
| Camera feature | 1×64×500×500 | 有返回点格子的采样均值 |
| validity mask | 1×1×500×500 | 相机语义可用区域，非占据/可见性真值 |
| fused BEV | 1×64×500×500 | 统一几何与语义 |
| center heatmap | 1×14×500×500 | 每类中心 logits |
| box regression | 1×8×500×500 | dx/dy、z、log l/w/h、sin/cos yaw |
| semantic | 1×6×500×500 | road/sidewalk/grass/building/obstacle/unknown |
| occupancy | 1×3×500×500 | free/occupied/unknown |
| image logits | 1×14×180×320 | 原型像素类别辅助头；不是 2D detector |

当前轻量 CNN 是方便验证数据流的随机初始化实现。训练版首选带 stride 4 FPN 的 ResNet-18 或轻量 ConvNeXt，以实测小目标 recall、延迟和显存选型；ResNet 更便于替换和导出。Swin/DINOv2 可作为离线精度对照，不能假设大 backbone 带来实时收益。正式图像检测头保留多尺度位置/分类分支；不能把所有细粒度语义压进稀疏 BEV。

## LiDAR 定义的空间网格

`P_robot = T_robot_lidar · P_lidar`，以 robot x 前、y 左、z 上栅格化。范围为半开区间；`col=floor((x-x_min)/r)`，`row=floor((y-y_min)/r)`。零计数表示无返回，不等于 free。

原型实现确定性统计；下一步替换为 PointPillars PFN：点输入 xyz/intensity，加 cluster-relative 和 pillar-center-relative 坐标，逐点 MLP+max pooling，然后散射同一网格。按点上限确定性采样，报告被丢弃比例。柱状表示有利实时性但损失垂向结构；建议追加低/中/高分层 pillar，避免桥下、树冠、杆体在 BEV 混叠。

VoxelNet 保留 xyz 体素结构但编码成本更高；SECOND 用稀疏 3D 卷积保留垂向几何，适合柱状编码定位不足、GPU 预算足够时替换。先评估 PointPillars，再决定是否升级，不能直接用车类默认 voxel size。

## Camera → BEV 的具体路径

统一约定 `T_a_b` 是 b→a，列向量齐次坐标。相机使用 optical 坐标（x 右、y 下、z 前）：

```text
P_camera = T_camera_lidar P_lidar
u = fx X/Z + cx,  v = fy Y/Z + cy,  Z > 0
P_robot  = T_robot_lidar P_lidar
feature(u,v) → pillar(P_robot.x,P_robot.y)
```

必须先按 K/D 去畸变，并使用校正图像对应的新 K；相机 link 若是 x 向前，必须先变到 optical frame，不能直接套 z-depth 公式。FPN采样以原图像像素映射到 feature 网格，当前 convention 是 `u_f=u·W_f/W`、`v_f=v·H_f/H`，`grid_sample` 使用 align_corners=False；更换 backbone/裁剪流程必须核对像素中心偏移。不同图像裁剪/缩放应相应变更 K。

当前：删除 NaN/相机后方/出画面点，同像素保留最近 Z，再对 P2 bilinear sampling，按机器人 BEV cell 均值 scatter；用 mask 表示有效 Camera 语义。z-buffer 防止同像素远处点获取近处物体语义，但不能解决相邻像素遮挡、透明表面或异步运动。

| 提升方法 | 对本任务的取舍 |
|---|---|
| 地面平面 geometry projection | 便宜；竖立设施不在地面，不能为通用物体提供真实高度 |
| LSS depth distribution | 可为无点区域提供稠密语义；需要深度监督/训练与深度 bins 显存 |
| BEV query cross attention | 灵活；仍需几何或深度解释，单相机视野外语义不足，调试复杂 |
| LiDAR-assisted depth lifting | 优先；有实测深度，误差来源可解释，但稀疏返回限制小目标覆盖 |
| point-feature correspondence | 本原型采用；简单可验证，但放弃无点像素的稠密语义 |

后续采用 LiDAR 稀疏深度监督一个小 depth head，而非无约束扩散：预测深度分布、用可靠点监督、遮挡与置信度加权，按深度 bins 提升补充稠密 Camera BEV。其贡献必须与当前稀疏提升分别评估。LiDAR 已提供 metric grid，因此首版没有引入 BEVFormer-style 全局 query 或时序 attention 的必要。

## 融合选择

本原型的 gate 输入 `[F_lidar,F_camera,M]`，输出逐通道 sigmoid 权重：

`F_fused = Conv3x3(F_lidar + sigmoid(Conv1x1([...])) ⊙ F_camera ⊙ M)`。

掩码保证没有 Camera 对应时不会注入该分支；它并不等于学习得到了可靠的置信度。训练时做 Camera dropout、点稀疏化、曝光扰动与轻微标定扰动，避免过度依赖相机。

| 机制 | 成本与取舍 |
|---|---|
| concat + CNN | 适合消融，但不能显式处理模态缺失 |
| 固定 weighted fusion | 很便宜，无法自适应传感器质量 |
| gated fusion | 首版选择：局部、可解释、O(HWC) 量级 |
| 全局 cross attention / Transformer | 250k cells 的全局 attention 是 O((HW)²)，不适合该高分辨率网格 |
| window cross attention | 后续可用 LiDAR Q、Camera K/V，控制邻域窗口 |
| deformable attention | 固定少量采样点，利于对齐偏差；部署和训练复杂度较高 |

双向 attention 只在单向融合精度收益明确时追加，不为架构形式增加算力。

## 小目标网格与检测头

0.3m 目标在 0.5m cell 上不到一格；0.1m 上约 3 格，0.05m 上约 6 格。**检测输出 stride 同样重要**：0.1m 栅格若输出 stride=4，又变成0.4m。当前保持全分辨率输出；训练版采用 BEV FPN，小物体在 stride=1 或 2 的 head，大物体在低分辨率 head。

先用规则 0.1m 网格完成端到端验证，再追加独立近场 crop：x=[0,15)、y=[-10,10)、0.05m，400×300。中/远场保留规则 0.1/0.2m。各 head 在米制坐标解码，再按类别合并/NMS；crop 交界用重叠区域与明确的训练归属，避免重复/漏检。分段非均匀网格会破坏常规卷积的固定米制邻域，首版不使用一张非均匀 tensor。

CenterPoint-style head 使用按类别中心热图与中心处回归。建议以 Gaussian focal heatmap loss、中心 offset/z SmoothL1、log-size SmoothL1、sin/cos yaw loss 训练。尺寸小的目标设置最小 Gaussian 半径并保留 sub-cell offset；同类同 cell 双中心仍冲突，增加近场分辨率或额外 query head。消防栓/锥桶等近旋转对称物体可减小 yaw 权重，不能把随机朝向当精度问题。

解码契约（尚未实现）：

```text
x = x_min + (col + offset_x) * resolution
y = y_min + (row + offset_y) * resolution
z = center_z
length,width,height = exp(log_size)
yaw = atan2(sin_yaw, cos_yaw)
object = {class_id, confidence, position:[x,y,z], size:[width,length,height], yaw}
```

热图局部峰值/top-K，类别相关阈值，米制距离或 oriented-box NMS。发布带 timestamp/frame 的对象。世界坐标要求检测时刻的 TF；盒子朝向与位置一起变换，不能只改 header。机器人倾斜时完整 3D 朝向不一定能用单 yaw 表示，需要额外姿态约定。

图像 2D head 与 3D box 投影做几何关联，避免把相邻同类目标错配；在匹配且可见时，以验证集校准的 logit 加权或学习式融合得到类别置信度。图像未见、点数太少、两个目标投影重叠时不得强制配对。图像分支可以确认类别，但没有几何支持时输出“未定位”，不能编造 xyz。

## Loss、标注和训练流程（下一阶段）

总 loss 建议：`L_center + λoff L_offset + λz L_z + λsize L_size + λyaw L_yaw + λimg L_image + λsem L_semantic + λocc L_occupancy + λdepth L_depth`。先按量纲归一，再依据验证集梯度/误差调权；不能假设固定权重适合所有类。

- Object：每帧 class、机器人/世界坐标 3D center、l/w/h、yaw、可见性/遮挡与忽略区域；框按完整物体几何标注，不能直接用少量返回点的 extent 当真值。
- Image：对应时间的 2D boxes/类别，细粒度语义可用实例/语义 masks。稀疏 box 标签只适用于检测 head，不能直接当稠密语义监督。
- Semantic BEV：米制 road/sidewalk/grass/building/obstacle 区域和可标注范围；无监督区域 ignore，unknown 类需定义一致。
- Occupancy：明确 z 高度带与二维投影规则；从经过 deskew 的点云 ray tracing 标 free，终点 occupied，遮挡后/未观测 unknown。不能把 count=0 标成 free。
- Calibration：K、D、外参版本、图像尺寸、时间戳、机器人 pose、扫描时间/每点时间；采样日志保留原始标定。

训练流程：按场景/路线/时间段划分 train/val/test，防止相邻帧泄漏；统一类名字典；完成 LiDAR 单分支训练；预训练/训练图像 detector；校验投影后训练 gate 和 3D head；先冻结 backbone 稳定 head，再低学习率联合微调；追加 semantic/occupancy 多任务监督；最后做稠密深度/时序扩展。

RGB、点云和 boxes 做**一致**的几何增强，相机 K/外参同步更新；允许仅图像颜色增强。采用小类 oversampling、类别平衡 heatmap loss、近场高分辨率 crop、hard negatives（杆/消防栓/桶混淆）、点 dropout、标定扰动。不要仅靠增大类别权重解决真实样本缺失。

指标按类别、距离、点数、遮挡、靠墙与相邻目标分组报告：3D/BEV AP、中心距离误差、召回、类别混淆、semantic mIoU、occupancy free/occupied/unknown 混淆；小目标报告 center-distance 指标以补充过于苛刻的 IoU。训练配方、target assignment、loss、解码与训练脚本尚未落地，当前不能运行训练命令。

推理流程（最终）：同步/deskew → 校正与 TF → PFN/voxel → FPN → lifting+mask → gate → 三个 head → 对象解码+联合分类 → 发布带 frame/stamp 的对象与 maps。当前 ROS 只到几何诊断，离线 forward 到原始 head。

## 显存和实时性

单张 500×500×64 FP16 BEV 为 32,000,000 bytes，约 30.5MiB；Camera/LiDAR/Fused 三张约 91.6MiB。0.05m 全范围变成 1000×1000，单张约122.1MiB；0.2m 为250×250，约7.6MiB。FP32 加倍。以上只含三个 feature tensor，未计 concat、gate、encoder 中间激活、FPN、梯度、optimizer、卷积 workspace。

默认门控 concat 是129通道，FP16约61.5MiB；一层500×500、64→64的3×3卷积约9.2G MACs，参考网络的全分辨率多层卷积可能成为主要瓶颈。虽然 gate 比 attention 简单，当前原型并不保证实时。生产版用 depthwise/separable 或稀疏/低分辨率 context + 高分辨率轻 head；ROS NumPy scatter、Python读取点云和逐点 overlay 也需要计时。

以10Hz为初始实验目标，即端到端100ms预算；分别测同步等待、点读取/deskew、backbone、projection、scatter、fusion/head、解码、发布。GPU测量需同步、warmup、统计P50/P95，记录型号/功耗/分辨率/点数/精度模式；不要把模型 forward latency 当整套机器人延迟。通过 AMP、稀疏采样、局部窗口、近场crop优化后，再评估 TensorRT/ONNX 对 scatter/grid_sample 的支持。

## 与原方法的取舍（仅文档比较，不保留实现）

| 项目 | YOLO + ROI/聚类 | 新 BEV 融合 |
|---|---|---|
| 小目标 | 图像类别可能强；少点时定位失败 | 高分辨率+图像辅助可改善，仍受点密度限制 |
| 遮挡 | ROI截断、聚类偏移 | 可学习结构先验；完全遮挡仍无当前证据 |
| 相邻目标 | ROI点混合 | 中心分离可改善；同cell冲突仍存在 |
| 靠墙 | 背景点污染聚类 | 能学习几何/语义区别，但需靠墙标注 |
| 背景污染 | 规则过滤依赖场景 | 联合特征可抑制；投影错误仍注入错误语义 |
| 深度 | 返回点直接测量 | LiDAR辅助仍使用真实几何；无返回不能凭空精确定位 |
| 类别 | 现成2D训练较便利 | 仍保留图像分支，需多类跨模态训练 |
| 3D精度 | 可见点质心不等于物体中心 | 预测中心和完整框；监督偏差会导致系统性错误 |
| 实时 | 通常实现较简单 | 高分辨率密集特征成本较高，必须实测 |
| GPU | 主要2D detector | 额外BEV encoder/fusion/head显存与算力 |
| 标注 | 2D框+少量调参 | 3D框、多任务地图、标定/同步数据更昂贵 |
| 部署 | ROS规则较直接 | 模型/算子/坐标契约复杂度增加 |

当场景简单、类别有限、点密度低、3D标签不足、标定漂移、算力不够时，BEV可能不值得投入；新结构不能保证优于既有系统。虽不在本目录保留旧实现，实验报告仍需用历史录包或既有结果评估是否取得真实改善。

## 分阶段验收

| 阶段 | 产物 / 验收 | 当前实现 |
|---|---|---|
| 1 LiDAR BEV | 点云坐标与cell正确；替换PFN后验证几何 | 统计基线已实现，学习编码待做 |
| 2 Camera | stride4 FPN、小目标2D recall | FPN/语义logits已实现，2D head待做 |
| 3 correspondence | 校正图像投影墙角/路缘吻合，静动态残差统计 | geometry + ROS诊断已实现 |
| 4 fusion | Camera缺失能退化到LiDAR，梯度/消融正确 | gate/scatter已实现，训练待做 |
| 5 detection | target/loss/decoder/训练，按类3D指标 | raw heads已实现，其余待做 |
| 6 small objects | near crop/BEV FPN，远近/少点分组收益 | 设计阶段 |
| 7 semantic/occupancy | 监督与unknown规则、导航地图适配 | raw heads已实现，训练/地图发布待做 |
| 8 temporal | pose warp、过期mask、动态目标ghost评估 | 设计阶段 |

时序阶段使用 `T_current_robot_previous_robot = inv(T_world_current_robot) · T_world_previous_robot` warp 历史 BEV，再用 ConvGRU 或轻门控融合。pose不可用/跳变/长时间断流则清空历史；动态目标需要运动估计或较短历史，不能当静态背景堆叠。二维warp对坡面和大姿态变化有限。

## 可复用模块与来源

优先复用 [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) 的 PointPillars / SECOND / CenterPoint 配置、target assignment、训练框架及自定义数据接口；按实际任务调整 class map、point range、voxel size、head stride，不能直接套汽车配置。[自定义数据指南](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/advanced_guides/customize_dataset.md) 提供数据适配入口。

[MIT BEVFusion](https://github.com/mit-han-lab/bevfusion) 可参考统一BEV与优化pooling；原仓库已归档，复用时核对旧依赖与license，不能把其多相机权重当单目多类现成模型。[CenterPoint 原论文](https://arxiv.org/abs/2006.11275) 提供中心式3D检测路线；其结论不能直接视为本机器人小设施上的性能证据。
