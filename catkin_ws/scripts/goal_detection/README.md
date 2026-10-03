# 单目 RGB + LiDAR BEV 特征融合

本目录只包含新方案，不包含原 YOLO + ROI 点云聚类、导航决策或速度控制代码。原 `meituan/ws2/src/lty` 不变。核心路线是 **LiDAR 建立米制几何网格，Camera 注入高分辨率语义，BEV 统一空间表达**。

当前是按阶段实施的研究原型，并非已经训练完成的多类别检测器：

| 模块 | 当前状态 |
|---|---|
| LiDAR → BEV | 已实现计数/高度 pillar 统计；尚非学习式 PointPillars PFN |
| 单目特征 | 已实现轻量 CNN + P2/P3/P4 FPN；随机初始化 |
| 标定对应 | 已实现 optical 相机投影、前方/FOV 过滤、像素深度筛选、BEV scatter |
| 特征融合 | 已实现可学习门控残差融合，带 Camera 有效掩码 |
| Object / Semantic / Occupancy | 已实现原始预测头；尚无训练、目标分配、loss、解码或权重 |
| 图像语义分支 | 已实现 P2 类别 logits；完整 2D box head 和联合评分待实现 |
| ROS | 同步传感器、TF、去畸变及几何诊断；未接入学习式推理 |
| 时序 | 仅设计，尚未实现 |

`launch/decision.launch` 名称沿用启动入口，但现在只启动新方案的几何验证节点，**不会输出可用的 3D 检测或控制机器人**。三类学习输出必须经过标注、训练和独立验证才能使用。详细架构、loss、训练计划与资源估算见 [docs/design.md](docs/design.md)。

## 目录

```text
bev/geometry.py           米制栅格、刚体变换、相机对应
bev/model.py              FPN、稀疏特征提升、门控融合、三个任务头
bev/cli.py                离线预处理 / 未训练网络前向验证
config/bev.json           离线网格、类别及张量契约
scripts/bev_features_node.py  ROS1 几何诊断
launch/decision.launch    新方案启动入口
tests/                    几何与融合验证
```

## 离线使用（Ubuntu）

进入本目录后执行：

```bash
python3 -m pip install numpy
python3 -m unittest discover -s tests -v
```

安装与你的 Python / CUDA 环境相匹配的 PyTorch 后，才能验证网络前向；依赖列表为 `requirements-bev.txt`。ROS Noetic 常用 Python 3.8，与较新 PyTorch 的 Python 支持需要分别确认，建议离线研究与 ROS 诊断使用独立环境。

输入 `frame.npz` 必须含以下数组，禁止使用 object 数组：

| 键 | 格式 |
|---|---|
| `image_rgb` | 去畸变后的 `H×W×3` RGB uint8，当前不做 letterbox |
| `points_lidar` | `N×3` 或 `N×4`，单位米；前三列 xyz |
| `K` | 与去畸变图像对应的 `3×3` 内参 |
| `T_camera_lidar` | `4×4`，LiDAR → camera optical |
| `T_robot_lidar` | `4×4`，LiDAR → 机器人 |

```bash
python3 -m bev.cli --input frame.npz --output outputs/features.npz
python3 -m bev.cli --input frame.npz --output outputs/raw_heads.npz --forward
```

第一条产生 pillar 特征与对应索引；第二条额外产生**随机初始化**网络的 raw logits / box regression，只用于形状和数据流检查，不代表检测结果。默认大网格网络前向较慢，CPU 验证可另建配置，将 x/y 范围缩小到各 8m；不要改变真实数据的坐标单位。

## ROS1 使用（Linux / ROS Noetic）

此目录放在 `catkin_ws/scripts`，catkin 不会自动把它当作 `src` 包。从仓库根目录创建相对软链接：

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

启动传感器或带 `/tf`、`/tf_static`、CameraInfo 的 rosbag；播放时可先设置 `rosparam set use_sim_time true`，再 `rosbag play --clock your.bag`。要求：

- 图像为 `sensor_msgs/CompressedImage`；点云为 `PointCloud2`，含 xyz。
- CameraInfo 与图像尺寸一致，`header.frame_id` 为同一个 **optical frame**；提供实测 K/D。本节点接收原始畸变图像，以 K 为新内参去畸变，仅支持 `plumb_bob`。已校正图像应提供对应 K 且 D=0，避免重复去畸变。
- TF 连通 `fixed_frame → robot_frame → lidar_frame / camera_optical_frame`。节点用双时间 TF lookup 补偿帧间自车运动；这不能代替 LiDAR 扫描内逐点 deskew。
- 时间戳非零、同一时间基准；`sync_slop` 默认 50ms 是启动配置，不是可接受标定误差的保证。高速运动应缩小并评估残差。

诊断话题（`sensor_msgs/Image`）：

| 话题 | 含义 |
|---|---|
| `/bev_features_node/pillar_log_count` | 每格 log(1+点数) |
| `/bev_features_node/pillar_mean_height` | 机器人坐标系平均 z，米 |
| `/bev_features_node/camera_observed_mask` | 有投影对应的格子为 255；无对应为 0 |
| `/bev_features_node/projection_overlay` | 校正图像叠加 LiDAR 投影 |

可用 `rqt_image_view` 查看。BEV 图像行沿 y、列沿 x 增加，原点与分辨率见配置；它不是 RGB 鸟瞰图，也不是 OccupancyGrid。空格不能解释为 free。ROS 的网格范围目前固定为 x=[0,50)、y=[-25,25)，`resolution` 可调；`config/bev.json` 只供离线 CLI 使用。

没有输出 `/detected_positions`，因为目前尚无训练完成的对象。新方案完成训练后，需另写带时间戳/frame 的 3D object 发布器及旧决策接口适配，不能直接把 raw head 当位置。

## 验证及已知限制

几何测试覆盖变换方向、边界、NaN、空点云、相机后方、同像素遮挡。PyTorch 测试覆盖无 Camera 有效区域、散射聚合、梯度与输出形状；缺少 torch 时会明确 skip。当前机器没有 PyTorch 或 ROS，网络和 ROS 实际运行尚未验证。

单相机仅覆盖视野内有可靠几何对应的区域；稀疏 lifting 对没有 LiDAR 返回的小物体无能为力。像素 z-buffer 只排除同像素更远的点，不解决所有遮挡。几何统计尚未滤地面、按高度分层或学习点编码，所有这些属于后续阶段，不能以新架构名称代替实测收益。
