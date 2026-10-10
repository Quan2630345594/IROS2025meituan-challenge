# IROS2025 The 3rd Low-Altitude Economy Intelligent Flight Management Challenge

指令解析模块包含结构化推理、数据构建、LoRA 训练入口与评估工具

## 目录结构

```text
catkin_ws/scripts/language_parse/
                      ├── parse_goal_direction.py   结构化 JSON 推理
                      ├── contract.py / schema.json 词表、输出契约与校验
                      ├── system_prompt.txt         解析规则
                      ├── build_dataset.py          数据构建与分组划分
                      ├── check_dataset.py          数据一致性检查
                      ├── train_lora.py / training/  LoRA 训练入口与配置
                      ├── evaluate.py               模型对比与离线评分
                      └── data/ / tests/            数据与回归测试

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