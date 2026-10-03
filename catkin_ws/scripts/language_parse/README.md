# 结构化导航指令解析与 Qwen LoRA-SFT

本目录实现“方向—物体”的有序多步序列。默认仍调用 Ollama 的 `qwen2.5vl:7b`，使用 JSON Schema、temperature=0 和一次请求；生成后再检查字段与状态。仅提供结构化 JSON 接口，不再保留旧算法、旧提示词或字符串兼容接口。

## 标注约定

方向只有 `front/back/left/right`，斜向报 `unsupported_direction`。`go straight` 等表达统一映射；物体仍保留原来的八类。允许的物体别名集中在 `contract.py`，移除了 timber/bucket/seat/lorry/signboard 等可能跨类别的映射。宽泛中文“桶”同样不直接归为 barrel。

| 指令 | 标注 |
| --- | --- |
| Go straight to the tree | front/tree |
| Front to tree, turn right, then go straight to hydrant | front/tree → right/fire hydrant |
| Walk forward and turn left at tree, then continue to traffic cone | front/tree → left/traffic cone |
| Go right to bench, then go straight to tree | right/bench → front/tree |
| Go to tree / 去树那里 | needs_clarification，missing_direction |
| The bench is in front of the tree | needs_clarification，no_action |
| Go to the bench in front of the tree | needs_clarification，missing_direction |
| Turn left, no, right to tree | right/tree |
| Do not turn left; go right to tree | right/tree |

`at X` 是前一段移动的终点和下一转向的起点，`to X` 是该段目标。转向后、到达目标前的直行属于该转向步骤；到达目标后的新直行段才是新的 front。无接近方向的“在树处左转然后去交通锥”保留 left/traffic cone，但对接近树的过程报 missing_direction。

忽略礼貌、速度和短暂停顿。位置关系、未确定的条件分支、否定与纠正不能作为噪声删除。缺字段、未知物体、未解决的选择及真正歧义使用 issues，完整规则在 `system_prompt.txt`。

`target_gap_m` 表示最终位置与该目标物体的间隔，统一米；没有明确间隔就输出 null。50 cm=0.5 m，2 ft=0.6096 m，0 合法。“向前走2米到树”中的路程不属于目标间隔。间隔单位缺失、冲突或负值须澄清，省略受影响步骤。

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

complete 必须至少有一步，且 issues 为空。needs_clarification 必须有 issues，可保留完全确定的步骤供展示，不能直接执行。所有实际步骤都必须完整，不能用 null 代替缺失方向或目标。Schema 限制字段形状，程序检查状态一致性；语义、指令顺序和间隔是否有原文依据仍须用标注测试集评估。

## Ubuntu 推理与程序接入

解析、数据处理和离线测试只依赖 Python 标准库（Python 3.8+）。启动已安装的 Ollama，并准备原模型后，从仓库根目录在 Ubuntu Bash 运行：

```bash
cd catkin_ws/scripts/language_parse
python3 parse_goal_direction.py '向前走到树那里，右转后直走到消防栓'
python3 parse_goal_direction.py 'Go straight to the tree, turn right to the fire hydrant'
```

`--model`、`LANGUAGE_PARSE_MODEL` 可切换模型；`--base-url`、`OLLAMA_BASE_URL` 可配置 Ollama 地址，兼容原来的 `http://localhost:11434/v1/`。默认返回结构化 JSON，待澄清退出码为2，网络或格式失败为1。不会在错误时返回猜测结果，也不会重复投票。

```python
from parse_goal_direction import parse_instruction

result = parse_instruction("向前走到距离树50厘米的位置")
if result["status"] == "complete":
    steps = result["steps"]
else:
    issues = result["issues"]
```

Ollama 使用原生 `/api/chat` 的 `format` Schema。[Ollama API 文档](https://docs.ollama.com/api/chat)

## 数据与复现

数据包含 111 条通过规则筛选的旧种子，另外24条因大小写重复、过宽物体映射或凭空补 front 而隔离到 `data/seed_quarantine.jsonl`。补充中英文方向×物体组合，以及 at/to、转向后直行、否定、纠正、缺项、空间关系、单位换算、条件分支和歧义样本，去重后共241条。

| 集合 | 条数 | 改写族数 |
| --- | ---: | ---: |
| train | 185 | 95 |
| validation | 23 | 12 |
| test | 33 | 15 |

同一有序方向—物体路线（含中英文改写、不同间隔）归入同一族；待澄清案例按显式改写族分组。固定随机种子42，约80/10/10按族划分，并将罕见问题码的族保留在训练集。6条未见表达专门留在测试集，其整个族不进入训练。8条运行时 few-shot 只取自 train。

`*.records.jsonl` 保存 ID、族、标签和来源，用于审查与评估；`train/validation/test.jsonl` 只含 ms-swift 的 system/user/assistant messages，assistant 只有规范 JSON。数据标签由规则筛选和手写模板生成，尚未做人工全量语义审查；开始正式训练前应审查隔离记录、边界案例和保留种子。这里提供的是小规模起始数据，不能据此声称泛化效果。

```bash
# 用仓库已保存的种子重建，无需旧脚本和模型
python3 build_dataset.py
python3 check_dataset.py
python3 -m unittest discover -s tests -v

```

检查器验证词表/Schema、所有标签、SFT目标一致性、族隔离、few-shot来源和训练类别覆盖。修改别名或系统规则后须重新 build，防止推理提示与训练样本不一致。

## LoRA 训练

第一轮基座为 `Qwen/Qwen2.5-VL-7B-Instruct`，须核对它与原 Ollama 模型的来源和 revision；Ollama 模型包不能直接作为 ms-swift 训练权重。默认 r=8、alpha=32、学习率1e-4、最多3轮、all-linear，仅训练语言部分 LoRA，冻结视觉模块与对齐模块。每轮验证/保存，以验证 loss 选择最佳 checkpoint，连续1次未改善则早停。`loss_scale=last_round` 只监督最终 assistant JSON。

配置在 `training/lora_config.json`。当前机器未检测到 swift/Ollama/NVIDIA 命令；未下载模型、安装训练栈或实际训练。下面在单独的 Linux/CUDA 环境执行，依赖固定为 ms-swift 3.5.3 与 transformers 4.53.3，CUDA/PyTorch需按训练机环境配置；这个训练栈尚未在本机运行验证。

```bash
cd catkin_ws/scripts/language_parse
python3 -m pip install -r requirements-training.txt
python3 check_dataset.py
python3 train_lora.py             # 只打印经过检查的参数列表
python3 train_lora.py --run       # 实际训练；也可 --model models/base
```

`training/early_stop.py` 通过 ms-swift 3.5 的 extra_callbacks 注册 Transformers EarlyStoppingCallback。测试集不送入训练器。不要通过测试集挑轮数或调规则。

[Qwen ms-swift SFT 示例](https://qwen.readthedocs.io/en/v2.5/training/SFT/ms_swift.html)、[ms-swift 参数](https://swift.readthedocs.io/en/v3.5/Instruction/Command-line-parameters.html)、[数据格式](https://swift.readthedocs.io/en/v3.5/Customization/Custom-dataset.html)

## 部署训练结果

可先用 ms-swift 合并适配器，再用支持 Qwen2.5-VL 的 vLLM 服务合并权重。这条路径避免把 safetensors 适配器误当作 Ollama 已可加载的模型。用训练日志记录的 best checkpoint 替换下面相对路径，merge目录以 export 命令实际输出为准：

```bash
swift export --adapters output/qwen2.5vl-navigation/checkpoint-best --merge_lora true
vllm serve output/qwen2.5vl-navigation/merged --served-model-name navigation-lora --max-model-len 8192
```

解析器的 `--backend openai` 接入该本地服务，通过标准 `response_format=json_schema` 约束。服务端必须支持 Schema，否则请求报错，不会静默退回自由文本。vLLM在单独环境安装，避免与训练依赖冲突；当前只做了模拟服务响应测试，真实服务待训练机验证。

```bash
python3 parse_goal_direction.py '向前走到树那里' --backend openai --base-url http://localhost:8000/v1 --model navigation-lora --no-few-shot
```

Ollama部署取决于已安装版本对 Qwen2.5-VL权重/量化/适配器的支持，确认导入成功后可直接通过 `--model` 指定新名称。不要假定 `FROM qwen2.5vl:7b` 加 `ADAPTER` 就能加载本方案的适配器。

[vLLM Schema 输出](https://docs.vllm.ai/en/stable/features/structured_outputs/)

## 两组评估

```bash
python3 evaluate.py --mode schema --model qwen2.5vl:7b
python3 evaluate.py --mode lora-schema --backend openai --model navigation-lora --base-url http://localhost:8000/v1
```

schema 与 lora-schema 使用同一份规则、few-shot 和 Schema。比较 LoRA 增益时应统一后端、基座来源与精度；研究去掉 few-shot 的效果时，两组均加 `--no-few-shot`。

报告保存在忽略的 `reports/`：完整序列准确率、按位置的方向/物体/间隔准确率、显式间隔准确率、步骤数与顺序、status、问题码、待澄清precision/recall，以及中英文/未见表达子集。缺失或多出步骤均扣分；请求失败也纳入分母。issues的message措辞不要求逐字匹配，比较问题码。

```bash
# 对已经收集的模型输出离线评分，JSONL行需有id/output或id/error
python3 evaluate.py --predictions reports/schema.predictions.jsonl --mode schema
```

离线回归测试，覆盖模拟HTTP传输、格式边界、状态、数据隔离及计分正确性。模拟服务和“用标准标签检查计分器”的测试不能作为模型准确率；两组实际模型评估及LoRA训练尚未执行。


所有命令中的文件路径相对于当前工作目录；训练子进程固定在本模块目录运行，默认数据、插件和输出参数均使用相对路径。模型名称可使用 Hugging Face 标识；本地权重路径按启动命令时的目录解释。各代码块单独执行时先从仓库根目录进入本模块。
