# Ctn因果网络混合模型

**Causal Token Network（CTN）：因果词元网络**

项目覆盖规划中的第一、第二阶段：可重复模拟、结构化事件/X-Y-C-S 数据、干预反事实、模型对照与消融评估。模拟实验只检验模型在这些合成环境上的学习和泛化，不证明真实语言中的因果发现能力。

## 功能

- Thermal、mechanics、weather 三个可 reset/step/observe/intervene 的简化模拟器。
- Pydantic 事件模板、事件实例、世界状态、因果关系和因果样本结构。
- 保留 X/Y/C/S 原始特征；独立、确定性 32 维哈希编码器用于接口原型，不代表已学到因果知识。
- 固定随机种子的 JSONL 训练/验证/测试生成以及质量报告。
- JSON 因果图、领域事件模板和 pytest 验收测试。
- 30 个事件模板、包含五类关系的因果图、严格配对的事实/反事实数据。
- XY、XYC、XYS、XYCS 与 Transformer 五种模型配置，多随机种子、留出组合和长链评估。

## 安装与运行

需要 Python 3.11+。

```powershell
cd D:\CTNMODEL\ctn_testcode
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m ctn.cli --output data/generated --seed 42
pytest
```

安装研究依赖并运行三随机种子对照实验：

```powershell
python -m pip install -e '.[research]'
python -m ctn.experiment_cli --data data/generated --output results --epochs 20 --seeds 11 23 37
python -m ctn.audit
```

也可自定义规模：`python -m ctn.cli --train 5000 --validation 1000 --test 1000 --seed 42`。

## 数据

每条样本包含 `sample_id/domain/cause/X_raw/Y_raw/C_raw/S_raw/state_before/target_state/target_event/relation_label/intervention/pair_id/split/event`。JSONL 生成路径为 `data/generated/`。同一 `pair_id` 有事实和反事实两条记录，复用相同随机外生参数，只切换一个指定条件；不同 split 使用分离的场景编号。

## 模拟假设

- 热学：温度按 `10 × (heating_power - cooling_power) × dt` 更新；低于或等于 0°C 为冰，否则为液态；省略沸腾/潜热/压力。
- 力学：一维显式更新，`a=(force-friction×velocity)/mass`；固定物体速度归零且不位移。
- 天气：湿度增量 `rainfall × (1-roof_cover) - drainage`，裁剪到 `[0,1]`。

这些规则是可检查的玩具环境，不是精确物理模型。每个世界的 `intervene` 允许对状态变量执行外科式设定；`step(..., intervention=True)` 只标记事件来源，不改变动力学方程。

## 目录

```text
ctn/                 数据模型、编码器、模拟器、数据生成CLI
data/schemas/        JSON Schema
data/templates/      三个领域的事件模板
data/causal_graph/   因果节点与边
data/generated/      生成的 JSONL 与质量报告
configs/             默认生成参数
tests/               验收测试
```

## 当前范围与限制

每个数据 split 只含完整配对样本；测试数据留出训练中未见过的力-摩擦、初始温度和降雨-排水组合。训练脚本比较 XY、XYC、XYS、XYCS 与 Transformer，记录训练/验证/测试损失、分领域误差、配对反事实一致性、长链误差、参数量及耗时。各模型参数量保持在约 18k–23k。XY 系列消融有意改变可见输入；Transformer 与 XYCS 接收相同的 X/C/S 信息。

完整实验结果与解释见 `results/REPORT.md` 和 `results/experiment.json`。这些分数只适用于本项目定义的合成结构化目标，不能作为自然语言能力或现实因果发现的证据。

第二阶段机制与数据诊断结果见 `reports/v1/phase2_audit.md`。每次运行 `python -m ctn.audit` 会自动创建 `reports/vN/` 新版本目录，包含总报告、特征干预、配对反事实、数据质量和逐 epoch 训练诊断文件，不覆盖旧版。审计阶段只记录模型输入、逐 epoch 训练/验证指标与干预预测，不修改模型架构。

第一阶段的随机哈希编码有碰撞且非可学习编码，只用于验证形状和数据通路。当前实验模型直接读取保留的结构化数值特征。
