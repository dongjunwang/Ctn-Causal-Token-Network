# Ctn因果网络混合模型

**Causal Token Network（CTN）：因果词元网络**

第一阶段原型：建立可重复的模拟世界、结构化事件/X-Y-C-S 样本，以及干预数据生成基础。本阶段不训练语言模型，也不声称模拟实验证明了真实语言中的因果发现能力。

## 功能

- Thermal、mechanics、weather 三个可 reset/step/observe/intervene 的简化模拟器。
- Pydantic 事件模板、事件实例、世界状态、因果关系和因果样本结构。
- 保留 X/Y/C/S 原始特征；独立、确定性 32 维哈希编码器用于接口原型，不代表已学到因果知识。
- 固定随机种子的 JSONL 训练/验证/测试生成以及质量报告。
- JSON 因果图、领域事件模板和 pytest 验收测试。

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

第一阶段提供用于后续建模的编码接口，不包含训练器、Transformer 对照模型或统计因果发现。随机哈希编码有碰撞且非可学习编码，只用于验证形状和数据通路。
