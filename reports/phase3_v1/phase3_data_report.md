# CTN 第三阶段：数据契约与模拟世界 V2 验收（phase3_v1）

## 结果摘要

- 数据版本：`v2.1`；固定种子：`20261102`。
- 样本规模：train 50,000、validation 10,000、test 10,000；记录全是成对事实/反事实。
- 测试集同时含常规和预留未见组合样本：OOD 标记样本 4,998 / 10,000。
- 单字段直接机制干预：21/21 项符合预期。
- 数据文件 SHA256：见 `phase3_data_hashes.json`。

## 对验收问题的回答

1. **三个世界均具备显式机制。** Mechanics 的 signed force、摩擦和固定约束计算速度/位移变化；Thermal 的热输入、环境交换、热容和冻结阈值决定温度与物态；Weather 的有效降雨、遮挡、排水及湿度裁剪决定湿度变化。
2. **每字段的作用**已写入 `phase3_feature_schema.json` 和模板文件；数值、布尔、类别有不同明确编码。Mechanics 的 position 是有效状态值但对位移增量不产生影响；其整体平移不变性被作为无效干预测试。
3. **有效因素**：Mechanics X(force/direction/duration)、C(friction/fixed)、S(mass/velocity)；Thermal X(heating/cooling/duration)、C(ambient_temperature)、S(initial_temperature/initial_phase)，freezing_point 只作用 final_phase；Weather X(rainfall/rainfall_duration)、C(roof_cover/drainage)、S(ground_wetness)。
4. **条件性因素**：Mechanics fixed=true 时力不再产生位移；Thermal freezing_point 只在最终温度位于阈值两侧时改变 phase；Weather 初始湿度仅在上下界裁剪条件下改变 wetness_delta。
5. **无效干预**：零降雨时改变 roof_cover；Mechanics 只平移 position 不改变 delta。两者都有自动测试。热学 freezing_point 对温度增量是无效干预，但可改变最终物态。
6. **编码器保留信息**：字段顺序固定，连续值按 train-only 均值/标准差编码，bool 编为 0/1，类别 one-hot 并保留 `__MISSING__`/`__UNK__`；缺失数值用训练均值填补。每字段保留原值到编码位置的 trace。目标归一化按 domain/target 单独拟合，仅 numeric target；final_phase 保留类别标签。
7. **泄漏检查**：pair_id 在 split 内；生成逻辑整对分配后再打乱，pair/world ID 带 split 命名空间。Y_raw/target_state 仅为监督和状态结果，预测输入契约仅包含 X/C/S 编码和 domain。
8. **配对严格性**：同一对只改变 intervention 指定的一个 X/C/S 字段；其余 raw 输入、外部变量与 seed 固定。若干预的是状态变量，state_before 仅该被干预字段相应变化。
9. **未见组合**：测试集一半为训练支持范围内的常规组合，一半使用独立 OOD 条件族：mechanics 高摩擦/力组合、thermal 新冻结点、weather 新降雨/遮挡/排水组合。具体分布和标签在统计 JSON 中。
10. **复现**：`python -m ctn.v2.cli --output data/v2 --seed 20261102` 完全确定性；三个 split、模板、因果图、schema、统计文件均写 SHA256。标准化器只在 train rows 上拟合，再可从保存 schema 恢复用于 val/test。

## 简化假设与边界

热学采用常数环境交换系数和“冰热容是液态两倍”的简化规则；不模拟潜热、沸腾、压力、多物质。力学是一维显式更新，不模拟碰撞和非线性摩擦。天气使用线性有效降雨与排水后将湿度裁剪到 `[0,1]`。这些是明确的程序因果机制，不是现实物理的精密模型。

本阶段没有训练任何模型、没有使用测试集调参，也未更改 V1 数据和旧实验报告。测试结果见提交记录/终端，自动化测试文件为 `tests/test_v2_*.py`。
