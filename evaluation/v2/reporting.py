from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


MAIN = ["XY", "MLP-Full", "Transformer-Full", "CTN"]
DOMAINS = ["mechanics", "thermal", "weather"]
ABLAT = ["CTN-no-C", "CTN-no-S", "CTN-no-XC", "CTN-no-XS", "CTN-no-CS", "CTN-no-transition", "CTN-no-pair-loss"]


def _mean_sd_ci(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return {"mean": None, "std": None, "ci95": None, "n": 0, "per_seed": []}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values))
    return {"mean": mean, "std": sd, "ci95": [mean-half, mean+half], "n": len(values), "per_seed": values}


def _nested_metrics(records, metric_key):
    out = {}
    for track in ("A", "B"):
        out[track] = {}
        for model in MAIN:
            selected = [r for r in records if r["track"] == track and r["model"] == model]
            out[track][model] = {"seeds": [r["seed"] for r in selected], "domains": {}}
            for domain in DOMAINS:
                names = sorted({target for r in selected for target in r[metric_key].get(domain, {})})
                domain_result = {}
                for target in names:
                    acc = [r[metric_key][domain][target].get("accuracy") for r in selected if target in r[metric_key].get(domain, {})]
                    if any(x is not None for x in acc):
                        domain_result[target] = {"accuracy": _mean_sd_ci(acc), "confusion_matrices": [r[metric_key][domain][target]["confusion_matrix"] for r in selected if target in r[metric_key].get(domain, {})]}
                        continue
                    vals = [r[metric_key][domain][target].get("standardized_mse") for r in selected if target in r[metric_key].get(domain, {})]
                    maes = [r[metric_key][domain][target].get("raw_mae") for r in selected if target in r[metric_key].get(domain, {})]
                    if vals:
                        domain_result[target] = {"standardized_mse": _mean_sd_ci(vals), "raw_mae": _mean_sd_ci(maes)}
                out[track][model]["domains"][domain] = domain_result
    return out


def build_report(run_dir="reports/phase4/v1"):
    root = Path(run_dir)
    curves_path = root / "training_curves.json"
    run = json.loads(curves_path.read_text(encoding="utf-8"))
    records = run["records"]

    comparison = {
        "data_version": run["data_version"],
        "seeds": run["seeds"],
        "metrics": {
            "test": _nested_metrics(records, "test_metrics"),
            "ood_condition": _nested_metrics(records, "ood_metrics"),
            "numeric_range_outside_train": _nested_metrics(records, "out_of_range_metrics"),
        },
        "parameter_counts": {m: sorted({r["parameters"] for r in records if r["model"] == m}) for m in MAIN},
        "parameters_within_10_percent": max(max(r["parameters"] for r in records if r["model"] == m) for m in MAIN[1:]) / min(min(r["parameters"] for r in records if r["model"] == m) for m in MAIN[1:]) - 1 < 0.10,
    }
    (root / "model_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    cf_out = {}
    for track in ("A", "B"):
        cf_out[track] = {}
        for model in MAIN:
            cf_out[track][model] = {}
            chosen = [r for r in records if r["track"] == track and r["model"] == model]
            for domain in DOMAINS:
                numeric = [r["counterfactual_metrics"][domain] for r in chosen]
                targets = sorted({k for x in numeric for k in x["targets"]})
                cf_out[track][model][domain] = {
                    "direction_accuracy": _mean_sd_ci([x["direction_accuracy"] for x in numeric]),
                    "direction_pairs_per_seed": [x["direction_pairs"] for x in numeric],
                    "invalid_intervention_preservation_rate": _mean_sd_ci([x["invalid_intervention_preservation_rate"] for x in numeric]),
                    "invalid_pair_count_per_seed": [x["invalid_pair_count"] for x in numeric],
                    "targets": {},
                }
                for target in targets:
                    vals = [x["targets"][target] for x in numeric if target in x["targets"]]
                    cf_out[track][model][domain]["targets"][target] = {
                        "change_mae": _mean_sd_ci([v["change_mae"] for v in vals]),
                        "mean_abs_error_before": _mean_sd_ci([v["mean_abs_error_before"] for v in vals]),
                        "mean_abs_error_after": _mean_sd_ci([v["mean_abs_error_after"] for v in vals]),
                    }
    (root / "counterfactual_evaluation.json").write_text(json.dumps(cf_out, ensure_ascii=False, indent=2), encoding="utf-8")

    ablation = {}
    for model in ABLAT:
        ablation[model] = {}
        selected = [r for r in records if r["track"] == "A" and r["model"] == model]
        for domain in DOMAINS:
            ablation[model][domain] = {}
            target_names = sorted({n for r in selected for n in r["test_metrics"].get(domain, {})})
            for target in target_names:
                entries = [r["test_metrics"][domain][target] for r in selected if target in r["test_metrics"].get(domain, {})]
                if entries and "standardized_mse" in entries[0]:
                    ablation[model][domain][target] = {"standardized_mse": _mean_sd_ci([x["standardized_mse"] for x in entries]), "raw_mae": _mean_sd_ci([x["raw_mae"] for x in entries])}
                elif entries and "accuracy" in entries[0]:
                    ablation[model][domain][target] = {"accuracy": _mean_sd_ci([x["accuracy"] for x in entries]), "confusion_matrices": [x["confusion_matrix"] for x in entries]}
    (root / "ablation_results.json").write_text(json.dumps(ablation, ensure_ascii=False, indent=2), encoding="utf-8")

    target_rows = []
    for domain in DOMAINS:
        for target in sorted({name for r in records for name in r["test_metrics"].get(domain, {})}):
            target_rows.append((domain, target))
    lines = [
        "# CTN Phase 4 因果关系网络与公平对照实验报告（版本 v1）", "",
        f"- 数据：{run['data_version']}；官方数据生成种子 20261102。", f"- 训练设备：{run['device']}；随机种子：{', '.join(map(str, run['seeds']))}。",
        f"- 运行数：{len(records)}（主模型：40；消融：35）。训练配置：AdamW，lr={run['learning_rate']}，每 batch {run['batch_pairs']} 对，最多 {run['max_epochs']} epoch，验证集早停 patience={run['early_stopping_patience']}。", "",
        "## 方法与完整性", "",
        "模型输入只有 X/C/S 编码和 domain。Y 与 target_state 仅用于监督和评估；CTN 的状态转移头以 target_state 监督训练，推理时只接收 X/C/S/domain。普通监督轨道 A 不加配对损失；配对监督轨道 B 对所有架构采用同一配对变化损失。测试集仅在训练结束后用于汇报。",
        "V2.2 数据文件未重新生成。数据哈希校验详见 reproducibility.json。参数量比较只包括 MLP-Full、Transformer-Full、CTN；XY 是历史基线。模型关系门控和激活值仅记录为内部信号，不解释为真实因果效应。", "",
        "## 常规测试集：每领域每目标", "", "下表报告 Track A（监督）/Track B（配对监督）的跨种子标准化 MSE 均值 ± 标准差。分类目标报告准确率。",
        "", "| 领域 | 目标 | XY A/B | MLP-Full A/B | Transformer-Full A/B | CTN A/B |", "|---|---|---:|---:|---:|---:|"
    ]
    def fmt(model, track, domain, target, metric):
        obj=comparison['metrics']['test'][track][model]['domains'][domain].get(target,{}).get(metric,{})
        if obj.get('mean') is None: return '—'
        return f"{obj['mean']:.4f} ± {obj['std']:.4f}"
    for domain,target in target_rows:
        metric="accuracy" if target=="final_phase" else "standardized_mse"
        cells=[domain,target]
        for model in MAIN:
            cells.append(f"{fmt(model,'A',domain,target,metric)} / {fmt(model,'B',domain,target,metric)}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## OOD 与反事实", "",
        "OOD 组合误差按样本的 ood_condition 分组；数值范围外推按输入是否超出训练范围分组。两类结果按领域和目标保存在 model_comparison.json，不能将小样本 OOD 子集的波动视为稳定结论。",
        "反事实方向准确率仅统计真实变化非零的配对；无效干预保持率独立计算，当前阈值为预测结果的原始单位绝对变化不超过 0.05。变化量 MAE 及配对数按目标、领域、轨道、种子记录于 counterfactual_evaluation.json。", "",
        "## 训练诊断", "",
        "每个 epoch 的训练损失、按目标损失、验证损失和梯度范数均记录在 training_curves.json。最佳 epoch 按验证损失选取；若训练损失继续下降而验证损失在最佳 epoch 后回升，视为过拟合迹象。"
    ]
    overfit=[]; grad_bad=0
    for r in records:
        losses=r['epochs']; grad_bad += sum(e.get('nonfinite_gradient_steps',0) for e in losses)
        if len(losses)>1 and r['best_epoch']<losses[-1]['epoch'] and losses[-1]['train_loss']<losses[r['best_epoch']-1]['train_loss']:
            overfit.append(f"{r['track']}/{r['model']}/seed{r['seed']}")
    resumed_count=sum(bool(r.get("resumed_from_checkpoint")) for r in records)
    lines += [f"", f"检测到非有限梯度步数：{grad_bad}。符合上述过拟合迹象的运行：{', '.join(overfit) if overfit else '无（仅针对有逐 epoch 历史的运行）'}。因意外关机后从 checkpoint 恢复的运行数：{resumed_count}；这些运行的原逐 epoch 训练/验证曲线未写盘，无法恢复，因此其 epochs 为空，未伪造曲线。", "",
        "## 必答问题", "",
        "1. CTN 是否优于 MLP-Full / Transformer-Full？查看常规测试和 OOD 的逐目标均值、跨种子标准差与 CI；若结果未在领域与种子间一致，不宣称总体优越。",
        "2. CTN 是否有 OOD 优势？以 ood_condition 子集逐领域评估；范围外推另列。训练数据规模、条件组合覆盖和 OOD 子集样本数限制结论。",
        "3. CTN 的反事实响应如何？方向准确率排除零变化对；变化量误差和无效干预保持率分开报告。配对监督轨道不能与架构贡献混为一谈。",
        "4. C 与 S 是否贡献？看 CTN-no-C、CTN-no-S 与 CTN 的逐领域消融差值；这是当前训练协议下的预测贡献，不是现实因果效应证明。",
        "5. 去掉关系交互/状态转移后是否下降？由 CTN-no-XC、no-XS、no-CS、no-transition 分别检查；以五种子方差衡量稳定性。",
        "6. 配对监督是否帮助所有模型？比较 Track B 与 Track A 同架构结果。no-pair-loss 对照对应 Track A CTN。",
        "7. 当前证据是否支持 CTN 有独立架构价值？需同时满足与相近参数量 MLP/Transformer 的稳定测试优势、OOD/反事实增益和消融证据；不得根据单一 MSE 或内部门控权重下结论。", "",
        "## 限制与下一步", "",
        "本实验使用确定性模拟器，所学响应受模拟机制和数据支持范围约束；不能由模拟表现直接推出真实世界因果有效性。数值范围外推组按输入字段训练区间定义，不代表所有组合级 OOD。当前配对损失是统一的预测变化监督，权重固定为 1；未做权重敏感性搜索。消融可能改变有效容量，即使主模型参数量匹配，也应视为针对模块的诊断而非完美容量控制。",
        "下一轮应优先审查各领域 OOD /反事实失败样例，扩大并独立设计未见机制组合，预注册配对损失系数与调参预算，再用真实或外部模拟机制验证。", "",
        "## 产物", "",
        "- model_comparison.json：常规/OOD/范围外推逐目标跨种子统计。", "- counterfactual_evaluation.json：反事实指标。", "- ablation_results.json：消融指标。", "- training_curves.json：epoch 训练诊断和完整逐次记录。", "- reproducibility.json：数据版本、哈希、运行配置。", "- predictions/：每次运行的测试预测 JSONL。", "- checkpoints/：每次运行的最佳验证集 checkpoint。", ""
    ]
    (root / "phase4_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"runs": len(records), "report": str(root / "phase4_experiment_report.md")}


if __name__ == "__main__":
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))
