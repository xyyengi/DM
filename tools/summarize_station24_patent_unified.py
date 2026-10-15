"""Write the one concise handoff report required by the patent comparison."""
import argparse
import json
from pathlib import Path


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--raw-run", required=True)
    p.add_argument("--train-run", required=True)
    return p.parse_args()


def row(path: Path):
    m = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return [
        m["station_average"]["wind"]["crps"],
        m["station_average"]["solar_daylight"]["crps"],
        m["aggregate_mw"]["renewable"]["crps"],
        m["joint"]["energy_score_pu"],
        m["aggregate_mw"]["renewable"]["coverage_90"],
        m["aggregate_mw"]["renewable"]["width_90"],
        m["joint"]["spatial_corr_rmse_all_pairs"],
        m["joint"]["spatial_corr_rmse_wind_solar"],
    ]


def main():
    a = args(); root = Path(a.root)
    items = [
        ("Raw500", root / "raw_n500_seed424242"),
        ("Raw400+Tail100", root / "mixture_raw400_tail100_n500"),
        ("Raw450+Tail50", root / "mixture_raw450_tail50_n500"),
    ]
    lines = [
        "# RUN SUMMARY", "",
        "- 选择依据：同一validation seed=424242下，canonical body_tail_moe纯主体与后续sampler_es_l1纯主体参数相同；后者仅旧Tail参数变化。为保持Lightweight初始化语义，复用canonical checkpoint。",
        f"- 主体checkpoint/run：`{a.raw_run}`；生成显式使用raw state并将旧成员级Tail路由概率设为0。",
        f"- Lightweight训练run：`{a.train_run}`；Raw参数和buffer冻结证据见preflight及posttrain_freeze_audit。",
        "- 数据：23个validation窗口；generation seed=424242；Test未使用；Energy Score统一使用80成员。",
        "- 成员规则：20%使用Raw前400条+全部Tail100条；10%使用Raw前450条+Tail前50条。规则在评价前固定，不读取事件结果。",
        "- 服务器任务：完整训练、生成、合并、评价和归档均已执行完毕。",
        "", "| 方案 | Wind CRPS | Solar daylight CRPS | Renewable CRPS | Energy Score | 90% coverage | 90% width | spatial RMSE | wind-solar RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, path in items:
        values = row(path)
        lines.append("| " + name + " | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += [
        "", "事件onset/duration/depth与member-hit详见两个`evaluation_*/continuous_event_evaluation`目录及各自`RESULT_SUMMARY.md`。",
        "", "## 重要限制", "",
        "仅有4个冻结独立validation事件；本结果不证明跨数据集泛化，也未进行第二轮调参或更换generation seed。",
    ]
    (root / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
