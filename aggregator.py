#!/usr/bin/env python3
"""
aggregator.py — 批量审计「万象渊鉴·场域诊断测试集」47 个 case

================================================================================
一、职责
================================================================================
1. 遍历输入目录（或单个文件）中的所有 case；
2. 复用 trajectory_v3_ultra.py 的解析与 U/D/A/H 评分逻辑，逐 case 流式读取；
3. 输出交付用金标准 CSV：gold_standard_47_cases.csv
   列：turn_id, session.id, gen_ai.input.messages, gen_ai.output.messages,
       gen_ai.usage.input_tokens, gen_ai.usage.output_tokens,
       gen_ai.usage.total_tokens, U, D, A, H, case_id, group
4. 输出诊断汇总 CSV：diagnosis_summary.csv
   列：case_id, case_name, group, turn_count, point_count, a_max,
       a_flip_threshold, flip_triggered, u_end, u_end_diff_vs_control,
       control_case, cumul_a_peak, d_volatility_mean, h_min, h_max

================================================================================
二、U / D / A / H 与去重逻辑
================================================================================
定义与去重逻辑完全来自 trajectory_v3_ultra.py：
- U：关键词所在句子的平均语义一致性（字符 3-gram 余弦）；
- D：关键词首现字符位置归一化后的标准差 * 2.5（上限 1.0）；
- A：0.6 * 对抗标记句占比 + 0.4 * min(1, 互斥短语命中数 * 0.3)；
- H：0.4*U + 0.4*D - 0.2*A；
- 每个 assistant turn 只基于本轮增量（本轮 user + 本轮 assistant）计分，
  不拼接历史；完全重复或空的 turn 会被跳过并计数。

================================================================================
三、用法
================================================================================
    python aggregator.py --input_dir 万象渊鉴·场域诊断测试集.txt --char_step 10 \
        --output_gold gold_standard_47_cases.csv \
        --output_diag diagnosis_summary.csv
"""

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# 允许从任意工作目录直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trajectory_v3_ultra import (  # noqa: E402
    DEFAULT_CHAR_STEP,
    DEFAULT_MIN_CHARS,
    DEFAULT_TOP_TERMS,
    TURN_COLUMNS,
    Case,
    iter_case_rows,
    load_cases,
)

GOLD_COLUMNS = TURN_COLUMNS + ["case_id", "group"]

DIAG_COLUMNS = [
    "case_id", "case_name", "group", "turn_count", "point_count",
    "a_max", "a_flip_threshold", "flip_triggered", "u_end",
    "u_end_diff_vs_control", "control_case",
    "cumul_a_peak", "d_volatility_mean", "h_min", "h_max",
]

INPUT_EXTENSIONS = (".json", ".txt")


# ---------------------------------------------------------------------------
# 输入收集
# ---------------------------------------------------------------------------
def collect_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    for root, _dirs, names in os.walk(path):
        for name in sorted(names):
            if name.lower().endswith(INPUT_EXTENSIONS):
                files.append(os.path.join(root, name))
    return sorted(files)


def load_all_cases(path: str) -> List[Case]:
    cases: List[Case] = []
    for file_path in collect_files(path):
        try:
            cases.extend(load_cases(file_path))
        except Exception as exc:  # 单个文件损坏不应中断整批
            print(f"[skip] {file_path}: {exc}", file=sys.stderr)
    return cases


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _empty_case_stats() -> Dict[str, float]:
    return {
        "turn_count": 0,
        "point_count": 0,
        "a_max": 0.0,
        "cumul_a_peak": 0.0,
        "dvol_sum": 0.0,
        "dvol_count": 0,
        "h_min": 1.0,
        "h_max": 0.0,
        "u_end": 0.0,
    }


def _pair_benchmark_cases(stats: Dict[str, Dict[str, float]]) -> None:
    """BENCH-01..10 与 BENCH-11..20 配对，计算 U 末值差。"""
    bench: Dict[int, str] = {}
    for case_id in stats:
        match = re.fullmatch(r"BENCH-(\d+)", case_id)
        if match:
            bench[int(match.group(1))] = case_id

    for number, case_id in bench.items():
        partner_number = number + 10 if number <= 10 else number - 10
        partner = bench.get(partner_number)
        if not partner:
            continue
        subject = stats[case_id]
        control = stats[partner]
        subject["control_case"] = partner
        subject["u_end_diff_vs_control"] = round(
            subject["u_end"] - control["u_end"], 4
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量审计 47 个 case 并生成金标准 / 诊断汇总 CSV"
    )
    parser.add_argument("--input_dir", required=True,
                        help="测试集目录或单个测试集文件")
    parser.add_argument("--char_step", type=int, default=DEFAULT_CHAR_STEP,
                        help="超细步长，默认 10 字符")
    parser.add_argument("--min_chars", type=int, default=DEFAULT_MIN_CHARS,
                        help="起始采样字符下限，默认 1")
    parser.add_argument("--top_terms", type=int, default=DEFAULT_TOP_TERMS,
                        help="每轮关键词数量，默认 10")
    parser.add_argument("--a_threshold", type=float, default=0.3,
                        help="A 值 flip 判定阈值，默认 0.3")
    parser.add_argument("--output_gold", default="gold_standard_47_cases.csv",
                        help="金标准 CSV（每轮计分）")
    parser.add_argument("--output_diag", default="diagnosis_summary.csv",
                        help="诊断汇总 CSV")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.input_dir):
        print(f"input not found: {args.input_dir}", file=sys.stderr)
        return 1

    cases = load_all_cases(args.input_dir)
    if not cases:
        print(f"no cases parsed from {args.input_dir}", file=sys.stderr)
        return 1

    totals = {
        "turns_scored": 0,
        "duplicate_history_skipped": 0,
        "empty_turns_skipped": 0,
        "trace_points": 0,
    }
    stats: Dict[str, Dict[str, float]] = {}
    case_names: Dict[str, Tuple[str, str]] = {}

    with open(args.output_gold, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOLD_COLUMNS)
        writer.writeheader()

        for case in cases:
            case_stats = _empty_case_stats()
            case_stats.update({
                "a_flip_threshold": args.a_threshold,
                "control_case": "",
                "u_end_diff_vs_control": "",
            })
            case_names[case.case_id] = (case.name, case.group)

            for kind, row in iter_case_rows(
                case,
                char_step=args.char_step,
                min_chars=args.min_chars,
                top_terms=args.top_terms,
                stats=totals,
            ):
                if kind == "trace":
                    case_stats["point_count"] += 1
                    case_stats["a_max"] = max(case_stats["a_max"], float(row["A"]))
                    case_stats["cumul_a_peak"] = max(
                        case_stats["cumul_a_peak"], float(row["cumul_A"])
                    )
                    case_stats["dvol_sum"] += float(row["D_volatility"])
                    case_stats["dvol_count"] += 1
                    case_stats["h_min"] = min(case_stats["h_min"], float(row["H"]))
                    case_stats["h_max"] = max(case_stats["h_max"], float(row["H"]))
                else:
                    record = dict(row)
                    record["case_id"] = case.case_id
                    record["group"] = case.group
                    writer.writerow(record)
                    case_stats["turn_count"] += 1
                    case_stats["u_end"] = float(row["U"])

            stats[case.case_id] = case_stats
            print(
                f"{case.case_id:<10} {case.name[:22]:<24} "
                f"turns={int(case_stats['turn_count']):<3} "
                f"points={int(case_stats['point_count']):<5} "
                f"a_max={case_stats['a_max']:.4f} "
                f"u_end={case_stats['u_end']:.4f}"
            )

    _pair_benchmark_cases(stats)

    with open(args.output_diag, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAG_COLUMNS)
        writer.writeheader()
        for case in cases:
            case_stats = stats[case.case_id]
            name, group = case_names[case.case_id]
            writer.writerow({
                "case_id": case.case_id,
                "case_name": name,
                "group": group,
                "turn_count": int(case_stats["turn_count"]),
                "point_count": int(case_stats["point_count"]),
                "a_max": round(case_stats["a_max"], 4),
                "a_flip_threshold": args.a_threshold,
                "flip_triggered": "true" if case_stats["a_max"] > args.a_threshold else "false",
                "u_end": round(case_stats["u_end"], 4),
                "u_end_diff_vs_control": case_stats.get("u_end_diff_vs_control", ""),
                "control_case": case_stats.get("control_case", ""),
                "cumul_a_peak": round(case_stats["cumul_a_peak"], 4),
                "d_volatility_mean": round(
                    case_stats["dvol_sum"] / case_stats["dvol_count"]
                    if case_stats["dvol_count"] else 0.0,
                    4,
                ),
                "h_min": round(case_stats["h_min"], 4) if case_stats["h_min"] <= 1 else "",
                "h_max": round(case_stats["h_max"], 4),
            })

    print(
        f"cases={len(cases)} turns_scored={totals['turns_scored']} "
        f"duplicate_history_skipped={totals['duplicate_history_skipped']} "
        f"empty_turns_skipped={totals['empty_turns_skipped']} "
        f"trace_points={totals['trace_points']}"
    )
    print(f"wrote {args.output_gold} and {args.output_diag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
