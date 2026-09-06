"""Render the persisted taxonomy audit into a reviewable Markdown report."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from alpha_crowding.data import industry_snapshot_summary


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "audit" / "industry_taxonomy"
PROBE = ROOT / "data" / "raw" / "manifests" / "industry_transition_probe.json"
OUTPUT = ROOT / "docs" / "industry_taxonomy_audit.md"


def main() -> int:
    snapshots = [
        industry_snapshot_summary(pd.read_csv(path))
        for path in sorted(RAW.glob("industry_all_*.csv"))
    ]
    if not snapshots:
        raise RuntimeError("no full-market industry snapshots are available")
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    probe_rows = []
    for item in probe["probes"]:
        labels = [record["industry"] for record in item["records"]]
        coded_share = sum(bool(re.match(r"^[A-Z]\d{2}", label)) for label in labels) / len(labels)
        probe_rows.append((item["requested_date"], coded_share, labels))
    first_coded = next(row for row in probe_rows if row[1] == 1.0)
    prior_uncoded = [row for row in probe_rows if row[0] < first_coded[0] and row[1] == 0.0][-1]

    lines = [
        "# 历史行业分类断点审计",
        "",
        "审计状态：**TRANSITION BRACKETED; RESIDUAL-SYNC IMPACT OPEN**",
        "",
        "本报告来自可重复的BaoStock全市场快照和三只长期上市股票月度探测。它用于冻结工程处理规则，",
        "不构成预测结果，也不关闭全部PIT数据审计。",
        "",
        "## 已定位的断点",
        "",
        f"月度探测中最后一个完全无编码日期为`{prior_uncoded[0]}`，第一个三股均带CSRC代码前缀的日期为`{first_coded[0]}`。",
        "全市场数据还显示2012-06至2012-12原生行业数从257降至104；2012-12已使用新的行业名称层级，",
        "2013-01再加入字母数字代码。因此按两阶段制度切换处理，不能只删除代码前缀后假定分类完全连续。",
        "",
        "## 全市场快照",
        "",
        "| requested | rows | native industries | coded share | stable-map coverage | future updates |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in snapshots:
        lines.append(
            f"| {row['requested_date']} | {row['rows']} | {row['native_industry_count']} | "
            f"{row['coded_prefix_share']:.1%} | {row['stable_mapping_coverage']:.1%} | "
            f"{row['future_update_rows']} |"
        )
    lines.extend(
        [
            "",
            "稳定映射将新体系首字母门类和旧体系中文语义归并为13个粗行业。现有三个全市场截面非空标签",
            "均达到100%映射；原始快照、行数和SHA-256通过运行manifest保存。",
            "",
            "## 实施规则",
            "",
            "1. 主测量使用当日原生PIT行业，保持当期真实组合和placebo在同一分类体系内。",
            "2. 2013-01-01作为分类代际边界；跨边界的HistoricalZ历史窗口不得混用，边界后重新积累至少40个周度观察。",
            "3. 13类稳定粗行业作为强制稳健性口径，检查Residual Sync和模型增量是否依赖行业粒度。",
            "4. 未映射标签、未来`updateDate`或同日重复证券均使该快照审计失败，不能静默归入“其他”。",
            "",
            "## 尚未关闭",
            "",
            "必须在日收益数据接入后量化原生分类与稳定映射对Residual Sync的差异，报告边界前后缺失周数和",
            "确认期M3/M2敏感性。完成前D012保持OPEN。",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
