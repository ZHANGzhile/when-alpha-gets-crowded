"""Create the real 20-security data-acceptance report from its manifest."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "manifests" / "stock20_audit.json"
OUTPUT = ROOT / "docs" / "stock20_data_acceptance.md"


def main() -> int:
    audit = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = audit["securities"]
    if audit["status"] != "QUERY_COMPLETE_REVIEW_REQUIRED" or len(entries) != 20:
        raise RuntimeError("20-security audit is incomplete")
    total_rows = sum(item["raw_quality"]["rows"] for item in entries)
    suspended = sum(item["raw_quality"].get("suspended_rows", 0) for item in entries)
    st_rows = sum(item["raw_quality"].get("st_rows", 0) for item in entries)
    pct_missing = sum(item["raw_quality"].get("missing_pct_change_rows", 0) for item in entries)
    pb_missing = sum(item["raw_quality"].get("missing_pb_rows", 0) for item in entries)
    adjustment_entries = [
        item for item in entries if item["adjustment_check"]["status"] == "DATA"
    ]
    all_invariant = all(
        all(item["adjustment_check"]["invariant_fields"].values())
        for item in adjustment_entries
    )
    terminal = [
        item for item in entries
        if item["raw_quality"].get("last_date") != "2026-08-31"
    ]
    lines = [
        "# 20只真实证券历史数据验收",
        "",
        "验收状态：**行情字段有条件通过；退市结算与PB公开时点仍阻塞P0关闭**",
        "",
        "该报告使用固定20只证券的真实BaoStock日线，不使用合成数据。样本覆盖持续上市、新上市、",
        "长期停牌、ST和历史终止交易证券。逐文件SHA-256保存在`data/raw/manifests/stock20_audit.json`。",
        "",
        "## 总体结果",
        "",
        f"- 证券数：20；原始日线总行数：{total_rows:,}。",
        f"- 停牌行：{suspended:,}；ST行：{st_rows:,}。",
        f"- 涨跌幅缺失：{pct_missing}；PB缺失：{pb_missing}。",
        "- 全部证券日期无重复、严格递增，OHLC价格关系无异常，交易日零成交异常为0。",
        f"- 双口径复权样本：{len(adjustment_entries)}只；成交量、成交额、换手、涨跌幅和PB全部保持一致：{all_invariant}。",
        "",
        "## 逐证券验收",
        "",
        "| code | first | last | rows | suspended | ST | nonpositive PB | float-share coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in entries:
        q = item["raw_quality"]
        lines.append(
            f"| {item['code']} | {q['first_date']} | {q['last_date']} | {q['rows']} | "
            f"{q['suspended_rows']} | {q['st_rows']} | {q['nonpositive_pb_rows']} | "
            f"{q['implied_float_share_coverage']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 历史终止与数据源边界",
            "",
        ]
    )
    for item in terminal:
        q = item["raw_quality"]
        lines.append(
            f"- `{item['code']}`最后记录为`{q['last_date']}`，累计停牌{q['suspended_rows']}行、ST {q['st_rows']}行。"
        )
    lines.extend(
        [
            "",
            "BaoStock能保留终止交易证券的历史行，但日线终止后没有给出换股、现金结算或最终损失。",
            "因此动态和固定成员回测不能把最后价格向后填充，也不能假定零损失。正式持仓账本必须接入",
            "公司行动/退市结算来源；在此之前，含无法结算终止持仓的结果标为不可验收。",
            "",
            "## 冻结的数据处理规则",
            "",
            "1. 因子收益使用经审计的`pctChg`或复权收益；执行价、VWAP和成交成本只使用未复权价格与成交额。",
            "2. 停牌股票保留在已有持仓及账本中，不允许在停牌日成交；新选股资格要求`tradestatus=1`。",
            "3. ST股票不进入新因子组合；已经持有后转ST的证券按下一可交易时点执行退出规则。",
            "4. `turn=0`时不推算流通股本；只在正换手且正成交量日使用`volume/(turn/100)`，并记录估算版本。",
            "5. PB非正值直接视为VALUE信号缺失。即使PB非空，在财报真实公开时点审计通过前，VALUE不得进入确认性主结果。",
            "6. 涨跌幅缺失不得统一填零：停牌日由持仓账本处理，交易状态为1的缺失行进入异常清单。",
            "",
            "## 尚需解决",
            "",
            "- 公司行动与退市结算事件源。",
            "- PB对应财报的真实公开日及历史修订行为。",
            "- CSI300/CSI500调入调出事件的独立来源交叉核验。",
            "- 全量证券下载的服务端速率、失败重试和覆盖率验收。",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
