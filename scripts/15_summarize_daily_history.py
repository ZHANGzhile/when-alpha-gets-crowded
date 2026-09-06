"""Create an auditable acceptance report for the complete daily-history universe."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_crowding.data import daily_history_acceptance_failures


ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "data" / "interim" / "weekly_membership.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "daily_download.json"
WORKERS = ROOT / "data" / "raw" / "manifests" / "daily"
OUTPUT = ROOT / "docs" / "daily_history_acceptance.md"
END_DATE = pd.Timestamp("2026-08-31")


def main() -> int:
    run = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if run["status"] != "COMPLETE":
        raise RuntimeError("daily-history download is not complete")
    membership = pd.read_parquet(MEMBERSHIP, columns=["code"])
    codes = sorted(membership["code"].dropna().astype(str).unique())
    if len(codes) != run["universe_codes"]:
        raise ValueError("daily run did not cover the complete PIT membership universe")
    entries = [
        json.loads(
            (WORKERS / f"{code.replace('.', '_')}.json").read_text(encoding="utf-8")
        )
        for code in codes
    ]
    if len(entries) != run["universe_codes"]:
        raise ValueError(
            f"worker manifests={len(entries)} but universe codes={run['universe_codes']}"
        )
    rejected = [
        (entry["code"], daily_history_acceptance_failures(entry["quality"]))
        for entry in entries
        if daily_history_acceptance_failures(entry["quality"])
    ]
    if rejected:
        raise ValueError(f"daily histories fail hard acceptance: {rejected[:10]}")

    quality = pd.DataFrame(
        [{"code": entry["code"], **entry["quality"]} for entry in entries]
    )
    quality["first_date"] = pd.to_datetime(quality["first_date"])
    quality["last_date"] = pd.to_datetime(quality["last_date"])
    terminal = quality[quality["last_date"] < END_DATE - pd.Timedelta(days=30)].copy()
    terminal = terminal.sort_values(["last_date", "code"])
    total_rows = int(quality["rows"].sum())
    pb_observed = total_rows - int(quality["missing_pb_rows"].sum())
    float_rows = int(quality["implied_float_share_rows"].sum())
    lines = [
        "# CSI800历史成员日行情验收",
        "",
        "验收状态：**COMPLETE — universe-wide raw daily history**",
        "",
        f"- 历史唯一证券：{len(quality):,}；日记录：{total_rows:,}。",
        f"- 总体日期范围：{quality['first_date'].min().date()}至{quality['last_date'].max().date()}。",
        f"- 停牌记录：{int(quality['suspended_rows'].sum()):,}；ST记录：{int(quality['st_rows'].sum()):,}。",
        f"- PB非缺失覆盖：{pb_observed / total_rows:.2%}；可推算流通股本覆盖：{float_rows / total_rows:.2%}。",
        "- 重复日期、日期乱序、成交日零量、OHLC次序异常：均为0，否则本报告拒绝生成。",
        f"- 在数据截止日前超过30天终止的证券历史：{len(terminal)}只；这些证券进入退市/合并结算审计，不能向后填充价格。",
        f"- 证券集合SHA-256：`{run['universe_codes_sha256']}`。",
        "",
        "## 终止历史清单",
        "",
        "| code | first date | last date | rows | suspended | ST |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in terminal.itertuples(index=False):
        lines.append(
            f"| {row.code} | {row.first_date.date()} | {row.last_date.date()} | "
            f"{row.rows} | {row.suspended_rows} | {row.st_rows} |"
        )
    lines.extend(
        [
            "",
            "本报告确认的是原始行情的完整抓取和硬完整性规则。PB历史发布时间、指数成员的独立来源",
            "交叉核验，以及终止证券的现金/换股结算仍分别由P0审计关闭；在关闭前不会把VALUE",
            "纳入确认性模型，也不会把终止价格静默当成零收益。",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
