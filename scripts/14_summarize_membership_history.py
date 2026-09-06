"""Create the production PIT universe acceptance report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "weekly_membership.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "membership_download.json"
OUTPUT = ROOT / "docs" / "membership_history_acceptance.md"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETE":
        raise RuntimeError("membership download is not complete")
    frame = pd.read_parquet(DATA)
    frame["requested_date"] = pd.to_datetime(frame["requested_date"]).dt.normalize()
    frame["updateDate"] = pd.to_datetime(frame["updateDate"]).dt.normalize()
    frame["update_lag_days"] = (frame["requested_date"] - frame["updateDate"]).dt.days
    if (frame["update_lag_days"] < 0).any():
        raise ValueError("membership contains future updateDate values")
    expected_sizes = {"CSI300": 300, "CSI500": 500}
    snapshot_sizes = frame.groupby(["requested_date", "index"]).size()
    for index, expected in expected_sizes.items():
        observed = snapshot_sizes.xs(index, level="index")
        if not observed.eq(expected).all():
            raise ValueError(f"{index} has a snapshot with size other than {expected}")
    duplicate_keys = int(frame.duplicated(["requested_date", "index", "code"]).sum())
    cross_index_overlap = int(
        (frame.groupby(["requested_date", "code"])["index"].nunique() > 1).sum()
    )
    if duplicate_keys or cross_index_overlap:
        raise ValueError("membership contains duplicate keys or cross-index overlap")
    changes = []
    for index, group in frame.groupby("index", sort=True):
        snapshots = {
            date: set(rows["code"])
            for date, rows in group.groupby("requested_date", sort=True)
        }
        prior_date = None
        prior = None
        for date, current in snapshots.items():
            if prior is not None and current != prior:
                changes.append(
                    {
                        "index": index,
                        "previous_week": prior_date.date().isoformat(),
                        "effective_week": date.date().isoformat(),
                        "added": len(current - prior),
                        "removed": len(prior - current),
                    }
                )
            prior_date, prior = date, current
    lines = [
        "# CSI300/CSI500周度PIT成员历史验收",
        "",
        "验收状态：**COMPLETE — weekly point-in-time membership dataset**",
        "",
        f"- 周度决策日：{frame['requested_date'].nunique()}。",
        f"- 成员行：{len(frame):,}；历史唯一证券：{frame['code'].nunique()}。",
        f"- 日期范围：{frame['requested_date'].min().date()}至{frame['requested_date'].max().date()}。",
        f"- 实际成员变化周：{len(changes)}个指数周事件。",
        f"- 来源更新日滞后：中位数{frame['update_lag_days'].median():.0f}天，最大{frame['update_lag_days'].max()}天；未来更新日0行。",
        f"- 重复成员键：{duplicate_keys}；同日跨指数重叠：{cross_index_overlap}。",
        f"- BaoStock合并换股时点缺口经官方指数公告修复：{int(frame['membership_source'].eq('official_index_announcement').sum())}行。",
        "- 每个截面CSI300恰好300只、CSI500恰好500只。",
        "",
        "## 逐年覆盖",
        "",
        "| year | decision weeks | unique securities | change events |",
        "| ---: | ---: | ---: | ---: |",
    ]
    annual = frame.assign(year=frame["requested_date"].dt.year).groupby("year").agg(
        decision_weeks=("requested_date", "nunique"),
        unique_securities=("code", "nunique"),
    )
    change_frame = pd.DataFrame(changes)
    if not change_frame.empty:
        change_frame["year"] = pd.to_datetime(change_frame["effective_week"]).dt.year
        annual_changes = change_frame.groupby("year").size()
    else:
        annual_changes = pd.Series(dtype=int)
    for year, row in annual.iterrows():
        lines.append(
            f"| {year} | {int(row['decision_weeks'])} | "
            f"{int(row['unique_securities'])} | {int(annual_changes.get(year, 0))} |"
        )
    lines.extend(
        [
        "",
        "## 检测到的成员变化周",
        "",
        "| index | previous week | effective week | added | removed |",
        "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for change in changes:
        lines.append(
            f"| {change['index']} | {change['previous_week']} | {change['effective_week']} | "
            f"{change['added']} | {change['removed']} |"
        )
    lines.extend(
        [
            "",
            "数据集保存每个周末交易日的实际响应，而不是使用当前成分回填。逐截面SHA-256和合并文件",
            "SHA-256保存在`data/raw/manifests/membership_download.json`。2019年中国外运吸收合并外运发展、",
            "2021年中国能建吸收合并葛洲坝期间，BaoStock分别提前删除旧代码、延后显示继承代码；5个",
            "周度缺口使用事前已发布的上交所/中证指数联合公告修复，逐行保留来源URL、公告可用日和生效日。",
            "其他定期调样仍需独立公告源交叉核验后才能关闭P0。",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
