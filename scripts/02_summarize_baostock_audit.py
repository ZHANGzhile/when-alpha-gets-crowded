"""Turn the bounded BaoStock probe into a human-reviewable capability report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data" / "raw" / "audit"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "baostock_capability_audit.json"
OUTPUT = ROOT / "docs" / "data_capability_audit.md"


def _yes(value: bool) -> str:
    return "PASS" if bool(value) else "FAIL"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    constituent_rows = []
    membership_sets: dict[tuple[str, str], set[str]] = {}
    for path in sorted(AUDIT_DIR.glob("constituents_*.csv")):
        frame = pd.read_csv(path, dtype=str)
        index = frame["index"].iat[0]
        requested = frame["requested_date"].iat[0]
        expected = 300 if index == "CSI300" else 500
        unique_codes = frame["code"].nunique()
        update_dates = pd.to_datetime(frame["updateDate"], errors="coerce")
        request_date = pd.Timestamp(requested)
        membership_sets[(index, requested)] = set(frame["code"])
        constituent_rows.append(
            {
                "index": index,
                "requested": requested,
                "source_update": ",".join(sorted(frame["updateDate"].dropna().unique())),
                "rows": len(frame),
                "unique": unique_codes,
                "count_check": _yes(len(frame) == expected and unique_codes == expected),
                "asof_check": _yes(update_dates.notna().all() and (update_dates <= request_date).all()),
            }
        )

    overlap_rows = []
    dates = sorted({date for _, date in membership_sets})
    for date in dates:
        csi300 = membership_sets.get(("CSI300", date), set())
        csi500 = membership_sets.get(("CSI500", date), set())
        overlap_rows.append({"date": date, "intersection": len(csi300 & csi500), "union": len(csi300 | csi500)})

    turnover_rows = []
    for index in ("CSI300", "CSI500"):
        index_dates = sorted(date for key, date in membership_sets if key == index)
        for earlier, later in zip(index_dates, index_dates[1:]):
            old = membership_sets[(index, earlier)]
            new = membership_sets[(index, later)]
            turnover_rows.append(
                {
                    "index": index,
                    "interval": f"{earlier} → {later}",
                    "added": len(new - old),
                    "removed": len(old - new),
                }
            )

    industry_rows = []
    for path in sorted(AUDIT_DIR.glob("industry_*.csv")):
        frame = pd.read_csv(path, dtype=str)
        for row in frame.to_dict("records"):
            update = pd.Timestamp(row["updateDate"])
            requested = pd.Timestamp(row["requested_date"])
            industry_rows.append(
                {
                    "code": row["code"],
                    "requested": row["requested_date"],
                    "source_update": row["updateDate"],
                    "industry": row["industry"],
                    "asof_check": _yes(update <= requested),
                }
            )

    bars2 = pd.read_csv(AUDIT_DIR / "bars_sh_600000_2017-05_adjust2.csv")
    bars3 = pd.read_csv(AUDIT_DIR / "bars_sh_600000_2017-05_adjust3.csv")
    bars = bars2.merge(bars3, on=["date", "code"], suffixes=("_adj2", "_raw"))
    invariant_columns = ["volume", "amount", "turn", "tradestatus", "pctChg", "pbMRQ", "isST"]
    invariant_checks = {
        column: bool(
            np.allclose(
                pd.to_numeric(bars[f"{column}_adj2"], errors="coerce"),
                pd.to_numeric(bars[f"{column}_raw"], errors="coerce"),
                equal_nan=True,
            )
        )
        for column in invariant_columns
    }
    raw_vwap = bars["amount_raw"] / bars["volume_raw"]
    vwap_inside_bar = bool(((raw_vwap >= bars["low_raw"]) & (raw_vwap <= bars["high_raw"])).all())
    ratio = bars["close_adj2"] / bars["close_raw"]
    corporate_action_ratio_changed = bool(ratio.nunique() > 1)

    critical_checks = [
        all(row["count_check"] == "PASS" for row in constituent_rows),
        all(row["asof_check"] == "PASS" for row in constituent_rows),
        all(row["asof_check"] == "PASS" for row in industry_rows),
        all(invariant_checks.values()),
        vwap_inside_bar,
        corporate_action_ratio_changed,
    ]
    bounded_status = "PASS" if all(critical_checks) else "FAIL"

    def table(rows: list[dict[str, object]]) -> str:
        if not rows:
            return "_No rows._"
        headers = list(rows[0])
        rendered = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
            rendered.append("| " + " | ".join(values) + " |")
        return "\n".join(rendered)

    text = f"""# BaoStock 数据能力审计（第一轮小样本）

审计状态：**BOUNDED PROBE {bounded_status}; FULL PIT AUDIT OPEN**  
接口版本：`{manifest.get('baostock_version', 'unknown')}`  
查询完成时间：`{manifest.get('finished_at_utc', 'unknown')}`

本报告只说明有限样本的接口行为，不构成完整数据源验收。原始响应的路径、行数与SHA-256保存在
`data/raw/manifests/baostock_capability_audit.json`；CSV位于被Git忽略的`data/raw/audit/`。

## 历史指数成分

{table(constituent_rows)}

同日CSI300/CSI500集合关系：

{table(overlap_rows)}

跨样本日期确有成员变化，说明接口没有简单返回当前成分：

{table(turnover_rows)}

当前判断：行数、唯一性和`updateDate <= requested_date`均需通过；仍要对照独立来源抽查具体调入/调出事件，
并覆盖指数调整日前后相邻交易日，之后才能关闭PIT成分审计。

## 历史行业

{table(industry_rows)}

样本显示历史查询会返回不晚于请求日的更新日期，但2012年的行业名称与后期编码体系不同。这是明显的
分类口径断点。同期真实组合与placebo仍可使用当日分类，但HistoricalZ可能在分类制度切换时发生机械跳变。
正式实现必须定位全部分类口径切换日期，并比较“当日原生分类”与稳定映射/切断窗口的稳健性。

## 行情、复权和单位

- adjustflag=2与不复权数据的成交量、成交额、换手率、涨跌幅、PB和交易状态一致：
  `{json.dumps(invariant_checks, ensure_ascii=False)}`。
- `amount / volume`得到的VWAP位于不复权日内最低价和最高价之间：`{_yes(vwap_inside_bar)}`。
- 2017-05-25公司行为前后，前复权/不复权价格比发生变化：`{_yes(corporate_action_ratio_changed)}`。

实施含义：收益序列可以使用经核验的复权价格或BaoStock涨跌幅；成交额、VWAP、流通股数估计和执行模拟
必须配合不复权价格。`volume`与`turn`可用于`float_shares=volume/(turn/100)`，但还需跨更多证券和公司行为验证。

## 尚未关闭的问题

1. 历史成分的外部事件级交叉核验。
2. 行业分类体系切换的全市场日期与稳定映射。
3. PB是否严格按财报真实公开日进入历史日线；在此之前VALUE不进入确认性主研究。
4. 退市股票、长期停牌、ST切换、上市日和异常零成交覆盖。
5. 20只股票的复权、换手和流通股估算抽查。
6. 全量抓取速度、重试、断点续传与服务端限速。

## 放行决定

本轮允许继续开发数据适配器和小样本工程流程；**不允许冻结最终研究协议，也不允许开始确认性结果分析**。
只有上列问题按预设验收规则完成后，P0数据能力审计才可关闭。
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"bounded_status={bounded_status}")
    return 0 if bounded_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
