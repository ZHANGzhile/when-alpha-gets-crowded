"""Stable coarse sectors for auditing historical industry-taxonomy changes."""

from __future__ import annotations

import re

import pandas as pd


# Ordered from specific to broad.  The mapping is deliberately coarse because
# old CSRC labels combine several service industries split by newer taxonomies.
SECTOR_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("finance", ("银行", "保险", "证券", "金融", "货币")),
    ("real_estate", ("房地产",)),
    ("information", ("信息技术", "软件", "互联网", "电信", "信息传输")),
    ("transport", ("交通运输", "仓储", "邮政")),
    ("commerce", ("批发", "零售", "商业经纪")),
    ("construction", ("建筑", "土木工程")),
    ("utilities", ("电力", "燃气", "煤气", "水的生产", "水生产")),
    ("manufacturing", ("制造",)),
    ("extractive", ("采掘", "采矿", "石油和天然气开采")),
    ("primary", ("农、林、牧、渔", "农林牧渔", "农业", "林业", "畜牧", "渔业")),
    ("culture", ("传播与文化", "文化", "体育", "娱乐")),
    ("conglomerate", ("综合",)),
    (
        "broad_services",
        (
            "社会服务",
            "住宿",
            "餐饮",
            "租赁",
            "商务服务",
            "科学研究",
            "技术服务",
            "水利",
            "环境",
            "公共设施",
            "居民服务",
            "修理",
            "教育",
            "卫生",
            "社会工作",
            "公共管理",
        ),
    ),
)

CODED_SECTORS = {
    "A": "primary",
    "B": "extractive",
    "C": "manufacturing",
    "D": "utilities",
    "E": "construction",
    "F": "commerce",
    "G": "transport",
    "H": "broad_services",
    "I": "information",
    "J": "finance",
    "K": "real_estate",
    "L": "broad_services",
    "M": "broad_services",
    "N": "broad_services",
    "O": "broad_services",
    "P": "broad_services",
    "Q": "broad_services",
    "R": "culture",
    "S": "conglomerate",
}


def stable_industry_sector(industry: object) -> str | None:
    """Map an old or new Chinese industry label into a stable coarse sector."""

    if industry is None or pd.isna(industry):
        return None
    label = str(industry).strip()
    if not label:
        return None
    coded = re.match(r"^([A-Z])\d{2}", label.upper())
    if coded:
        return CODED_SECTORS.get(coded.group(1))
    # Newer labels often start with a CSRC letter and two digits.  Removing the
    # prefix lets the same semantic rules operate on both label generations.
    normalized = re.sub(r"^[A-Z]\d{2}", "", label.upper())
    for sector, tokens in SECTOR_RULES:
        if any(token in normalized for token in tokens):
            return sector
    return None


def classify_industry_snapshot(
    frame: pd.DataFrame,
    *,
    industry_col: str = "industry",
) -> pd.DataFrame:
    """Attach taxonomy-generation and stable-sector audit fields."""

    if industry_col not in frame:
        raise KeyError(f"missing industry column: {industry_col!r}")
    result = frame.copy()
    labels = result[industry_col].fillna("").astype(str).str.strip()
    result["taxonomy_has_code_prefix"] = labels.str.match(r"^[A-Z]\d{2}")
    result["stable_sector"] = labels.map(stable_industry_sector)
    return result


def industry_snapshot_summary(
    frame: pd.DataFrame,
    *,
    requested_date_col: str = "requested_date",
    code_col: str = "code",
    update_date_col: str = "updateDate",
) -> dict[str, object]:
    """Validate and summarize one full-market dated industry response."""

    required = {requested_date_col, code_col, update_date_col, "industry"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing industry audit columns: {sorted(missing)}")
    requested_values = pd.to_datetime(frame[requested_date_col], errors="raise").dt.normalize()
    if requested_values.nunique() != 1:
        raise ValueError("one snapshot must contain exactly one requested date")
    if frame[code_col].duplicated().any():
        raise ValueError("industry snapshot contains duplicate security codes")
    update = pd.to_datetime(frame[update_date_col], errors="coerce").dt.normalize()
    requested = requested_values.iloc[0]
    classified = classify_industry_snapshot(frame)
    nonempty = classified["industry"].fillna("").astype(str).str.strip().ne("")
    denominator = int(nonempty.sum())
    mapped = int(classified.loc[nonempty, "stable_sector"].notna().sum())
    return {
        "requested_date": requested.date().isoformat(),
        "rows": int(len(frame)),
        "unique_codes": int(frame[code_col].nunique()),
        "nonempty_industry": denominator,
        "coded_prefix_share": float(
            classified.loc[nonempty, "taxonomy_has_code_prefix"].mean()
        ) if denominator else None,
        "stable_mapping_coverage": float(mapped / denominator) if denominator else None,
        "future_update_rows": int((update > requested).fillna(False).sum()),
        "missing_update_rows": int(update.isna().sum()),
        "native_industry_count": int(classified.loc[nonempty, "industry"].nunique()),
        "stable_sector_count": int(classified.loc[nonempty, "stable_sector"].nunique()),
        "unmapped_labels": sorted(
            classified.loc[nonempty & classified["stable_sector"].isna(), "industry"]
            .astype(str)
            .unique()
            .tolist()
        ),
    }
