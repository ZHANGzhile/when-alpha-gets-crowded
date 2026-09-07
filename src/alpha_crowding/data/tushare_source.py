"""Narrow Tushare adapter for controller-only reference data."""

from __future__ import annotations

import os

import pandas as pd


class TushareQueryError(RuntimeError):
    """Raised when a required Tushare response is missing or malformed."""


def create_tushare_client(token: str | None = None):
    """Create a Pro client without logging or persisting the credential."""

    credential = token or os.environ.get("TUSHARE_TOKEN")
    if not credential:
        raise RuntimeError("set TUSHARE_TOKEN before downloading controller data")
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError(
            "tushare is not installed; install the project's controller-data extra"
        ) from exc
    return ts.pro_api(credential)


def tushare_code_to_internal(code: str) -> str:
    """Convert ``600000.SH``/``000001.SZ`` to the repository code convention."""

    value = str(code).strip().upper()
    try:
        symbol, exchange = value.split(".")
    except ValueError as exc:
        raise ValueError(f"invalid Tushare security code: {code!r}") from exc
    prefixes = {"SH": "sh", "SZ": "sz", "BJ": "bj"}
    if exchange not in prefixes or not symbol.isdigit() or len(symbol) != 6:
        raise ValueError(f"invalid Tushare security code: {code!r}")
    return f"{prefixes[exchange]}.{symbol}"


def fetch_csi800_weights(
    client,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch one bounded month of CSI800 constituent weights."""

    frame = client.index_weight(
        index_code="000906.SH",
        start_date=start_date,
        end_date=end_date,
    )
    if not isinstance(frame, pd.DataFrame):
        raise TushareQueryError("index_weight did not return a DataFrame")
    required = ["index_code", "con_code", "trade_date", "weight"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise TushareQueryError(f"index_weight lacks fields: {sorted(missing)}")
    result = frame[required].copy()
    if not result["index_code"].astype(str).eq("000906.SH").all():
        raise TushareQueryError("index_weight returned another index")
    result["weight_date"] = pd.to_datetime(
        result.pop("trade_date"), format="%Y%m%d", errors="raise"
    ).dt.normalize()
    result["code"] = result.pop("con_code").map(tushare_code_to_internal)
    result["weight"] = pd.to_numeric(result["weight"], errors="raise") / 100.0
    result = result.drop(columns="index_code")
    if result.duplicated(["weight_date", "code"]).any():
        raise TushareQueryError("index_weight contains duplicate date/security rows")
    if result[["weight_date", "code", "weight"]].isna().any().any():
        raise TushareQueryError("index_weight contains missing required values")
    return result.sort_values(["weight_date", "code"]).reset_index(drop=True)


def fetch_daily_stock_limits(client, *, trade_date: str) -> pd.DataFrame:
    """Fetch the official pre-open upper and lower price limits for one session."""

    frame = client.stk_limit(trade_date=trade_date)
    if not isinstance(frame, pd.DataFrame):
        raise TushareQueryError("stk_limit did not return a DataFrame")
    required = ["trade_date", "ts_code", "up_limit", "down_limit"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise TushareQueryError(f"stk_limit lacks fields: {sorted(missing)}")
    result = frame[required].copy()
    result["date"] = pd.to_datetime(
        result.pop("trade_date"), format="%Y%m%d", errors="raise"
    ).dt.normalize()
    result["code"] = result.pop("ts_code").map(tushare_code_to_internal)
    for column in ("up_limit", "down_limit"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result.duplicated(["date", "code"]).any():
        raise TushareQueryError("stk_limit contains duplicate date/security rows")
    invalid = (
        result[["up_limit", "down_limit"]].isna().any(axis=1)
        | (result["down_limit"] <= 0.0)
        | (result["up_limit"] <= result["down_limit"])
    )
    if invalid.any():
        raise TushareQueryError("stk_limit contains invalid price bounds")
    return result.sort_values(["date", "code"]).reset_index(drop=True)
