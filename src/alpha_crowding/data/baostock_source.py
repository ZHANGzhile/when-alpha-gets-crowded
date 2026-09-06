"""Small, auditable BaoStock adapter used by the data capability audit."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd


DAILY_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
)

INDEX_DAILY_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "pctChg",
)


class BaoStockQueryError(RuntimeError):
    """Raised when BaoStock returns a non-success status."""


def _load_baostock():
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError(
            "baostock is not installed; install project dependencies before data audit"
        ) from exc
    return bs


@contextmanager
def session() -> Iterator[object]:
    """Log in once and guarantee logout even when a query fails."""

    bs = _load_baostock()
    login = bs.login()
    if getattr(login, "error_code", None) != "0":
        raise BaoStockQueryError(
            f"BaoStock login failed: {getattr(login, 'error_code', None)} "
            f"{getattr(login, 'error_msg', '')}"
        )
    try:
        yield bs
    finally:
        bs.logout()


def result_to_frame(result: object, *, query_name: str) -> pd.DataFrame:
    """Fully materialize a BaoStock ResultData and reject protocol errors."""

    code = getattr(result, "error_code", None)
    message = getattr(result, "error_msg", "")
    if code != "0":
        raise BaoStockQueryError(f"{query_name} failed: {code} {message}")
    fields = list(getattr(result, "fields", []) or [])
    if not fields:
        raise BaoStockQueryError(f"{query_name} returned no schema")
    rows: list[list[str]] = []
    while result.next():
        row = list(result.get_row_data())
        if len(row) != len(fields):
            raise BaoStockQueryError(
                f"{query_name} returned {len(row)} values for {len(fields)} fields"
            )
        rows.append(row)
    terminal_code = getattr(result, "error_code", "0")
    if terminal_code != "0":
        raise BaoStockQueryError(
            f"{query_name} pagination failed: {terminal_code} "
            f"{getattr(result, 'error_msg', '')}"
        )
    return pd.DataFrame(rows, columns=fields)


def fetch_constituents(bs: object, *, index: str, date: str) -> pd.DataFrame:
    """Fetch a dated CSI300 or CSI500 constituent snapshot."""

    key = index.upper()
    functions = {
        "CSI300": bs.query_hs300_stocks,
        "CSI500": bs.query_zz500_stocks,
    }
    if key not in functions:
        raise ValueError("index must be CSI300 or CSI500")
    frame = result_to_frame(functions[key](date), query_name=f"{key} constituents {date}")
    frame.insert(0, "requested_date", date)
    frame.insert(0, "index", key)
    return frame


def fetch_trade_calendar(
    bs: object, *, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch the exchange calendar including non-trading calendar days."""

    frame = result_to_frame(
        bs.query_trade_dates(start_date=start_date, end_date=end_date),
        query_name=f"trade calendar {start_date}:{end_date}",
    )
    return frame


def fetch_industry(bs: object, *, date: str, code: str = "") -> pd.DataFrame:
    """Fetch dated industry data for one code or the entire response."""

    frame = result_to_frame(
        bs.query_stock_industry(code=code, date=date),
        query_name=f"industry {code or 'ALL'} {date}",
    )
    frame.insert(0, "requested_date", date)
    return frame


def fetch_daily_bars(
    bs: object,
    *,
    code: str,
    start_date: str,
    end_date: str,
    adjustflag: str,
) -> pd.DataFrame:
    """Fetch one daily bar slice with explicit adjustment convention."""

    if adjustflag not in {"1", "2", "3"}:
        raise ValueError("adjustflag must be 1, 2, or 3")
    result = bs.query_history_k_data_plus(
        code,
        ",".join(DAILY_FIELDS),
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjustflag,
    )
    frame = result_to_frame(
        result,
        query_name=f"daily bars {code} {start_date}:{end_date} adjust={adjustflag}",
    )
    frame.insert(0, "requested_adjustflag", adjustflag)
    return frame


def fetch_index_bars(
    bs: object,
    *,
    code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch a daily index series for benchmark-relative outcomes."""

    result = bs.query_history_k_data_plus(
        code,
        ",".join(INDEX_DAILY_FIELDS),
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    return result_to_frame(
        result,
        query_name=f"index bars {code} {start_date}:{end_date}",
    )
