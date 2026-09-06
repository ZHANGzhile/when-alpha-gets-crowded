"""Build continuous permuted-rank pseudo memberships and full return ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.experiments import (
    PSEUDO_ALGORITHM_VERSION,
    build_continuous_permuted_memberships,
)
from alpha_crowding.factors import combine_long_short_returns, simulate_factor_leg_returns


ROOT = Path(__file__).resolve().parents[1]
REAL_MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
DAILY = ROOT / "data" / "interim" / "daily_market"
EXPERIMENT_CONFIG = ROOT / "config" / "experiments.yaml"
FACTOR_CONFIG = ROOT / "config" / "factors.yaml"
MEMBERSHIP_CHECKPOINTS = ROOT / "data" / "processed" / "pseudo_memberships_by_date"
RETURN_CHECKPOINTS = ROOT / "data" / "processed" / "pseudo_returns_by_strategy"
AUDIT_OUTPUT = ROOT / "data" / "processed" / "pseudo_membership_diagnostics.parquet"
LEG_OUTPUT = ROOT / "data" / "processed" / "pseudo_leg_returns.parquet"
FACTOR_OUTPUT = ROOT / "data" / "processed" / "pseudo_factor_returns.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "continuous_pseudo_returns.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _calendar() -> pd.DatetimeIndex:
    paths = list((ROOT / "data" / "raw" / "membership").glob("trade_calendar_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one production trade calendar; found {len(paths)}")
    frame = pd.read_csv(paths[0])
    return pd.DatetimeIndex(
        pd.to_datetime(
            frame.loc[
                pd.to_numeric(frame["is_trading_day"], errors="raise").eq(1),
                "calendar_date",
            ]
        )
    ).normalize().sort_values()


def _materialize_membership_date(
    date: pd.Timestamp,
    candidates: pd.DataFrame,
    *,
    strategy_count: int,
    base_seed: int,
    quantile: float,
    source_sha256: str,
) -> dict[str, object]:
    directory = MEMBERSHIP_CHECKPOINTS / str(date.date())
    membership_path = directory / "memberships.parquet"
    audit_path = directory / "diagnostics.parquet"
    marker = directory / "status.json"
    signature = {
        "strategy_count": strategy_count,
        "base_seed": base_seed,
        "quantile": quantile,
        "source_sha256": source_sha256,
        "algorithm_version": PSEUDO_ALGORITHM_VERSION,
    }
    if marker.exists() and membership_path.exists() and audit_path.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in signature.items()):
            existing["source"] = "checkpoint"
            return existing

    result = build_continuous_permuted_memberships(
        candidates,
        strategy_count=strategy_count,
        base_seed=base_seed,
        quantile=quantile,
    )
    _atomic_parquet(result.memberships, membership_path)
    _atomic_parquet(result.diagnostics, audit_path)
    payload = {
        **signature,
        "decision_at": str(date.date()),
        "source": "computed",
        "membership_rows": len(result.memberships),
        "diagnostic_rows": len(result.diagnostics),
        "membership_sha256": _sha256(membership_path),
        "diagnostics_sha256": _sha256(audit_path),
    }
    _atomic_json(payload, marker)
    return payload


def _load_security_returns() -> pd.DataFrame:
    paths = sorted(DAILY.glob("year=*/*.parquet"))
    if not paths:
        raise FileNotFoundError("daily research panel partitions are missing")
    parts = [
        pd.read_parquet(path, columns=["date", "code", "daily_return"])
        for path in paths
    ]
    result = pd.concat(parts, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    if result.duplicated(["date", "code"]).any():
        raise ValueError("daily research panel contains duplicate date/code rows")
    return result


def _strategy_memberships(
    dates: pd.DatetimeIndex,
    pseudo_id: str,
) -> pd.DataFrame:
    parts = []
    for date in dates:
        path = MEMBERSHIP_CHECKPOINTS / str(date.date()) / "memberships.parquet"
        frame = pd.read_parquet(
            path,
            filters=[("pseudo_strategy_id", "==", pseudo_id)],
        )
        if not frame.empty:
            parts.append(frame)
    if not parts:
        raise ValueError(f"no memberships found for {pseudo_id}")
    return pd.concat(parts, ignore_index=True)


def _materialize_strategy_returns(
    pseudo_id: str,
    dates: pd.DatetimeIndex,
    security_returns: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    source_sha256: str,
) -> dict[str, object]:
    directory = RETURN_CHECKPOINTS / pseudo_id
    leg_path = directory / "leg_returns.parquet"
    factor_path = directory / "factor_returns.parquet"
    marker = directory / "status.json"
    signature = {
        "pseudo_strategy_id": pseudo_id,
        "source_sha256": source_sha256,
        "algorithm_version": PSEUDO_ALGORITHM_VERSION,
    }
    if marker.exists() and leg_path.exists() and factor_path.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in signature.items()):
            existing["source"] = "checkpoint"
            return existing

    memberships = _strategy_memberships(dates, pseudo_id)
    executable = memberships.loc[memberships["date"].lt(calendar.max())].copy()
    legs = simulate_factor_leg_returns(executable, security_returns, calendar)
    factors = combine_long_short_returns(legs)
    for frame in (legs, factors):
        frame["strategy_key"] = frame["factor"]
        frame["original_factor"] = frame["factor"].str.split("|", n=1).str[0]
        frame["pseudo_strategy_id"] = pseudo_id
    _atomic_parquet(legs, leg_path)
    _atomic_parquet(factors, factor_path)
    payload = {
        **signature,
        "source": "computed",
        "leg_rows": len(legs),
        "factor_rows": len(factors),
        "leg_sha256": _sha256(leg_path),
        "factor_sha256": _sha256(factor_path),
    }
    _atomic_json(payload, marker)
    return payload


def main(*, strategy_count: int | None, maximum_dates: int | None) -> int:
    experiment = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    factor_config = yaml.safe_load(FACTOR_CONFIG.read_text(encoding="utf-8"))
    configured_count = int(experiment["placebos"]["continuous_strategy_count"])
    count = configured_count if strategy_count is None else int(strategy_count)
    if count < 1 or count > configured_count:
        raise ValueError(f"strategy_count must be between 1 and {configured_count}")
    base_seed = int(experiment["placebos"]["continuous_strategy_seed"])
    quantile = float(factor_config["portfolio_quantile"])
    real = pd.read_parquet(REAL_MEMBERSHIPS)
    real["date"] = pd.to_datetime(real["date"], errors="raise").dt.normalize()
    dates = pd.DatetimeIndex(sorted(real["date"].unique()))
    if maximum_dates is not None:
        if maximum_dates < 1:
            raise ValueError("maximum_dates must be positive")
        dates = dates[:maximum_dates]
    source_sha256 = _sha256(REAL_MEMBERSHIPS)

    date_summaries = []
    for position, date in enumerate(dates, start=1):
        summary = _materialize_membership_date(
            date,
            real.loc[real["date"].eq(date)],
            strategy_count=count,
            base_seed=base_seed,
            quantile=quantile,
            source_sha256=source_sha256,
        )
        date_summaries.append(summary)
        print(
            f"continuous pseudo memberships {position}/{len(dates)} "
            f"date={date.date()} source={summary['source']}",
            flush=True,
        )

    audits = pd.concat(
        [
            pd.read_parquet(
                MEMBERSHIP_CHECKPOINTS / str(date.date()) / "diagnostics.parquet"
            )
            for date in dates
        ],
        ignore_index=True,
    )
    expected_groups = len(
        real.loc[real["date"].isin(dates), ["date", "factor"]].drop_duplicates()
    )
    coverage = audits.groupby("pseudo_strategy_id").size()
    if not coverage.eq(expected_groups).all():
        raise ValueError("a continuous pseudo identity is missing decision/factor groups")
    _atomic_parquet(audits, AUDIT_OUTPUT)

    security_returns = _load_security_returns()
    calendar = _calendar()
    if maximum_dates is not None:
        final_position = int(calendar.get_loc(dates.max()))
        calendar = calendar[: min(len(calendar), final_position + 6)]
    membership_digest = hashlib.sha256(
        "\n".join(summary["membership_sha256"] for summary in date_summaries).encode("utf-8")
    ).hexdigest()
    return_summaries = []
    for number in range(count):
        pseudo_id = f"pseudo_{number:03d}"
        summary = _materialize_strategy_returns(
            pseudo_id,
            dates,
            security_returns,
            calendar,
            source_sha256=membership_digest,
        )
        return_summaries.append(summary)
        print(
            f"continuous pseudo returns {number + 1}/{count} "
            f"strategy={pseudo_id} source={summary['source']}",
            flush=True,
        )

    leg_returns = pd.concat(
        [
            pd.read_parquet(RETURN_CHECKPOINTS / f"pseudo_{number:03d}" / "leg_returns.parquet")
            for number in range(count)
        ],
        ignore_index=True,
    )
    factor_returns = pd.concat(
        [
            pd.read_parquet(
                RETURN_CHECKPOINTS / f"pseudo_{number:03d}" / "factor_returns.parquet"
            )
            for number in range(count)
        ],
        ignore_index=True,
    )
    _atomic_parquet(leg_returns, LEG_OUTPUT)
    _atomic_parquet(factor_returns, FACTOR_OUTPUT)
    status = "PARTIAL" if maximum_dates is not None or count != configured_count else "COMPLETE"
    payload = {
        "schema_version": 1,
        "purpose": "continuous_within_industry_permuted_rank_membership_and_return_pipeline",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm_version": PSEUDO_ALGORITHM_VERSION,
        "strategy_count": count,
        "configured_strategy_count": configured_count,
        "base_seed": base_seed,
        "decision_dates": len(dates),
        "factor_count": int(real["factor"].nunique()),
        "diagnostic_rows": len(audits),
        "leg_return_rows": len(leg_returns),
        "factor_return_rows": len(factor_returns),
        "membership_checkpoint_digest": membership_digest,
        "return_checkpoint_count": len(return_summaries),
        "outputs": {
            str(AUDIT_OUTPUT.relative_to(ROOT)): _sha256(AUDIT_OUTPUT),
            str(LEG_OUTPUT.relative_to(ROOT)): _sha256(LEG_OUTPUT),
            str(FACTOR_OUTPUT.relative_to(ROOT)): _sha256(FACTOR_OUTPUT),
        },
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-count", type=int)
    parser.add_argument("--maximum-dates", type=int)
    args = parser.parse_args()
    raise SystemExit(
        main(strategy_count=args.strategy_count, maximum_dates=args.maximum_dates)
    )
