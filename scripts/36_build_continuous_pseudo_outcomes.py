"""Build each continuous pseudo strategy's mature dynamic LS outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alpha_crowding.experiments import require_protocol_freeze
from alpha_crowding.outcomes import (
    MatureTailSpec,
    build_mature_tail_labels,
    forward_window_outcome,
)


ROOT = Path(__file__).resolve().parents[1]
RETURNS = ROOT / "data" / "processed" / "pseudo_factor_returns.parquet"
STATE = ROOT / "data" / "processed" / "pseudo_state_panel.parquet"
CONFIG = ROOT / "config" / "outcomes.yaml"
EXPERIMENTS = ROOT / "config" / "experiments.yaml"
CHECKPOINTS = ROOT / "data" / "processed" / "pseudo_outcomes_by_strategy"
OUTPUT = ROOT / "data" / "processed" / "pseudo_dynamic_outcomes.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "continuous_pseudo_outcomes.json"
STATE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "continuous_pseudo_state.json"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"


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


def _decision_digest(decisions: pd.DataFrame) -> str:
    return hashlib.sha256(
        decisions.sort_values(["decision_at", "original_factor"])
        .to_csv(index=False)
        .encode("utf-8")
    ).hexdigest()


def _strategy_outcomes(
    pseudo_id: str,
    returns: pd.DataFrame,
    decisions: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: dict[str, object],
    *,
    source_sha256: str,
    freeze_sha256: str,
) -> dict[str, object]:
    directory = CHECKPOINTS / pseudo_id
    output_path = directory / "outcomes.parquet"
    marker = directory / "status.json"
    strategy_decisions = decisions.loc[
        decisions["pseudo_strategy_id"].eq(pseudo_id),
        ["decision_at", "original_factor"],
    ]
    signature = {
        "pseudo_strategy_id": pseudo_id,
        "return_source_sha256": source_sha256,
        "outcome_config_sha256": _sha256(CONFIG),
        "protocol_freeze_sha256": freeze_sha256,
        "decision_digest": _decision_digest(strategy_decisions),
    }
    if marker.exists() and output_path.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in signature.items()):
            existing["source"] = "checkpoint"
            return existing

    research_horizons = sorted(
        {
            int(config["primary"]["horizon_trading_days"]),
            *[int(value) for value in config["secondary"]["horizons_trading_days"]],
        }
    )
    rows = []
    strategy_returns = returns.loc[returns["pseudo_strategy_id"].eq(pseudo_id)]
    for factor, group in strategy_decisions.groupby("original_factor", sort=True):
        factor_returns = strategy_returns.loc[
            strategy_returns["original_factor"].eq(factor)
        ].sort_values("date")
        series = factor_returns.set_index("date")["long_short_return"]
        for decision_at in sorted(group["decision_at"].unique()):
            for horizon in research_horizons:
                measured = forward_window_outcome(
                    series,
                    calendar,
                    decision_at=decision_at,
                    horizon_sessions=horizon,
                )
                rows.append(
                    {
                        "original_factor": factor,
                        "pseudo_strategy_id": pseudo_id,
                        "target_family": "research_ls",
                        "membership_mode": "continuous_pseudo_dynamic",
                        **measured,
                    }
                )
    output = pd.DataFrame(rows)
    mature_mask = output["outcome_mature"] & output["label_end_at"].notna()
    mature = output.loc[mature_mask].copy()
    group_cols = (
        "original_factor",
        "pseudo_strategy_id",
        "target_family",
        "membership_mode",
        "horizon_sessions",
    )
    quantiles = {
        0.10,
        *[float(value) for value in config["secondary"]["crash_quantiles"]],
    }
    for quantile in sorted(quantiles):
        suffix = f"q{int(round(quantile * 100)):02d}"
        labelled = build_mature_tail_labels(
            mature,
            sessions=calendar,
            group_cols=group_cols,
            spec=MatureTailSpec(
                quantile=quantile,
                lookback_sessions=int(config["primary"]["threshold_history_trading_days"]),
                min_history=int(config["primary"]["minimum_mature_weekly_observations"]),
            ),
            threshold_name=f"historical_tail_threshold_{suffix}",
            history_count_name=f"mature_history_count_{suffix}",
            label_name=f"crash_{suffix}",
        )
        threshold = f"historical_tail_threshold_{suffix}"
        history = f"mature_history_count_{suffix}"
        label = f"crash_{suffix}"
        output[threshold] = np.nan
        output[history] = 0
        output[label] = pd.Series(pd.NA, index=output.index, dtype="boolean")
        for column in (threshold, history, label):
            output.loc[mature_mask, column] = labelled[column].to_numpy()
        output[history] = output[history].astype("int64")
    output = output.sort_values(
        ["decision_at", "original_factor", "horizon_sessions"]
    ).reset_index(drop=True)
    _atomic_parquet(output, output_path)
    payload = {
        **signature,
        "source": "computed",
        "rows": len(output),
        "mature_rows": int(output["outcome_mature"].sum()),
        "main_label_rows": int(output["crash_q10"].notna().sum()),
        "main_crash_rows": int(output["crash_q10"].fillna(False).sum()),
        "output_sha256": _sha256(output_path),
    }
    _atomic_json(payload, marker)
    return payload


def main(*, maximum_strategies: int | None) -> int:
    freeze = require_protocol_freeze(FREEZE_MANIFEST, ROOT)
    freeze_sha = _sha256(FREEZE_MANIFEST)
    upstream = json.loads(STATE_MANIFEST.read_text(encoding="utf-8"))
    if upstream.get("status") != "COMPLETE":
        raise ValueError("pseudo outcomes require a COMPLETE pseudo state manifest")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiments = yaml.safe_load(EXPERIMENTS.read_text(encoding="utf-8"))
    calendar = _calendar()
    state = pd.read_parquet(
        STATE, columns=["decision_at", "original_factor", "pseudo_strategy_id", "leg"]
    )
    state["decision_at"] = pd.to_datetime(state["decision_at"]).dt.normalize()
    decisions = state[
        ["decision_at", "original_factor", "pseudo_strategy_id"]
    ].drop_duplicates()
    pseudo_ids = sorted(decisions["pseudo_strategy_id"].unique())
    expected = int(experiments["placebos"]["continuous_strategy_count"])
    if len(pseudo_ids) != expected:
        raise ValueError(f"pseudo outcomes require {expected} strategies; found {len(pseudo_ids)}")
    if maximum_strategies is not None:
        if maximum_strategies < 1 or maximum_strategies > expected:
            raise ValueError(f"maximum_strategies must be between 1 and {expected}")
        pseudo_ids = pseudo_ids[:maximum_strategies]
    returns = pd.read_parquet(
        RETURNS,
        columns=["date", "original_factor", "pseudo_strategy_id", "long_short_return"],
    )
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    source_sha = _sha256(RETURNS)
    summaries = []
    for number, pseudo_id in enumerate(pseudo_ids, start=1):
        summary = _strategy_outcomes(
            pseudo_id,
            returns,
            decisions,
            calendar,
            config,
            source_sha256=source_sha,
            freeze_sha256=freeze_sha,
        )
        summaries.append(summary)
        print(
            f"pseudo outcomes {number}/{len(pseudo_ids)} strategy={pseudo_id} "
            f"source={summary['source']} labels={summary['main_label_rows']}/{summary['rows']}",
            flush=True,
        )
    output = pd.concat(
        [
            pd.read_parquet(CHECKPOINTS / pseudo_id / "outcomes.parquet")
            for pseudo_id in pseudo_ids
        ],
        ignore_index=True,
    ).sort_values(
        ["decision_at", "original_factor", "pseudo_strategy_id", "horizon_sessions"]
    )
    _atomic_parquet(output, OUTPUT)
    status = "PARTIAL" if maximum_strategies is not None else "COMPLETE"
    payload = {
        "schema_version": 1,
        "purpose": "continuous_pseudo_dynamic_ls_outcomes_and_mature_tail_labels",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_cutoff": freeze["raw_data_cutoff"],
        "protocol_freeze_sha256": freeze_sha,
        "strategy_count": len(pseudo_ids),
        "configured_strategy_count": expected,
        "rows": len(output),
        "mature_rows": int(output["outcome_mature"].sum()),
        "main_label_rows": int(output["crash_q10"].notna().sum()),
        "main_crash_rows": int(output["crash_q10"].fillna(False).sum()),
        "checkpoint_count": len(summaries),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": _sha256(OUTPUT),
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-strategies", type=int)
    args = parser.parse_args()
    raise SystemExit(main(maximum_strategies=args.maximum_strategies))
