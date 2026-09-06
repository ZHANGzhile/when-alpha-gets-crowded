"""Auditable same-date structural measurements and matched controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import summarize_matched_placebo
from .placebo import draw_matched_placebos, stable_group_seed
from .statistics import industry_residual_returns, shrunk_correlation_features


ALGORITHM_VERSION = "structural_matched_placebo_v1"


@dataclass(frozen=True, slots=True)
class StructuralPlaceboResult:
    features: pd.DataFrame
    placebo_features: pd.DataFrame
    placebo_diagnostics: pd.DataFrame
    placebo_memberships: pd.DataFrame


def candidate_snapshot_sha256(candidates: pd.DataFrame) -> str:
    """Hash the matching inputs in a stable order for later reproduction."""

    columns = ["code", "industry", "lagged_liquidity"]
    missing = set(columns) - set(candidates.columns)
    if missing:
        raise KeyError(f"missing candidate hash fields: {sorted(missing)}")
    records = (
        candidates.loc[:, columns]
        .sort_values("code", kind="mergesort")
        .replace({np.nan: None})
        .to_dict("records")
    )
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_values(
    residual_matrix: pd.DataFrame,
    codes: list[str],
    *,
    minimum_days: int,
    minimum_assets: int,
) -> dict[str, object]:
    missing = set(codes) - set(residual_matrix.columns)
    if missing:
        raise ValueError(f"return window missing actual assets: {sorted(missing)[:5]}")
    matrix = residual_matrix.loc[:, codes]
    measured = shrunk_correlation_features(
        matrix, minimum_days=minimum_days, minimum_assets=minimum_assets
    )
    return measured.to_dict()


def measure_structural_placebos(
    daily_returns: pd.DataFrame,
    candidates: pd.DataFrame,
    actual_codes: list[str],
    *,
    decision_at: object,
    factor: str,
    leg: str,
    draws: int = 100,
    minimum_valid_draws: int = 80,
    base_seed: int = 42,
    minimum_days: int = 40,
    minimum_assets: int = 10,
    temperature: float = 0.75,
    maximum_mean_absolute_z_difference: float = 0.35,
    maximum_attempts_per_draw: int = 50,
    exclude_actual: bool = False,
) -> StructuralPlaceboResult:
    """Measure one factor leg and its same-date matched placebo distribution."""

    decision = pd.Timestamp(decision_at).normalize()
    candidate_columns = {"code", "industry", "lagged_liquidity"}
    missing_candidates = candidate_columns - set(candidates.columns)
    if missing_candidates:
        raise KeyError(f"missing candidate fields: {sorted(missing_candidates)}")
    if not actual_codes:
        raise ValueError("actual_codes must not be empty")
    candidate_frame = candidates.loc[:, sorted(candidate_columns)].copy()
    candidate_frame["code"] = candidate_frame["code"].astype(str)
    if candidate_frame["code"].duplicated().any():
        raise ValueError("candidates must be unique by code")
    actual = [str(code) for code in actual_codes]
    if len(actual) != len(set(actual)):
        raise ValueError("actual_codes must be unique")

    required_returns = {"date", "code", "daily_return"}
    missing_returns = required_returns - set(daily_returns.columns)
    if missing_returns:
        raise KeyError(f"missing daily-return fields: {sorted(missing_returns)}")
    panel = daily_returns.loc[:, ["date", "code", "daily_return"]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["code"] = panel["code"].astype(str)
    panel = panel[panel["code"].isin(candidate_frame["code"])]
    panel = panel.merge(
        candidate_frame[["code", "industry"]], on="code", how="left", validate="many_to_one"
    )
    residuals = industry_residual_returns(panel)
    residual_matrix = residuals.pivot(
        index="date", columns="code", values="industry_residual_return"
    )

    actual_values = _feature_values(
        residual_matrix,
        actual,
        minimum_days=minimum_days,
        minimum_assets=minimum_assets,
    )
    seed = stable_group_seed(
        base_seed, decision_at=decision, factor=str(factor), leg=str(leg)
    )
    snapshot_hash = candidate_snapshot_sha256(candidate_frame)
    sampled = draw_matched_placebos(
        candidate_frame,
        actual,
        draws=draws,
        seed=seed,
        exclude_actual=exclude_actual,
        temperature=temperature,
        maximum_mean_absolute_z_difference=maximum_mean_absolute_z_difference,
        maximum_attempts_per_draw=maximum_attempts_per_draw,
    )
    diagnostics = sampled.diagnostics.copy()
    diagnostic_prefix = {
        "decision_at": decision,
        "factor": str(factor),
        "leg": str(leg),
        "group_seed": seed,
        "candidate_count": int(len(candidate_frame)),
        "actual_member_count": int(len(actual)),
        "candidate_snapshot_sha256": snapshot_hash,
        "algorithm_version": ALGORITHM_VERSION,
        "exclude_actual": bool(exclude_actual),
    }
    for column, value in reversed(list(diagnostic_prefix.items())):
        diagnostics.insert(0, column, value)

    placebo_rows: list[dict[str, object]] = []
    membership = sampled.memberships.copy()
    valid_draw_members = {
        draw_id: group["code"].astype(str).tolist()
        for draw_id, group in membership.groupby("draw_id", sort=False)
    }
    for draw_number in range(draws):
        draw_id = f"draw_{draw_number:03d}"
        row_base = {
            "decision_at": decision,
            "factor": str(factor),
            "leg": str(leg),
            "draw_id": draw_id,
        }
        codes = valid_draw_members.get(draw_id)
        if codes is None:
            for feature in ("residual_sync", "eigen_concentration"):
                placebo_rows.append(
                    {
                        **row_base,
                        "feature": feature,
                        "value": np.nan,
                        "valid_days": 0,
                        "valid_assets": 0,
                        "invalid_reason": "matching_invalid",
                    }
                )
            continue
        try:
            values = _feature_values(
                residual_matrix,
                codes,
                minimum_days=minimum_days,
                minimum_assets=minimum_assets,
            )
            for feature in ("residual_sync", "eigen_concentration"):
                placebo_rows.append(
                    {
                        **row_base,
                        "feature": feature,
                        "value": float(values[feature]),
                        "valid_days": int(values["valid_days"]),
                        "valid_assets": int(values["valid_assets"]),
                        "invalid_reason": None,
                    }
                )
        except ValueError as exc:
            for feature in ("residual_sync", "eigen_concentration"):
                placebo_rows.append(
                    {
                        **row_base,
                        "feature": feature,
                        "value": np.nan,
                        "valid_days": 0,
                        "valid_assets": 0,
                        "invalid_reason": str(exc),
                    }
                )
    placebo_features = pd.DataFrame(placebo_rows)

    feature_rows = []
    for feature in ("residual_sync", "eigen_concentration"):
        values = placebo_features.loc[
            placebo_features["feature"].eq(feature), "value"
        ].to_numpy(dtype=float)
        summary = summarize_matched_placebo(
            float(actual_values[feature]),
            values,
            min_valid_draws=minimum_valid_draws,
        ).to_dict()
        feature_rows.append(
            {
                "decision_at": decision,
                "factor": str(factor),
                "leg": str(leg),
                "feature": feature,
                **summary,
                "actual_valid_days": int(actual_values["valid_days"]),
                "actual_valid_assets": int(actual_values["valid_assets"]),
                "window_start": residuals["date"].min(),
                "window_end": residuals["date"].max(),
                "candidate_snapshot_sha256": snapshot_hash,
                "algorithm_version": ALGORITHM_VERSION,
            }
        )
    features = pd.DataFrame(feature_rows)

    if not membership.empty:
        membership.insert(0, "decision_at", decision)
        membership.insert(1, "factor", str(factor))
        membership.insert(2, "leg", str(leg))
        membership["is_actual_member"] = membership["code"].isin(actual)
    return StructuralPlaceboResult(
        features=features,
        placebo_features=placebo_features,
        placebo_diagnostics=diagnostics,
        placebo_memberships=membership,
    )
