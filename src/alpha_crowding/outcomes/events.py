"""Independent crash episodes and auditable event-study windows."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def _date_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise KeyError(f"missing event column: {column!r}")
    values = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    if values.isna().any():
        raise ValueError(f"{column!r} contains missing dates")
    return values


def _merge_records(
    records: list[dict[str, object]],
    *,
    overlap_start: str,
    overlap_end: str,
) -> list[dict[str, object]]:
    if not records:
        return []
    ordered = sorted(records, key=lambda row: (row[overlap_start], row[overlap_end]))
    merged = [ordered[0].copy()]
    for row in ordered[1:]:
        current = merged[-1]
        if row[overlap_start] <= current[overlap_end]:
            current["interval_start_at"] = min(
                current["interval_start_at"], row["interval_start_at"]
            )
            current["interval_end_at"] = max(
                current["interval_end_at"], row["interval_end_at"]
            )
            current["event_at"] = min(current["event_at"], row["event_at"])
            current["window_start_at"] = min(
                current["window_start_at"], row["window_start_at"]
            )
            current["window_end_at"] = max(
                current["window_end_at"], row["window_end_at"]
            )
            current["source_event_count"] += row["source_event_count"]
        else:
            merged.append(row.copy())
    return merged


def merge_crash_episodes(
    events: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("factor", "target_family", "membership_mode"),
    event_flag_col: str = "tail_event",
    event_at_col: str = "breach_at",
    interval_start_col: str = "label_start_at",
    interval_end_col: str = "label_end_at",
    pre_weeks: int = 8,
    post_weeks: int = 4,
) -> pd.DataFrame:
    """Merge overlapping crash intervals and then overlapping study windows.

    Event zero is always the earliest observed breach in the merged episode.
    The two-pass merge follows the protocol literally: realized outcome
    intervals are consolidated first, then episodes whose ``[-8w,+4w]``
    windows still overlap are consolidated again.
    """

    if pre_weeks < 0 or post_weeks < 0:
        raise ValueError("event window lengths must be non-negative")
    required = {*group_cols, event_flag_col, event_at_col, interval_start_col, interval_end_col}
    missing = required - set(events.columns)
    if missing:
        raise KeyError(f"missing event columns: {sorted(missing)}")
    selected = events.loc[events[event_flag_col].fillna(False).astype(bool)].copy()
    output_columns = [
        *group_cols,
        "episode_id",
        "event_at",
        "interval_start_at",
        "interval_end_at",
        "window_start_at",
        "window_end_at",
        "source_event_count",
    ]
    if selected.empty:
        return pd.DataFrame(columns=output_columns)

    selected["event_at"] = _date_column(selected, event_at_col)
    selected["interval_start_at"] = _date_column(selected, interval_start_col)
    selected["interval_end_at"] = _date_column(selected, interval_end_col)
    if (selected["interval_end_at"] < selected["interval_start_at"]).any():
        raise ValueError("event interval end precedes start")
    outside = ~selected["event_at"].between(
        selected["interval_start_at"], selected["interval_end_at"]
    )
    if outside.any():
        raise ValueError("breach_at must lie inside its realized outcome interval")
    selected["window_start_at"] = selected["event_at"] - pd.to_timedelta(
        7 * pre_weeks, unit="D"
    )
    selected["window_end_at"] = selected["event_at"] + pd.to_timedelta(
        7 * post_weeks, unit="D"
    )
    selected["source_event_count"] = 1

    episodes: list[dict[str, object]] = []
    grouper: object = list(group_cols) if len(group_cols) > 1 else group_cols[0]
    for raw_key, group in selected.groupby(grouper, sort=True, dropna=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        records = group[
            [
                "event_at",
                "interval_start_at",
                "interval_end_at",
                "window_start_at",
                "window_end_at",
                "source_event_count",
            ]
        ].to_dict("records")
        interval_merged = _merge_records(
            records,
            overlap_start="interval_start_at",
            overlap_end="interval_end_at",
        )
        window_merged = _merge_records(
            interval_merged,
            overlap_start="window_start_at",
            overlap_end="window_end_at",
        )
        for sequence, record in enumerate(window_merged, start=1):
            for column, value in zip(group_cols, key):
                record[column] = value
            identity = "|".join(str(value) for value in key)
            record["episode_id"] = f"{identity}|{sequence:04d}"
            episodes.append(record)
    result = pd.DataFrame(episodes)
    return result[output_columns].sort_values([*group_cols, "event_at"]).reset_index(drop=True)


def build_event_study_panel(
    weekly_panel: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("factor", "target_family", "membership_mode"),
    date_col: str = "decision_at",
) -> pd.DataFrame:
    """Join weekly observations to episodes and assign event-relative weeks."""

    required_panel = {*group_cols, date_col}
    missing_panel = required_panel - set(weekly_panel.columns)
    if missing_panel:
        raise KeyError(f"missing weekly panel columns: {sorted(missing_panel)}")
    required_episodes = {
        *group_cols,
        "episode_id",
        "event_at",
        "window_start_at",
        "window_end_at",
    }
    missing_episodes = required_episodes - set(episodes.columns)
    if missing_episodes:
        raise KeyError(f"missing episode columns: {sorted(missing_episodes)}")

    panel = weekly_panel.copy()
    panel[date_col] = _date_column(panel, date_col)
    if panel.duplicated([*group_cols, date_col]).any():
        raise ValueError("weekly panel must be unique by group and decision date")
    calendar = pd.DatetimeIndex(panel[date_col].drop_duplicates().sort_values())
    parts: list[pd.DataFrame] = []
    for episode in episodes.to_dict("records"):
        event_at = pd.Timestamp(episode["event_at"]).normalize()
        event_position = int(calendar.searchsorted(event_at, side="left"))
        if event_position == len(calendar):
            continue
        mask = pd.Series(True, index=panel.index)
        for column in group_cols:
            mask &= panel[column].eq(episode[column])
        mask &= panel[date_col].between(
            pd.Timestamp(episode["window_start_at"]),
            pd.Timestamp(episode["window_end_at"]),
        )
        part = panel.loc[mask].copy()
        if part.empty:
            continue
        positions = calendar.get_indexer(pd.DatetimeIndex(part[date_col]))
        part["episode_id"] = episode["episode_id"]
        part["event_at"] = event_at
        part["aligned_event_at"] = calendar[event_position]
        part["relative_week"] = positions - event_position
        parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=[*weekly_panel.columns, "episode_id", "event_at", "aligned_event_at", "relative_week"]
        )
    return pd.concat(parts, ignore_index=True).sort_values(
        ["episode_id", "relative_week"]
    ).reset_index(drop=True)
