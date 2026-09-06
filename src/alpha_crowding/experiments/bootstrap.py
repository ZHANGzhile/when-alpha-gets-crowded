"""Calendar-week block bootstrap for paired model-loss differences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlockBootstrapResult:
    estimate: float
    lower_95: float
    upper_95: float
    probability_nonpositive: float
    repetitions: int
    block_length: int


def moving_block_bootstrap_mean(
    weekly_values: pd.Series,
    *,
    block_length: int = 13,
    repetitions: int = 10_000,
    seed: int = 42,
) -> BlockBootstrapResult:
    """Bootstrap the mean while preserving contiguous weekly dependence."""

    values = pd.to_numeric(weekly_values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    if block_length <= 0 or repetitions <= 0:
        raise ValueError("block_length and repetitions must be positive")
    if n < block_length:
        raise ValueError("fewer observations than the requested block length")
    starts = np.arange(0, n - block_length + 1)
    blocks_needed = int(np.ceil(n / block_length))
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for draw in range(repetitions):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block_length] for start in chosen])[:n]
        draws[draw] = sample.mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return BlockBootstrapResult(
        estimate=float(values.mean()),
        lower_95=float(lower),
        upper_95=float(upper),
        probability_nonpositive=float(np.mean(draws <= 0)),
        repetitions=repetitions,
        block_length=block_length,
    )

