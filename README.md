# When Alpha Gets Crowded

[English](README.md) · [简体中文](README.zh-CN.md)

An auditable research system for testing whether abnormal structure inside equity factor
portfolios predicts future factor fragility beyond ordinary market and portfolio risk.

The project studies weekly Momentum, Reversal, Low Volatility, and conditionally Value portfolios
in the historical CSI 300/CSI 500 universe. It separates three ideas that are often mixed together:

- **Structural Crowding (`C`)**: placebo-adjusted residual synchronization, eigenvalue
  concentration, and cross-factor strategy convergence.
- **Generic Portfolio Risk (`G`)**: raw correlation, concentration, turnover, and illiquidity.
- **Unwind Stress (`S`)**: recent turnover, liquidity, and factor-return shocks.

These metrics are market-implied proxies rather than observed institutional positions. The
confirmatory question is whether `C` improves strictly out-of-sample predictions after Market
State, Factor State, `G`, and `S` are already included.

## What is implemented

- Point-in-time weekly CSI 300/CSI 500 membership ingestion with checkpointed raw responses.
- Audited repairs for five merger-related 499-member snapshots with row-level provenance.
- Universe-wide BaoStock daily ingestion, acceptance gates, and resumable workers.
- Weekly factor signals, deterministic long/short legs, next-session execution, and weight drift.
- Ledoit-Wolf residual correlation and eigen concentration over exact 60-session windows.
- 100 same-date industry-exact, lagged-liquidity-matched placebo portfolios per factor leg.
- Joint multi-factor placebo draws for Strategy Convergence.
- Exact past-252-trading-session Historical Z-scores with current observations excluded.
- Market, factor, `C`, `G`, and `S` state tables with explicit validity and coverage fields.
- Dynamic LS, leg-level, Active Long/CSI 800, and fixed-membership outcome builders.
- Annual expanding M0-M4 logistic comparisons with purged inner validation and paired
  moving-block bootstrap inference.
- Strictly historical, same-factor OOS exposure schedules for M2/M3/M4, with a 52-week warm-up
  and next-session activation.
- A point-in-time contract for tradable CSI 800 replication weights; membership-only and
  equal-weight substitutes are rejected before stock-level controller accounting.
- A zero-active-weight executable CSI 800 control that uses the same opening constraints and
  cost engine, audits gross/net tracking error and gross active-return bias, and stops production
  when its pre-specified quality limits fail.
- A stock-level controller engine with post-cost target solving, daily holding drift, cash,
  side-specific opening constraints, actual-trade costs, and a per-security rejection ledger.
- Correct opening-rebalance return attribution: old holdings receive overnight returns, executed
  targets receive intraday returns, and the two legs reconstruct close-to-close returns.
- Absolute and CSI 800-relative P7 metrics, including Sharpe, Sortino, MDD, worst 20-session
  return, tracking error, information ratio, Active MDD/CVaR, Active-target crash episode loss,
  exposure, turnover, and rejections.
- An explicitly non-executable, ex-post constant exposure matched to each factor's realized M3
  mean, used only to distinguish timing value from average de-risking.
- A protocol hash gate that blocks confirmatory outcomes until the P0 data gates are closed.

The implementation rationale and problem log are in
[`docs/implementation_journal.md`](docs/implementation_journal.md). The research contract is in
[`research_protocol.md`](research_protocol.md), and the architecture is in
[`implementation_design.md`](implementation_design.md).

## Current research status

The code path reaches the primary M0-M4 walk-forward comparison. The project remains a research
work in progress because several gates are open:

- terminal/delisting settlement returns must be sourced rather than forward-filled;
- PB publication timing must pass point-in-time review before Value enters the primary analysis;
- the historical industry taxonomy transition needs a measured impact audit;
- membership events need an independent cross-check;
- Protocol-gated entry points now cover 5/10-session leads, crash event studies, M2/M3 placebo
  context, 26/52-week misalignment, discriminant validity, and the full continuous-strategy
  placebo path. Each of 100 pseudo-strategies has stable memberships, return ledgers,
  leave-one-out structural C, comparable G/S and factor states, identity-specific mature LS
  outcomes, and an independently tuned annual OOS M2/M3 comparison.
- The controller exposure schedule, benchmark data gate, pure-replication quality gate, and
  constrained stock-level accounting engine are implemented, including pre-specified fixed-low
  and trailing-volatility comparators. Production P7 remains blocked until the Tushare reference
  inputs pass their real-data acceptance checks.

Raw data, generated manifests, logs, model outputs, local reports, and source PDFs are excluded from
Git. The repository contains reproducible code, configuration candidates, and audit notes.

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The P7 reference-data downloader uses the optional Tushare client:

```powershell
python -m pip install -e ".[controller-data]"
$env:TUSHARE_TOKEN = "your-local-token"
```

The token stays in the process environment and is never written to repository files or manifests.
The account needs access to the `index_weight` and `stk_limit` endpoints.

If Python lives elsewhere, set `ALPHA_CROWDING_PYTHON` to its executable before using the
PowerShell helpers.

## Verification

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_checks.ps1
```

The behavior suite covers point-in-time joins, exact trading windows, portfolio accounting,
matched placebos, outcome maturity, temporal purging, and walk-forward fitting. Tests validate the
implementation; they are not research evidence.

## Production pipeline

Run individual stages:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage membership
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage daily
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage measurements
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage analysis
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage falsification
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage controller
```

Run the dependency-ordered pipeline:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage all
```

Network and long-running stages write atomic checkpoints under `data/`. After non-outcome
precomputation, the background supervisor enters `WAITING_PROTOCOL_FREEZE` and checks once per
minute. It starts confirmatory outcomes only when `protocol_freeze.json` exists and every recorded
artifact hash still matches.

## Repository layout

```text
config/                  Candidate definitions and thresholds
docs/                    Acceptance reports and implementation journal
scripts/                 Audits and numbered production stages
src/alpha_crowding/      Data, factor, measurement, outcome, experiment, and backtest code
tests/                   Behavioral and temporal-integrity checks
implementation_design.md Full implementation architecture
research_protocol.md     Confirmatory research contract (currently DRAFT)
```

## Reproducibility rules

1. Preserve raw responses and checksums locally; never overwrite them with cleaned data.
2. Use only information available by each decision timestamp.
3. Keep Long and Short measurements separate until the model layer.
4. Keep structural crowding, generic risk, and short-term stress in separate feature families.
5. Do not replace missing settlements, zero MAD, or invalid placebo distributions with constants.
6. Freeze and hash the protocol before constructing confirmatory outcomes.

This is a research codebase, not a live trading or brokerage system.
