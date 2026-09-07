# Research Protocol

Protocol status: **DRAFT — blocked on point-in-time data capability audit**  
Protocol version: 0.1.0  
Design date: 2026-09-05

This file is the machine-adjacent preregistration for the V1 empirical study. It must receive a
content hash and status `FROZEN` before confirmatory outcomes are inspected. Data-quality facts
may change feasibility; predictive results may not be used to change primary definitions.

## Confirmatory question

Does factor-specific abnormal structure improve out-of-sample prediction of dynamic long-short
20-trading-day factor crashes beyond market state, factor state, generic portfolio risk, and
stress main effects?

## Unit, universe, and information time

- Unit: factor × weekly decision date.
- Universe: point-in-time union of CSI300 and CSI500 constituents.
- Factors: Momentum, Reversal, Low Volatility, and Value, subject to the documented PIT rule.
- Raw interval: 2012-01-01 through 2026-08-31.
- Every input record must satisfy `available_at <= issue_at`.
- The final evaluable date is determined by `label_end_at <= raw_data_cutoff`.

## Primary endpoint

Dynamic weekly-rebalanced long-short factor cumulative return over the next 20 trading days,
converted to a factor-specific crash indicator using the historical 10% quantile.

The historical threshold uses only mature weekly 20-day outcomes whose `label_end_at` is no later
than the current information cutoff, within the previous 756 trading days. At least 104 mature
weekly outcomes are required. Otherwise the label is unavailable.

## Primary estimator and metric

- Estimator: pooled L2 Logistic with factor fixed effects and common slopes.
- No class weighting and no separate probability calibration for the primary model.
- Metric: out-of-sample Log Loss, evaluated by averaging losses across available factors within
  each week, then across weeks.
- Primary statistic: `LogLoss(M2) - LogLoss(M3)`. Positive values favor M3.
- Confirmatory success requires a positive statistic whose paired 13-week moving-block bootstrap
  95% interval has a lower bound above zero.

## Nested information sets

- M0: factor fixed effects + Market State.
- M1: M0 + Factor State.
- M2: M1 + Generic Portfolio Risk G + all Stress S main effects.
- M3: M2 + Structural Crowding C.
- M4: M3 + one preregistered composite C × composite S interaction.

M3 versus M2 is the sole confirmatory primary comparison. M4 versus M3 is a preregistered key
mechanism secondary comparison. All nested models use an identical sample mask, outer folds,
preprocessing rules, and hyperparameter search budget.

Because `Excess = Actual - PlaceboMean`, the following context diagnostic is mandatory:

- M2_context = M2 + matching PlaceboMean fields.
- M3_context = M2_context + C.

Actual, PlaceboMean, and Excess must not appear together as perfectly collinear columns.

## C, G, and S

Structural Crowding C contains only historical-z-scored, placebo-adjusted residual synchronization,
eigen concentration, and strategy convergence. It is generated separately for Long and Short.
The LS model retains both legs; the Active model uses Long only. `C_core` excludes strategy
convergence and is a mandatory ablation.

Generic Portfolio Risk G contains raw constituent/portfolio levels, including residual correlation,
eigen concentration, turnover level, and illiquidity level.

Stress S contains recent innovations: turnover shock, turnover synchronization, liquidity shock,
and negative factor return shock. Every component is directed so a larger value means more stress.
Variables included in the interaction also appear as main effects in M2–M4.

## Placebos

- Measurement placebo: 100 same-date portfolios per factor/leg with identical size and matched
  industry plus lagged liquidity; size is added only if its PIT estimate passes audit.
- A valid feature row requires at least 80 valid draws and positive placebo standard deviation.
- Strategy convergence nulls are generated jointly across all factors within a draw.
- Full-pipeline pseudo strategies maintain a stable identity through membership, returns, their own
  mature outcome thresholds, and walk-forward predictions.
- Required falsifications: matched portfolio, within-industry permuted ranks, 26/52-week time
  misalignment, and continuous random strategy pipeline.

## Time validation

- Development walk-forward: 2020-01-01 through 2023-12-31.
- Untouched confirmatory period: 2024-01-01 through the last mature label allowed by the raw cutoff.
- Annual expanding retraining is allowed. Once the confirmatory period begins, previous mature
  labels may enter the next annual fit, but protocol definitions and search spaces cannot change.
- Purging is based on real label intervals. Training and validation label intervals may not cross
  into the next evaluation interval.
- Scalers, imputers, bins, models, and optional calibrators are fit only on the corresponding past.

## Lead-time and mechanism analyses

- Leads: 0, 5, and 10 trading days.
- For lead L, an alert is issued at `target_anchor - L`; the outcome begins after target_anchor.
- Membership, C/G/S, thresholds, preprocessing, fitting, and probability handling are rebuilt using
  information available at alert time.
- Only positive lead results support the phrase “early warning.” Lead 0 alone supports only
  “near-term” or “nowcasting indicator.”
- Lagged-C/current-S interactions are mechanism analyses and are not themselves advance alerts.

Crash event study uses independent merged outcome intervals, aligns event zero to the first realized
20-day threshold breach, and displays -8 to +4 weeks for C, S, returns, volatility, liquidity, and
market stress. It is descriptive and cannot replace out-of-sample lead-time evidence.

## Secondary outcomes

- Dynamic LS continuous return and MDD.
- Fixed-membership 20-day return, MDD, and crash using its own mature threshold.
- Long and Short contribution decomposition.
- RankIC breakdown, 5/60-day horizons, 5%/15% thresholds.
- PR-AUC, Brier, calibration, ROC-AUC, and actual crash rate in the top predicted-risk decile.

Secondary analysis families use BH-FDR where multiple related hypotheses are reported.

## Application protocol

Application models are retrained on Active Long Crash, defined from Factor Long wealth relative to
CSI800 wealth. LS probabilities cannot control Long exposure unless a separate transfer experiment
shows stable out-of-sample value.

The confirmatory controller uses historical OOS risk-percentile tiers with active weights
1.00/0.75/0.50/0.25 at the 60/80/90/100 percentile boundaries. Comparators are full active,
fixed lower active, volatility control, M2, M3, and M4. All use identical execution delay and costs.
The fixed-lower comparator has active weight 0.825, the ex-ante expected exposure of the tier rule
under uniform historical risk percentiles. Volatility control estimates annualized volatility from
the latest 60 Factor Long-minus-CSI800 daily returns known by the decision close, requires at least
40 complete observations, targets 10% annualized volatility, and clips active weight to [0.25, 1].
Its weight becomes effective only at the next session open.

Primary application metric: net Information Ratio at 10bp single-side cost. Active MDD and Active
CVaR must not worsen. Five and 20bp are sensitivity assumptions. Application success is judged
mainly by M3/M4 versus M2 under comparable active-exposure budgets.

## Inference

Paired weekly loss differences retain all factors in the same calendar block. Primary moving-block
bootstrap: 10,000 repetitions, 13-week block; 8 and 26 weeks are sensitivity. Report point estimate,
95% interval, and the resampled proportion at or below zero. Report independent crash episode count
and effective calendar blocks rather than treating factor-week rows as independent observations.

## Interpretation constraints

- M3 > M2 does not by itself prove real institutional crowding.
- M4 > M3 supports predictive interaction, not causal unwind.
- If only M2 > M1, generic portfolio risk is informative but C is not incrementally informative.
- If real and pseudo strategies have similar improvement, call the signal portfolio fragility,
  not factor-specific crowding.
- Null and adverse results remain in the final report.

## Freeze checklist

- [ ] PIT data capability report accepted.
- [ ] All windows, fallback rules, match thresholds, and feature versions populated in config.
- [ ] Primary endpoint and sample maturity verified on synthetic data.
- [ ] Common sample-mask and interval-purge tests pass.
- [ ] Full-pipeline placebo identity and audit tables specified.
- [ ] Protocol hash generated and status changed to FROZEN before confirmatory outcomes are viewed.
