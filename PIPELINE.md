# ML Pipeline — NTU Datathon 2026 Track 1 (Smart Campus energy forecasting)

Note: before the ml-pipeline skill was installed (mid-session, 2026-10-03), exploratory prototyping
(quick CV of LightGBM and ridge variants on train.csv only) was already run. Those results are treated as
EDA hypotheses; everything is re-done formally from step 4 under the gates.

## Status
| # | Step | Status | Result |
|---|------|--------|--------|
| 1 | Data inspection | done | 8,000 x 11 train, 3,000 test; 0 duplicates; no leakage suspects; 4 numeric cols 1.4-1.5% missing in train, 6-7% in test |
| 2 | EDA | done | prev_usage r=0.95 (previous hour, legit); per-type daily profiles; temperature effect convex above ~31C; strong train->test covariate shift + heavier missingness in test |
| 3 | Problem contract | done | see below |
| 4 | Data cleaning | done | 0 impossible values found (humidity<=100, occupancy>=0, prev>=0, type matches building, valid day names) in train and test; rules kept as guards in features.clean(); 8000 -> 8000 rows, nothing dropped; NaNs kept for pattern-aware handling |
| 5 | Data engineering | done | single table, one row per building-hour; derived dow (0-6) and building_type from building_id map; no joins |
| 6 | Split | done | guard.split random (no datetime, no groups) 70/15/15 = 5600/1200/1200, seed 42; test frozen. Model selection = 5-fold CV + extrapolation folds on train+val |
| 7 | Feature engineering | done | features.design(): building, hour, dow, month one-hots; type x hour; type x weekend; per-building slopes for temperature/occupancy/prev; humidity; cooling hinge max(T-30,0) -> 296 columns. Missing-pattern handling via imputers |
| 8 | Preprocessing | done | all statistics/imputers fit on training folds only; whole thing stored as one artifact (ridge coefs + imputer boosters + feature code) used identically in the platform notebook |
| 9 | Baseline | done | 5-fold RMSE: mean dummy 18.47; plain linear 3.579 (MAE 2.81, R2 0.963); LightGBM 3.253 |
| 10 | Training | done | structural ridge + GBM imputers 3.040 (MAE 2.41, R2 0.973); residual-GBM, linear_tree, 10+ interaction variants, weighted ridge: none better (noise floor ~3.0). Log: ml_pipeline/runs.jsonl |
| 11 | Tuning | done | alpha {0.1..10} -> 1.0; cooling hinge {29..31,None} -> 29C; +humidity x type; imputer GBM params; offsets for missing inputs learned OOF on train NaN rows |
| 12 | Evaluation | done | CV 3.040 complete / 3.242 with test-like missingness; adversarially-weighted (test-like) 3.08 / 3.51. Public LB (diagnostic): LightGBM 3.575, ridge 3.445, final v2 artifact 3.339 (rows missing both occupancy+prev routed to tree model after public-LB segment analysis) |
| 13 | Error analysis | todo | |
| 14 | Final test | todo | |
| 15 | Deployment (platform notebook + model.pkl) | todo | |
| 16 | Monitoring plan | todo | |

## Step 3 — Prediction contract
- Target: energy_usage for one building-hour (units as given, ~13-144).
- Unit/population: one row = one building (12 buildings, 8 types) at one hour; rows are independent snapshots (no timestamps beyond hour/dow/month).
- Available at prediction time: building_id/type, hour, day_of_week, month, temperature, humidity, occupancy, previous_usage (previous hour's reading). Any of the 4 numeric features may be missing (up to 3 per row in test).
- Objective: minimise squared error on test.csv (public) and a hidden private set that "may contain different observations and edge cases".
- Metric: RMSE primary (leaderboard); MAE and R^2 reported (handbook judging criteria).
- Constraints: inference runs in a sandbox notebook with no internet, no train.csv, shuffled rows and no id column; libraries pinned to requirements-image.txt (sklearn 1.5.2, lightgbm 4.5.0, xgboost 2.1.3).
- Robustness requirement: validate on random K-fold AND on extrapolation folds (train on the middle of temperature/occupancy/prev/humidity, score on the tails) plus simulated test-like missingness.

## Gates
- Gate A: approved 2026-10-03
- Gate B: approved 2026-10-03
- Gate C: approved 2026-10-03

Model rationale:
- traits: regression, 8,000 rows (medium), no datetime column, no groups, strong train->test covariate shift (wider temperature/occupancy/prev tails), 18% of test rows with 1-3 missing inputs
- baseline: mean dummy + plain linear regression on one-hot categoricals and raw numerics
- candidates: structural ridge regression (building/type crosses + cooling hinge) with per-pattern GBM imputers; LightGBM (default tabular strong learner) for comparison; LightGBM on ridge residuals; LightGBM linear_tree; blends
- ruled out: neural nets (small data, cannot verify extrapolation); k-NN (no extrapolation, mixed scales); pure tree models as final model (cannot extrapolate to test's wider ranges - prototype extrapolation-fold RMSE 4.4-7.4 vs ~3.3)
- metric: RMSE primary (5-fold CV + extrapolation folds + simulated test missingness), MAE and R^2 secondary

## Figures
- Figure 01_missingness.png: 4 of 10 columns have missing values; worst is occupancy at 1.5%.
- Figure 02_target_balance.png: energy_usage ranges 13.2 to 144 with median 48.1; a mean-predictor baseline is the number to beat.
- Figure 02_distributions.png: Histograms of 6 numeric features; 0 are strongly skewed (none), which matters for scaling and outliers.
- Figure 02_correlations.png: Strongest correlation with energy_usage: previous_usage (0.95); anything above 0.95 is a leakage suspect.
- Figure 02_train_vs_test_shift.png: test.csv is drawn from a wider distribution than train: temperature 99th pct 35.4 vs 32.5 C, occupancy and previous_usage have heavier tails, so models must extrapolate (trees cannot; linear structure can).
- Figure 02_missing_per_row.png: 18% of test rows miss at least one feature (231 miss 2-3) vs 6% in train (almost always 1), so missing-value handling must be learned from complete rows, not from train's few NaN examples.
- Figure 02_prev_vs_target.png: previous_usage correlates 0.95 with the target; it is the previous hour's reading (known at prediction time, not leakage), and the spread around the diagonal is what other features must explain.
- Figure 02_hourly_profile_by_type.png: Each building type has its own daily shape, so hour must interact with building type rather than enter as one global effect.
- Figure 02_temperature_effect.png: After removing the previous-hour carry-over, energy rises roughly linearly with temperature and gets steeper above ~30-31 C (extra cooling), which matters because test has many more hot hours.
- Figure 04_split_distributions.png: Random 70/15/15 split (5600/1200/1200 rows): target and key features have the same distribution in all three parts, so validation scores are representative; the 15% internal test is frozen until step 14.
- Figure 07_test_missing_patterns.png: test.csv has 15 distinct missing patterns; each gets its own handling (impute the missing inputs from the available ones with models fit on complete training rows, then apply the full structural model), instead of one generic NaN rule.
- Figure 12_model_evaluation.png: Structural ridge beats every baseline (RMSE 3.040 vs LightGBM 3.253, plain linear 3.579); residuals are centred with spread growing with level (noise ~ proportional to usage); it stays accurate on extrapolation folds and degrades gracefully to 3.242 when test-like missingness is injected.

## Notes: public-leaderboard diagnostics (2026-10-03)
- 11 public uploads used as diagnostics (user-approved). Per-row additivity of MSE lets segment effects be separated.
- Finding: on test rows missing BOTH occupancy and previous_usage (43 rows) a NaN-native LightGBM beats ridge+imputation by a large margin; on single-missing rows ridge+imputation is better. Train has no such rows (its 11 multi-missing rows behave normally), so this is a test-only mechanism (structured missingness, corr ~0.2 between columns in test vs 0 in train).
- Risk: decisions informed by public LB may not transfer to the private set; only the large, systematic effect (43 rows, ~-0.86 MSE) was acted on.
- Figure 13_error_analysis.png: Errors are flat across hours and buildings except higher for high-usage science labs and late-evening transition hours; RMSE rises with predicted level (multiplicative noise), bias stays ~0 in every decile; hot/edge segments are slightly worse only because their usage level is higher.
- Step 14 final test: rmse=3.0649 (touch 1, 2026-10-03)
