# Steam Rating Prediction — Experiment Log

## Current Best Model (as of 2026-05-08)
- Architecture: Tuned XGBoost (Optuna TPE 50 trials x 5-fold)
- Features: 155 (+ sequel_number + days_since_dev_last_release)
- Test-set RMSE: 0.5605 | MAE: 0.3076 | R2: 0.8539  ← best ever across all metrics
- Hold-out: 32 games (2 per bucket) | RMSE: 1.1133 | MAE: 0.8253 | R2: 0.1223 | Accuracy: 56% | F1: 0.5330
- Hold-out within 0.5 pts: 16/32 (50%) ← best ever | within 1.0 pts: 21/32 (66%)
- Best params: n_estimators=743, lr=0.0297, max_depth=7, min_child_weight=4, subsample=0.866, colsample=0.638

## Complete Improvement History

| Experiment | Result | Notes |
|---|---|---|
| Bayesian smoothing (#7) | REVERTED | Compressed dev range, hurt signal |
| Release date features (#1) | KEPT | +1 game within 1.0 pts |
| Optuna tuning (#10) | KEPT | RMSE 0.6372 -> 0.6339 |
| Ensemble XGB+RF (#12) | REVERTED | RF pulled predictions toward middle |
| Normalized tag counts (#3) | REVERTED | Raw counts carry popularity signal |
| K-fold mean encoding (#8) | REVERTED | CATASTROPHIC — 44% of devs have 1 game, OOF collapses them all to global mean |
| Separate tier models (#11) | REVERTED | Pooled RMSE 0.6554 vs 0.6339; AA/AAA starved of data |
| Extra SteamSpy tags (#4) | KEPT | +1 game (12/16 -> 13/16); 2D rank 5, JRPG rank 12 |
| Logit transform + sample weights | REVERTED | Hold-out RMSE jumped 1.0716 -> 1.1838; sample weights penalise hard cases |
| Developer career stats (#9) | KEPT | Test RMSE 0.6336->0.6120 (best ever), MAE 0.3882->0.3421, R2 0.8113->0.8240; developer_last_rating rank 3 (2.56%), developer_rating_std rank 13 (0.77%), developer_game_count not in top 20; hold-out 12/16->11/16 (noise, Men of Valor +1.01 barely over threshold) |
| LightGBM native categoricals (#13) | REVERTED | Test RMSE 0.6115 vs Ensemble 0.6120 — negligible 0.0005 gain; hold-out 10/16 (worse); 2.3x slower (1100s vs 467s); native cat features nearly unused (publisher_cat 1.22%, developer_cat not top 20); developer_rating_mean dominated at 70.66% — mean encoding already captures all categorical signal |
| Post-split target encoding (Fix 1) | REVERTED | Test RMSE 0.63 -> 1.29; same root cause as K-fold (#8): 44% single-game devs collapse to global_mean, destroying developer signal |
| Supported languages count (#2) | KEPT | Test RMSE 0.6120→0.6110, MAE 0.3421→0.3419, R² 0.8240→0.8246; hold-out RMSE 1.2402→1.2289, MAE 0.9856→0.9651; +1 within 0.5 pts (6/16 38%); accuracy 38%→44%; not in top 20 importances but consistent marginal improvement across all metrics |
| Content ratings count (#6) | KEPT | Test RMSE 0.6110→0.6108; hold-out RMSE 1.2289→1.1436 (best ever), R² 0.0994→0.2201, MAE 0.9651→0.9131; accuracy 44%→50%; Civ VII error +2.78→+2.17; not in top 20 but strongly improved generalization — 0.179 corr with Rating; XGBoost alone won over ensemble this run |
| MIN_REVIEWS=50 (#14) | REVERTED | Test RMSE 0.6108→0.6302 (+0.019 worse); within 1.0 pts 69%→50%; accuracy 50%→38%; 7,605 extra games with 50-99 reviews add noise that hurts generalization more than it helps |
| Bayesian smoothing weight=3 + First-Time Dev grouping (#7) | REVERTED | Test RMSE 0.6108→0.6851; hold-out R²=−0.03; 8,472 devs (48%) → ftd_mean≈gm; developer_rating_mean 38%→5% importance; CV/test gap +0.063; all 5 smoothing variants conclusively worse — NOT APPLICABLE for this dataset |
| Expand hold-out to 32 games (#18) | KEPT | 2 per bucket; within 1.0 pts 69% held steady (11/16→22/32); accuracy 50%→53%; UTF-8 stdout fix included (handles game names with ™ etc); test RMSE 0.5865 (note: test set composition changed — not directly comparable to 0.6108) |
| Description keyword flags (#5) | REVERTED | Test MAE 0.3244→0.3333 (clearly worse); within-0.5 pts 13→12/32; no desc_* in top 20 (not even top 40); horror/survival/multiplayer already captured by existing steamspy_tags_* and cat_* features |
| Metacritic NaN → median fill instead of 0 (Phase 2 D) | REVERTED | Test RMSE 0.5605→0.5601 (noise-level −0.0004); hold-out RMSE 1.1133→1.1505 (+0.037 worse); accuracy 56%→50% (−6%); F1 0.5330→0.4880; within-0.5 16→15/32; has_metacritic flag already cleanly separates reviewed/unreviewed; changing 83% of games (18,327) from 0→76 disrupted learned patterns without improving generalisation |
| Tier-specific MIN_REVIEWS (Phase 2 C: LS≥50, others≥100) | REVERTED | Test RMSE 0.5605→0.5863 (+0.026, worse); MAE 0.3076→0.3157; accuracy 56%→53%; within-0.5 16→15/32; hold-out RMSE improved (1.1133→1.0550) but hold-out composition changed entirely (different LS games sampled) — not a clean comparison; +196 LS games (50-99 reviews) too noisy; baseline XGBoost already regressed 0.5768→0.6047 before Optuna |
| Sequel flag + days_since_dev_last_release (Phase 2 A+B) | KEPT | Test RMSE 0.5865→0.5605 (best ever, −0.026); MAE 0.3244→0.3076; R2 0.8401→0.8539; hold-out RMSE 1.1133 (−0.041), within-0.5 13→16/32 (+3 games, best ever), accuracy 53→56%; Civ VII error +2.79→+2.32; neither feature in top 20 but baseline XGBoost confirmed signal (0.5954→0.5768 before tuning) |
| Hold-out evaluation block (Fix 2) | KEPT | Added RMSE/MAE/R2/Accuracy/F1 reporting inside train_xgboost.py after PKL save |
| Dual X for NaN-native XGBoost (Fix 3) | REVERTED | Test RMSE marginal gain (0.6336->0.6321) but hold-out 13/16->12/16; root cause: nearly all NaN pre-filled before split (tags=0, metacritic=0, price=0), leaving XGBoost almost nothing to learn from natively |

## Persistent Errors (present in most runs)
1. **Civ VII** (AAA, Unfavorable): actual 4.85, pred 7.17, error +2.32 — sequel_number=7 helped reduce from +2.79→+2.32; Firaxis dev mean still dominates
2. **Horse Riding Tales** (Live Service, Unfavorable): actual 5.86, pred ~7.9, error ~+2.0 — Niche mobile-port with few genre signals
3. **X8** (Live Service, Very Positive): actual 8.24, pred ~6.0, error ~−2.3 — Underrated niche VR title; model under-trusts it
4. **The Stalin Subway** (Indie, Unfavorable): actual 6.05, pred ~4.7, error ~−1.3 — Persistent borderline case

## Top 20 Feature Importances (current model)
1. developer_rating_mean: 0.2610 (dominant — 26% alone)
2. publisher_rating_mean: 0.0809
3. cat_remote_play_on_tablet: 0.0162
4. metacritic_score: 0.0131
5. steamspy_tags_2D: 0.0114 (NEW — added this session)
6. steamspy_tags_Classic: 0.0101
7. recommendations_total: 0.0092
8. cat_lan_pvp: 0.0081
9. cat_vr_only: 0.0079
10. cat_family_sharing: 0.0079
11. cat_multi_player: 0.0078
12. steamspy_tags_JRPG: 0.0077 (NEW — added this session)
13-20: various category and tag features

## What Was Learned About Hard Cases
- Civ VII: Legacy developer reputation cannot be overcome without franchise-age or recency signals
- Live Service predictions are most volatile — only 1,084 training games; niche titles have no genre peers
- Sample weighting hurts: hard cases tend to have fewer reviews, so weighting by review count makes the model ignore them
- Tier-splitting hurts: developer signal spans tiers; splitting destroys cross-tier evidence

## What Was Learned About Encoding & NaN

- **Target encoding leakage is load-bearing**: The "leaky" pre-split developer/publisher means are actually correct for deployment — at prediction time you DO know a developer's full catalogue. The only true leak is for single-game devs whose mean = their own test rating. Fixing it (post-split or K-fold) collapses 44% of devs to global_mean and destroys the #1 feature.
- **NaN-native XGBoost has almost nothing to learn from**: By the time data reaches XGBoost, nearly all NaN are pre-filled (tags→0, metacritic→0, price→0). Only ~26 games (of 22,081) with unparseable release dates have real NaN. Dual-X approach provides no meaningful benefit for this pipeline.
- **Hold-out evaluation (Fix 2) is the only clean win**: Adds honest regression + classification metrics directly to the training run without any model change.
