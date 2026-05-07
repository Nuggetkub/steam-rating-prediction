# Steam Game Rating Prediction

Predict a Steam game's rating score (0–10) from metadata using XGBoost, then classify the result into a tier-specific **Predicted Outlook** label.

## Overview

**Target:** `Rating = steamspy_positive / (steamspy_positive + steamspy_negative) × 10`

The model is trained on 22,097 games (filtered to ≥ 100 reviews) from a 163,918-row Steam dataset. It predicts the rating a new game is likely to receive based on price, genre, category, developer/publisher track record, and SteamSpy tags.

## Results

| Metric | Value |
|---|---|
| Model | XGBoost (`tree_method=hist`) |
| Features | 128 |
| Test-set RMSE | 0.6376 |
| Test-set MAE | 0.3883 |
| Test-set R² | 0.8090 |
| Hold-out RMSE (16 games) | 1.1075 |
| Hold-out MAE (16 games) | 0.8681 |
| Hold-out within 0.5 pts | 5/16 (31%) |
| Hold-out within 1.0 pts | 13/16 (81%) |

### Hold-Out Results per Game

| # | Game | Tier | Actual | Predicted | Error | Pred Outlook |
|---|---|---|---|---|---|---|
| 1 | The Stalin Subway | Indie | 6.05 | 5.36 | -0.69 | At Risk |
| 2 | Pixel Puzzles 2: Anime | Indie | 7.44 | 7.90 | +0.46 ✅ | Strong |
| 3 | Hmmsim Metro | Indie | 8.95 | 7.98 | -0.97 | Strong |
| 4 | Meet Your Oshi | Indie | 9.04 | 9.75 | +0.71 | Exceptional |
| 5 | Men of Valor | AA | 6.88 | 7.62 | +0.74 | Promising |
| 6 | Mainlining | AA | 7.77 | 7.82 | +0.05 ✅ | Promising |
| 7 | Psychedelica of the Black Butterfly | AA | 8.60 | 7.67 | -0.93 | Promising |
| 8 | Fate Seeker | AA | 9.08 | 8.30 | -0.78 | Strong |
| 9 | Sid Meier's Civilization VII | AAA | 4.85 | 7.47 | +2.62 | At Risk |
| 10 | Zanki Zero: Last Beginning | AAA | 7.75 | 7.96 | +0.21 ✅ | Promising |
| 11 | Undernauts: Labyrinth of Yomi | AAA | 8.28 | 8.66 | +0.38 ✅ | Strong |
| 12 | 30XX | AAA | 9.10 | 9.14 | +0.04 ✅ | Strong |
| 13 | Horse Riding Tales | Live Service | 5.86 | 7.66 | +1.80 | Promising |
| 14 | Shadowverse CCG | Live Service | 7.37 | 8.05 | +0.68 | Strong |
| 15 | X8 | Live Service | 8.24 | 6.23 | -2.01 | At Risk |
| 16 | The Lab | Live Service | 9.49 | 8.67 | -0.82 | Strong |

> ✅ = within 0.5 pts · hold-out RMSE is higher than test-set RMSE as expected on only 16 games

### RMSE by Tier

| Tier | RMSE | MAE | Within 0.5 | Within 1.0 | Over 1.0 |
|---|---|---|---|---|---|
| Indie | 0.730 | 0.708 | 1 | 3 | 0 |
| AA | 0.711 | 0.625 | 1 | 3 | 0 |
| AAA | 1.328 | 0.812 | 3 | 0 | 1 |
| Live Service | 1.450 | 1.328 | 0 | 2 | 2 |

## Game Tiers

Games are classified into 4 tiers before applying rating thresholds:

| Tier | Criteria |
|---|---|
| **Live Service** | Price = $0 AND has online multiplayer tag |
| **AAA** | Metacritic ≥ 75 OR price ≥ $50 |
| **AA** | Metacritic ≥ 50 OR price ≥ $20 (not AAA) |
| **Indie** | Everything else |

## Predicted Outlook Labels

Each tier uses different thresholds to assign a **Predicted Outlook**:

| Tier | Exceptional | Strong | Promising | At Risk |
|---|---|---|---|---|
| Live Service | ≥ 9.0 | ≥ 8.0 | ≥ 7.0 | < 7.0 |
| AAA | ≥ 9.5 | ≥ 8.5 | ≥ 7.5 | < 7.5 |
| AA | ≥ 9.0 | ≥ 8.0 | ≥ 7.0 | < 7.0 |
| Indie | ≥ 9.0 | ≥ 7.5 | ≥ 6.5 | < 6.5 |

These labels are **model-specific forecasts**, not Steam's official review summaries.

## Feature Engineering

| Group | Features | Encoding |
|---|---|---|
| Price | `price_usd`, `price_log` | Raw + log transform |
| SteamSpy tags | 20 columns | Raw vote counts |
| Genres | 25 columns (`genre_*`) | Multi-hot (MultiLabelBinarizer) |
| Categories | 58 columns (`cat_*`) | Multi-hot (MultiLabelBinarizer) |
| Publisher | `publisher_rating_mean` | Mean target encoding |
| Developer | `developer_rating_mean` | Mean target encoding |
| Metacritic | `metacritic_score`, `has_metacritic` | Raw + flag |
| Release date | `release_year`, `release_month`, `release_quarter`, `days_since_release` | Parsed from `release_date_date` |
| Platform / ratings | OS flags, age ratings, recommendations | Raw |

**Top features by importance:**
1. `developer_rating_mean` — 38.0%
2. `publisher_rating_mean` — 5.6%
3. `metacritic_score`, `recommendations_total`, `days_since_release`, genre/category tags

## Files

| File | Description |
|---|---|
| `Rating_Over_Value.ipynb` | Main notebook (EDA → feature engineering → model → evaluation) |
| `steam_rating_model.pkl` | Saved model + all encoders (3.6 MB) |
| `manual_test_games.csv` | 16-game hold-out test set (1 per tier × rating class) |
| `train_xgboost.py` | Standalone local training script |
| `run_holdout.py` | Standalone hold-out evaluation script |

## Quickstart

```python
import pickle
import numpy as np

with open("steam_rating_model.pkl", "rb") as f:
    payload = pickle.load(f)

model        = payload["model"]
feature_cols = payload["feature_cols"]
imputer      = payload["imputer"]

# Build a feature row (missing columns default to 0)
row = {col: 0 for col in feature_cols}
row["price_usd"]        = 29.99
row["metacritic_score"] = 72
row["has_metacritic"]   = 1
row["developer_rating_mean"] = 8.2   # from payload["developer_means"]["Studio Name"]
row["publisher_rating_mean"] = 7.9

import pandas as pd
X = pd.DataFrame([row])[feature_cols]
X = pd.DataFrame(imputer.transform(X), columns=feature_cols)
predicted_rating = float(np.clip(model.predict(X)[0], 0, 10))
print(f"Predicted rating: {predicted_rating:.2f} / 10")
```

## Requirements

```
pandas
numpy
scikit-learn
xgboost
```

## Notes

- `steam_all_games.csv` (1.6 GB raw data) is excluded from this repo via `.gitignore`
- The notebook (`Rating_Over_Value.ipynb`) is designed to run on Google Colab with the CSV loaded from Google Drive
- `train_xgboost.py` runs the full pipeline locally given the raw CSV
