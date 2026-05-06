# Steam Game Rating Prediction

Predict a Steam game's rating score (0–10) from metadata using XGBoost, then classify the result into a tier-specific **Predicted Outlook** label.

## Overview

**Target:** `Rating = steamspy_positive / (steamspy_positive + steamspy_negative) × 10`

The model is trained on 22,097 games (filtered to ≥ 100 reviews) from a 163,918-row Steam dataset. It predicts the rating a new game is likely to receive based on price, genre, category, developer/publisher track record, and SteamSpy tags.

## Results

| Metric | Value |
|---|---|
| Model | XGBoost (`tree_method=hist`) |
| Features | 124 |
| Test-set RMSE | 0.6372 |
| Test-set R² | 0.8092 |
| Hold-out RMSE (16 games) | 1.0732 |
| Hold-out within 1.0 pts | 12/16 (75%) |

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
| Platform / ratings | OS flags, age ratings, recommendations | Raw |

**Top features by importance:**
1. `developer_rating_mean` — 36.4%
2. `publisher_rating_mean` — 8.2%
3. `metacritic_score`, `recommendations_total`, genre/category tags

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
