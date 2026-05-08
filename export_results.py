"""
Generates full_results.txt — complete test results report.
"""
import ast, pickle, sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT        = r"C:\Users\hp\Downloads\Steam_Rating\full_results.txt"
PKL        = r"C:\Users\hp\Downloads\Steam_Rating\steam_rating_model.pkl"
CSV        = r"C:\Users\hp\Downloads\Steam_Rating\steam_all_games.csv"
HO         = r"C:\Users\hp\Downloads\Steam_Rating\manual_test_games.csv"
EXTRA_TAGS = r"C:\Users\hp\Downloads\Steam_Rating\steamspy_extra_tags.csv"

lines = []
def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    lines.append(msg)

def save():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ── Load payload ────────────────────────────────────────────────────────────
with open(PKL, "rb") as f:
    p = pickle.load(f)

best_model      = p["model"]
model_rf        = p.get("model_rf")
is_ensemble     = p.get("is_ensemble", False)
feature_cols    = p["feature_cols"]
imputer         = p["imputer"]
mlb_genre       = p["genre_mlb"]
genre_cols      = p["genre_cols"]
mlb_cat         = p["cat_mlb"]
cat_cols        = p["cat_cols"]
publisher_means = p["publisher_means"]
developer_means = p["developer_means"]
global_mean     = p["global_mean"]

THRESHOLDS = {
    "Live Service": {"Exceptional": 9.0, "Strong": 8.0, "Promising": 7.0},
    "AAA":          {"Exceptional": 9.5, "Strong": 8.5, "Promising": 7.5},
    "AA":           {"Exceptional": 9.0, "Strong": 8.0, "Promising": 7.0},
    "Indie":        {"Exceptional": 9.0, "Strong": 7.5, "Promising": 6.5},
}

def categorize(rating, tier):
    t = THRESHOLDS[tier]
    if rating >= t["Exceptional"]: return "Exceptional"
    elif rating >= t["Strong"]:    return "Strong"
    elif rating >= t["Promising"]: return "Promising"
    return "At Risk"

def parse_list(val):
    if pd.isna(val) or str(val).strip() in ("", "nan", "[]"):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict):
                    desc = item.get("description", item.get("name", ""))
                    if desc: result.append(str(desc))
                else:
                    s = str(item).strip()
                    if s: result.append(s)
            return result
        elif isinstance(parsed, str) and parsed:
            return [parsed]
    except:
        s = str(val).strip()
        return [x.strip() for x in s.split(",") if x.strip()] if "," in s else ([s] if s else [])
    return []

def get_primary(val):
    lst = parse_list(val)
    return lst[0] if lst else "Unknown"

# ── Load + encode hold-out ───────────────────────────────────────────────────
df_ho  = pd.read_csv(HO)
df_raw = pd.read_csv(CSV, low_memory=False)
df_raw = df_raw[df_raw["type"] == "game"].copy()
df_lkp = df_raw[df_raw["name"].isin(df_ho["name"])].drop_duplicates("name").set_index("name")

# Merge extra tags into df_lkp so they can be looked up by game name
_et = pd.read_csv(EXTRA_TAGS)
_et["AppID"] = _et["AppID"].astype(int)
_extra_tag_cols = [c for c in _et.columns if c != "AppID"]
df_lkp = df_lkp.reset_index().merge(_et, on="AppID", how="left").set_index("name")
df_lkp[_extra_tag_cols] = df_lkp[_extra_tag_cols].fillna(0).astype(int)

for col in ["genres", "categories", "publishers", "developers", "release_date_date"]:
    df_ho[col] = df_ho["name"].map(df_lkp[col] if col in df_lkp.columns else pd.Series(dtype=str))

for col in _extra_tag_cols:
    df_ho[col] = df_ho["name"].map(df_lkp[col]).fillna(0).astype(int)

df_ho["_genre_list"]       = df_ho["genres"].apply(parse_list)
df_ho["_category_list"]    = df_ho["categories"].apply(parse_list)
df_ho["publisher_primary"] = df_ho["publishers"].apply(get_primary)
df_ho["developer_primary"] = df_ho["developers"].apply(get_primary)

genre_arr = mlb_genre.transform(df_ho["_genre_list"])
df_ho = pd.concat([df_ho, pd.DataFrame(genre_arr, columns=genre_cols, index=df_ho.index)], axis=1)
cat_arr = mlb_cat.transform(df_ho["_category_list"])
df_ho = pd.concat([df_ho, pd.DataFrame(cat_arr, columns=cat_cols, index=df_ho.index)], axis=1)
df_ho["publisher_rating_mean"] = df_ho["publisher_primary"].map(publisher_means).fillna(global_mean)
df_ho["developer_rating_mean"] = df_ho["developer_primary"].map(developer_means).fillna(global_mean)

df_test = df_ho.copy()
df_test["metacritic_score"] = df_test["metacritic_score"].fillna(0)
df_test["price_usd"] = df_test["steamspy_initialprice"].fillna(0) / 100
df_test["price_log"] = np.log1p(df_test["price_usd"])
df_test = df_test.drop(columns=["steamspy_initialprice"], errors="ignore")

# Release date features (must mirror train_xgboost.py)
REFERENCE_DATE = pd.Timestamp('2026-04-23')
def _parse_release_date(val):
    if pd.isna(val): return pd.NaT
    s = str(val).strip()
    for fmt in ('%d %b, %Y', '%b %d, %Y'):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.NaT

_rd = df_test["release_date_date"].apply(_parse_release_date)
df_test["release_year"]       = _rd.dt.year.astype("float")
df_test["release_month"]      = _rd.dt.month.astype("float")
df_test["release_quarter"]    = _rd.dt.quarter.astype("float")
df_test["days_since_release"] = (REFERENCE_DATE - _rd).dt.days.clip(lower=0).astype("float")


X_rows = []
for _, row in df_test.iterrows():
    r = {col: 0 for col in feature_cols}
    for col in feature_cols:
        if col in row.index:
            r[col] = row[col]
    X_rows.append(r)

X_hold = pd.DataFrame(X_rows)[feature_cols]
X_hold = pd.DataFrame(imputer.transform(X_hold), columns=feature_cols)
if is_ensemble and model_rf is not None:
    raw_preds = np.clip((best_model.predict(X_hold) + model_rf.predict(X_hold)) / 2, 0, 10)
else:
    raw_preds = np.clip(best_model.predict(X_hold), 0, 10)

df_r = df_ho[["name", "game_scale", "Rating", "Rating_class"]].copy()
df_r["pred_rating"]       = [round(float(p), 2) for p in raw_preds]
df_r["error"]             = (df_r["pred_rating"] - df_r["Rating"]).round(2)
df_r["abs_error"]         = df_r["error"].abs()
df_r["predicted_outlook"] = [categorize(float(p), str(s)) for p, s in zip(raw_preds, df_ho["game_scale"])]
df_r["developer"]         = df_ho["developer_primary"].values
df_r["dev_mean"]          = df_ho["developer_rating_mean"].round(3).values

TIER_ORDER = {"Indie": 0, "AA": 1, "AAA": 2, "Live Service": 3}
df_r["_t"] = df_r["game_scale"].map(TIER_ORDER)
df_r = df_r.sort_values("_t").drop(columns="_t").reset_index(drop=True)

# ── Write report ─────────────────────────────────────────────────────────────
W = 105
log("=" * W)
log("  STEAM RATING PREDICTION — FULL TEST RESULTS")
log("=" * W)

# Model info
log("")
log("  MODEL")
log(f"  {'Name':<30} {p['model_name']}")
log(f"  {'Type':<30} {type(best_model).__name__}")
log(f"  {'Features':<30} {p['n_features']}")
log(f"  {'Train size':<30} {p['train_size']:,} games")
log(f"  {'Test size':<30} {p['test_size']:,} games")
log("")
log("  BEST HYPERPARAMETERS")
params = p["best_params"]
for k in ["n_estimators","learning_rate","max_depth","min_child_weight",
          "subsample","colsample_bytree","gamma","reg_alpha","reg_lambda"]:
    if k in params:
        log(f"  {'  '+k:<30} {params[k]}")

# Test-set metrics
log("")
log("=" * W)
log("  TEST-SET METRICS  (4,417 games, 80/20 split)")
log("=" * W)
log(f"  {'RMSE':<20} {p['metrics']['rmse']}")
log(f"  {'MAE':<20} {p['metrics']['mae']}")
log(f"  {'R2':<20} {p['metrics']['r2']}")

# Hold-out results table
log("")
log("=" * W)
log("  HOLD-OUT TEST RESULTS  (16 games — 1 per Tier x Rating Class)")
log("  Legend: OK=|error|<=0.5   ~ =|error|<=1.0   ERR=|error|>1.0")
log("=" * W)
log(f"  {'#':>2}  {'Game':<42} {'Tier':<14} {'Actual':>7} {'Pred':>7} {'Error':>7}  {'Flag':<6} {'Steam Label':<26} {'Pred Outlook':<14} {'Developer'}")
log("  " + "-" * 101)

for i, row in df_r.iterrows():
    err  = row["abs_error"]
    flag = "OK  " if err <= 0.5 else "~   " if err <= 1.0 else "ERR "
    log(f"  {i+1:>2}  {str(row['name']):<42} {str(row['game_scale']):<14}"
        f" {row['Rating']:>7.2f} {row['pred_rating']:>7.2f} {row['error']:>+7.2f}"
        f"  [{flag}] {str(row['Rating_class']):<26} {row['predicted_outlook']:<14} {row['developer']}")

log("  " + "=" * 101)

# Summary metrics
h_rmse    = float(np.sqrt((df_r["error"]**2).mean()))
h_mae     = float(df_r["abs_error"].mean())
within_05 = int((df_r["abs_error"] <= 0.5).sum())
within_10 = int((df_r["abs_error"] <= 1.0).sum())

log("")
log("  HOLD-OUT METRICS")
log(f"  {'RMSE':<25} {h_rmse:.4f}")
log(f"  {'MAE':<25} {h_mae:.4f}")
log(f"  {'Within 0.5 pts':<25} {within_05}/16  ({within_05/16*100:.0f}%)")
log(f"  {'Within 1.0 pts':<25} {within_10}/16  ({within_10/16*100:.0f}%)")

# Per-tier breakdown
log("")
log("  RMSE BY TIER")
log(f"  {'Tier':<16} {'RMSE':>7} {'MAE':>7} {'Games':>6}  {'OK (<=0.5)':>10}  {'~  (<=1.0)':>10}  {'ERR (>1.0)':>10}")
log("  " + "-" * 75)
for tier in ["Indie", "AA", "AAA", "Live Service"]:
    g = df_r[df_r["game_scale"] == tier]
    if g.empty: continue
    rmse_t = float(np.sqrt((g["error"]**2).mean()))
    mae_t  = float(g["abs_error"].mean())
    ok  = int((g["abs_error"] <= 0.5).sum())
    mid = int((g["abs_error"] <= 1.0).sum()) - ok
    err = int((g["abs_error"] > 1.0).sum())
    log(f"  {tier:<16} {rmse_t:>7.3f} {mae_t:>7.3f} {len(g):>6}  {ok:>10}  {mid:>10}  {err:>10}")

# Outlook distribution
log("")
log("  PREDICTED OUTLOOK DISTRIBUTION")
for label in ["Exceptional", "Strong", "Promising", "At Risk"]:
    n   = int((df_r["predicted_outlook"] == label).sum())
    bar = "#" * n
    log(f"  {label:<14} {bar:<20} {n}")

# Feature importance
log("")
log("  TOP 20 FEATURE IMPORTANCES")
log(f"  {'Feature':<48} {'Importance':>10}  {'Cumulative':>10}")
log("  " + "-" * 72)
imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)  # always XGBoost
cumsum = 0.0
for feat, val in imp.head(20).items():
    cumsum += val
    log(f"  {feat:<48} {val:>10.4f}  {cumsum:>9.1%}")

log("")
log("=" * W)
log("  END OF REPORT")
log("=" * W)

save()
print(f"\nSaved to: {OUT}")
