"""
Full pipeline re-run with XGBoost as primary model.
Mirrors Rating_Over_Value.ipynb cells 1–55 locally.
Results written to train_results.txt
"""
import ast, pickle, time, sys, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = r"C:\Users\hp\Downloads\Steam_Rating\train_results.txt"
PKL = r"C:\Users\hp\Downloads\Steam_Rating\steam_rating_model.pkl"
CSV = r"C:\Users\hp\Downloads\Steam_Rating\steam_all_games.csv"
HO  = r"C:\Users\hp\Downloads\Steam_Rating\manual_test_games.csv"

lines = []
def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    lines.append(msg)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

log("=" * 60)
log("  STEAM RATING — XGBoost Pipeline")
log("=" * 60)

# ── 1. Load data ────────────────────────────────────────────────────────
log("\n[1/9] Loading CSV...")
t0 = time.time()
df = pd.read_csv(CSV, low_memory=False)
log(f"  Loaded: {df.shape}  ({time.time()-t0:.0f}s)")

df = df[df['type'] == 'game'].copy()
df = df.drop(columns=['type'])

# Compute Rating and filter
df['Rating'] = df['steamspy_positive'] / (df['steamspy_positive'] + df['steamspy_negative']) * 10
df_clean = df[(df['steamspy_positive'] + df['steamspy_negative']) >= 100].copy()
log(f"  After MIN_REVIEWS=100: {len(df_clean):,} games")

# ── 2. EDA preprocessing ────────────────────────────────────────────────
log("\n[2/9] Preprocessing...")
# Rating_class bins
bins   = [0, 7, 8, 9, 10]
labels = ['Unfavorable', 'Mostly Positive', 'Very Positive', 'Overwhelmingly Positive']
df_clean['Rating_class'] = pd.cut(df_clean['Rating'], bins=bins, labels=labels, include_lowest=True)

# Drop groups
drop_group1 = ['steamspy_owners','steamspy_ccu','steamspy_average_forever',
               'steamspy_average_2weeks','steamspy_median_forever','steamspy_median_2weeks',
               'steamspy_positive','steamspy_negative',
               'header_image','capsule_image','capsule_imagev5','background','background_raw','screenshots',
               'website','metacritic_url','support_info_url','support_info_email',
               'detailed_description','about_the_game','short_description','content_descriptors_notes',
               'steam_appid','steamspy_appid']
df_clean = df_clean.drop(columns=[c for c in drop_group1 if c in df_clean.columns])

tag_cols = [c for c in df_clean.columns if c.startswith('steamspy_tags_')]
df_clean[tag_cols] = df_clean[tag_cols].fillna(0)

drop_price = ['price_overview_currency','price_overview_initial','price_overview_final',
              'price_overview_discount_percent','price_overview_initial_formatted',
              'price_overview_final_formatted','steamspy_price','steamspy_discount']
df_clean = df_clean.drop(columns=[c for c in drop_price if c in df_clean.columns])

# OS flags
for col in ['pc_requirements_minimum','mac_requirements_minimum','linux_requirements_minimum']:
    if col in df_clean.columns:
        df_clean[col] = np.where(df_clean[col].notna(), 1, 0)

df_clean['has_metacritic'] = np.where(df_clean['metacritic_score'].notna(), 1, 0)

# ── 3. Parse text columns ────────────────────────────────────────────────
log("[3/9] Parsing genres / categories / publishers / developers...")

def _parse_steam_list(val):
    if pd.isna(val) or str(val).strip() in ('', 'nan', '[]'):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict):
                    desc = item.get('description', item.get('name', ''))
                    if desc: result.append(str(desc))
                else:
                    s = str(item).strip()
                    if s: result.append(s)
            return result
        elif isinstance(parsed, str) and parsed:
            return [parsed]
    except Exception:
        s = str(val).strip()
        return [x.strip() for x in s.split(',') if x.strip()] if ',' in s else ([s] if s else [])
    return []

def _get_primary(val):
    lst = _parse_steam_list(val)
    return lst[0] if lst else 'Unknown'

for col in ['genres','categories','publishers','developers']:
    if col not in df_clean.columns:
        log(f"  WARNING: '{col}' not in dataset — skipping")

df_clean['_genre_list']       = df_clean['genres'].apply(_parse_steam_list) if 'genres' in df_clean.columns else [[]]*len(df_clean)
df_clean['_category_list']    = df_clean['categories'].apply(_parse_steam_list) if 'categories' in df_clean.columns else [[]]*len(df_clean)
df_clean['publisher_primary'] = df_clean['publishers'].apply(_get_primary) if 'publishers' in df_clean.columns else 'Unknown'
df_clean['developer_primary'] = df_clean['developers'].apply(_get_primary) if 'developers' in df_clean.columns else 'Unknown'
log(f"  Sample genres    : {df_clean['_genre_list'].iloc[0]}")
log(f"  Sample publisher : {df_clean['publisher_primary'].iloc[0]}")

# ── 4. Tier classification + hold-out ────────────────────────────────────
log("\n[4/9] Tier classification & hold-out sampling...")
LIVE_TAGS = ['steamspy_tags_Multiplayer','steamspy_tags_Massively Multiplayer',
             'steamspy_tags_Online Co-Op','steamspy_tags_Battle Royale',
             'steamspy_tags_MMORPG','steamspy_tags_PvP']

def classify_game_scale(row):
    price_usd  = (row.get('steamspy_initialprice') or 0) / 100
    metacritic = row['metacritic_score'] if not pd.isna(row['metacritic_score']) else 0
    if price_usd == 0 and any(row.get(t, 0) > 0 for t in LIVE_TAGS if t in row.index):
        return 'Live Service'
    if metacritic >= 75 or price_usd >= 50:
        return 'AAA'
    elif metacritic >= 50 or price_usd >= 20:
        return 'AA'
    return 'Indie'

df_clean['game_scale'] = df_clean.apply(classify_game_scale, axis=1)

SCALES  = ['Indie','AA','AAA','Live Service']
CLASSES = ['Unfavorable','Mostly Positive','Very Positive','Overwhelmingly Positive']

holdout_rows = []
for scale in SCALES:
    for cls in CLASSES:
        bucket = df_clean[(df_clean['game_scale']==scale) & (df_clean['Rating_class']==cls)]
        if len(bucket) == 0:
            log(f"  SKIP: ({scale}, {cls})")
            continue
        holdout_rows.append(bucket.sample(n=1, random_state=42))

df_holdout = pd.concat(holdout_rows).reset_index(drop=True)
log(f"  Hold-out: {len(df_holdout)} games")

# Remove hold-out from df_clean
holdout_original_idx = pd.Index([
    idx
    for _, row in df_holdout.iterrows()
    for idx in df_clean[(df_clean['game_scale']==row['game_scale']) &
                        (df_clean['Rating_class']==row['Rating_class']) &
                        (df_clean['name']==row['name'])].index.tolist()
])
df_clean = df_clean.drop(index=holdout_original_idx).reset_index(drop=True)
log(f"  df_clean after hold-out removal: {len(df_clean):,} games")

# ── 5. Multi-hot encode genres & categories ──────────────────────────────
log("\n[5/9] Multi-hot encoding genres & categories...")
mlb_genre  = MultiLabelBinarizer()
genre_arr  = mlb_genre.fit_transform(df_clean['_genre_list'])
genre_cols = [f'genre_{g.lower().replace(" ","_")}' for g in mlb_genre.classes_]
df_clean   = pd.concat([df_clean, pd.DataFrame(genre_arr, columns=genre_cols, index=df_clean.index)], axis=1)

mlb_cat  = MultiLabelBinarizer()
cat_arr  = mlb_cat.fit_transform(df_clean['_category_list'])
cat_cols = [f'cat_{c.lower().replace(" ","_").replace("-","_")}' for c in mlb_cat.classes_]
df_clean = pd.concat([df_clean, pd.DataFrame(cat_arr, columns=cat_cols, index=df_clean.index)], axis=1)
log(f"  Genre cols: {len(genre_cols)}  Category cols: {len(cat_cols)}")

# ── 6. Mean target encode publisher & developer ──────────────────────────
log("[6/9] Mean target encoding publisher & developer...")
global_mean     = df_clean['Rating'].mean()
publisher_means = df_clean.groupby('publisher_primary')['Rating'].mean()
developer_means = df_clean.groupby('developer_primary')['Rating'].mean()
df_clean['publisher_rating_mean'] = df_clean['publisher_primary'].map(publisher_means).fillna(global_mean)
df_clean['developer_rating_mean'] = df_clean['developer_primary'].map(developer_means).fillna(global_mean)
log(f"  global_mean={global_mean:.4f}  publisher range [{df_clean['publisher_rating_mean'].min():.2f}–{df_clean['publisher_rating_mean'].max():.2f}]")

# ── 7. Feature engineering ────────────────────────────────────────────────
log("\n[7/9] Feature engineering...")

# Release date features
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

release_dates = df_clean['release_date_date'].apply(_parse_release_date)
df_clean['release_year']       = release_dates.dt.year.astype('float')
df_clean['release_month']      = release_dates.dt.month.astype('float')
df_clean['release_quarter']    = release_dates.dt.quarter.astype('float')
df_clean['days_since_release'] = (REFERENCE_DATE - release_dates).dt.days.clip(lower=0).astype('float')
n_parsed = release_dates.notna().sum()
log(f"  release_date parsed: {n_parsed:,}/{len(df_clean):,}  "
    f"year [{int(release_dates.dt.year.min())}-{int(release_dates.dt.year.max())}]  "
    f"days_since_release mean={df_clean['days_since_release'].mean():.0f}")

EXCLUDE = {'Rating','Rating_class','name','game_scale','steamspy_score_rank','AppID',
           '_genre_list','_category_list','publisher_primary','developer_primary',
           'genres','categories','publishers','developers',
           'supported_languages','packages','package_groups',
           'release_date_date','release_date_coming_soon',
           'type','content_descriptors_ids','ratings_steam_germany_descriptors',
           'ratings_dejus_descriptors','ratings_igrs_descriptors'}

df_model = df_clean.copy()
df_model['metacritic_score'] = df_model['metacritic_score'].fillna(0)
df_model['price_usd'] = df_model['steamspy_initialprice'].fillna(0) / 100
df_model['price_log'] = np.log1p(df_model['price_usd'])
df_model = df_model.drop(columns=['steamspy_initialprice'], errors='ignore')

feature_cols = [c for c in df_model.select_dtypes(include='number').columns
                if c not in EXCLUDE]
X = df_model[feature_cols]
y = df_model['Rating']
log(f"  Feature matrix: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log(f"  Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

# Imputer (fit on train only)
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=feature_cols)

# ── 8. Baseline models ────────────────────────────────────────────────────
log("\n[8/9] Baseline models...")

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    mae  = float(mean_absolute_error(y_te, preds))
    r2   = float(r2_score(y_te, preds))
    log(f"  {name:<35} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  ({time.time()-t0:.0f}s)")
    return model, preds, rmse

rf_b  = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
xgb_b = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                     random_state=42, verbosity=0, n_jobs=-1, tree_method='hist')

rf_b,  rf_b_preds,  rf_b_rmse  = evaluate('Random Forest (baseline)',     rf_b,  X_train, y_train, X_test, y_test)
xgb_b, xgb_b_preds, xgb_b_rmse = evaluate('XGBoost (baseline)',           xgb_b, X_train, y_train, X_test, y_test)

_best = min([(rf_b_rmse, rf_b, rf_b_preds, 'Random Forest'),
             (xgb_b_rmse, xgb_b, xgb_b_preds, 'XGBoost')], key=lambda x: x[0])
best_model, best_preds, best_name = _best[1], _best[2], _best[3]

# ── 9. Hyperparameter tuning ──────────────────────────────────────────────
log("\n[9/9] Hyperparameter tuning...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ── XGBoost: Optuna TPE (50 trials × 5-fold) ─────────────────────────────
N_TRIALS = 50
log(f"  Tuning XGBoost with Optuna TPE ({N_TRIALS} trials x 5-fold)...")
t0 = time.time()

def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma':            trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 5.0),
    }
    model = XGBRegressor(**params, random_state=42, verbosity=0,
                         n_jobs=-1, tree_method='hist')
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring='neg_root_mean_squared_error', n_jobs=1)
    return -scores.mean()

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)

best_xgb_params = study.best_params
xgb_best = XGBRegressor(**best_xgb_params, random_state=42, verbosity=0,
                         n_jobs=-1, tree_method='hist')
xgb_best.fit(X_train, y_train)
xgb_preds = xgb_best.predict(X_test)
xgb_rmse  = float(np.sqrt(mean_squared_error(y_test, xgb_preds)))
xgb_mae   = float(mean_absolute_error(y_test, xgb_preds))
xgb_r2    = float(r2_score(y_test, xgb_preds))
log(f"    Best CV RMSE={study.best_value:.4f}  Test RMSE={xgb_rmse:.4f}  "
    f"MAE={xgb_mae:.4f}  R2={xgb_r2:.4f}  ({time.time()-t0:.0f}s)")
log(f"    Best params: {best_xgb_params}")

# ── RandomForest: RandomizedSearchCV (15 candidates × 5-fold) ────────────
def cv_evaluate_rf(name, estimator, param_dist, n_iter=15):
    log(f"  Tuning {name} ({n_iter} candidates x 5-fold)...")
    t0 = time.time()
    search = RandomizedSearchCV(estimator, param_distributions=param_dist,
                                n_iter=n_iter, cv=cv,
                                scoring='neg_root_mean_squared_error',
                                n_jobs=-1, random_state=42, verbose=0)
    search.fit(X_train, y_train)
    best  = search.best_estimator_
    preds = best.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae   = float(mean_absolute_error(y_test, preds))
    r2    = float(r2_score(y_test, preds))
    log(f"    CV RMSE={-search.best_score_:.4f}  Test RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  ({time.time()-t0:.0f}s)")
    log(f"    Best params: {search.best_params_}")
    return best, preds, rmse, mae, r2

rf_params = {
    'n_estimators':      [200, 300, 500],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_leaf':  [1, 2, 5, 10],
    'max_features':      ['sqrt', 'log2', 0.5],
    'min_samples_split': [2, 5, 10],
}
rf_best, rf_preds, rf_rmse, rf_mae, rf_r2 = cv_evaluate_rf(
    'RandomForest', RandomForestRegressor(n_jobs=-1, random_state=42),
    rf_params, n_iter=15
)

# ── Results table ─────────────────────────────────────────────────────────
baseline_rmse = float(np.sqrt(mean_squared_error(y_test, best_preds)))
baseline_mae  = float(mean_absolute_error(y_test, best_preds))
baseline_r2   = float(r2_score(y_test, best_preds))

log("\n" + "=" * 60)
log("  MODEL COMPARISON")
log("=" * 60)
log(f"  {'Model':<35} {'RMSE':>7} {'MAE':>7} {'R2':>7}")
log(f"  {'-'*56}")
log(f"  {'Baseline '+best_name:<35} {baseline_rmse:>7.4f} {baseline_mae:>7.4f} {baseline_r2:>7.4f}")
log(f"  {'Tuned XGBoost':<35} {xgb_rmse:>7.4f} {xgb_mae:>7.4f} {xgb_r2:>7.4f}")
log(f"  {'Tuned RandomForest':<35} {rf_rmse:>7.4f} {rf_mae:>7.4f} {rf_r2:>7.4f}")
log(f"  {'(Old HGBR baseline — 39 features)':<35} {'1.2754':>7} {'0.9694':>7} {'0.2378':>7}")

candidates = {
    f'Baseline {best_name}': (baseline_rmse, best_model,  best_preds),
    'Tuned XGBoost':         (xgb_rmse,      xgb_best,    xgb_preds),
    'Tuned RandomForest':    (rf_rmse,        rf_best,     rf_preds),
}
best_name  = min(candidates, key=lambda k: candidates[k][0])
best_model = candidates[best_name][1]
best_preds = candidates[best_name][2]
best_rmse  = candidates[best_name][0]
log(f"\n  Winner: {best_name}  (RMSE {best_rmse:.4f})")

# ── Feature importance (top 20) ───────────────────────────────────────────
log("\n  Top 20 Feature Importances:")
try:
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).nlargest(20)
    for feat, val in imp.items():
        log(f"    {feat:<45} {val:.4f}")
except AttributeError:
    log("  (feature_importances_ not available for this model type)")

# ── Save PKL ──────────────────────────────────────────────────────────────
THRESHOLDS = {
    'Live Service': {'Exceptional': 9.0, 'Strong': 8.0, 'Promising': 7.0},
    'AAA':          {'Exceptional': 9.5, 'Strong': 8.5, 'Promising': 7.5},
    'AA':           {'Exceptional': 9.0, 'Strong': 8.0, 'Promising': 7.0},
    'Indie':        {'Exceptional': 9.0, 'Strong': 7.5, 'Promising': 6.5},
}

model_payload = {
    'model':            best_model,
    'feature_cols':     feature_cols,
    'imputer':          imputer,
    'thresholds':       THRESHOLDS,
    'best_params':      best_model.get_params(),
    'metrics':          {'rmse': round(best_rmse, 4),
                         'mae':  round(float(mean_absolute_error(y_test, best_preds)), 4),
                         'r2':   round(float(r2_score(y_test, best_preds)), 4)},
    'model_name':       best_name,
    'n_features':       len(feature_cols),
    'train_size':       len(X_train),
    'test_size':        len(X_test),
    'genre_mlb':        mlb_genre,
    'genre_cols':       genre_cols,
    'cat_mlb':          mlb_cat,
    'cat_cols':         cat_cols,
    'publisher_means':  publisher_means,
    'developer_means':  developer_means,
    'global_mean':      global_mean,
}

with open(PKL, 'wb') as f:
    pickle.dump(model_payload, f)

import os
log(f"\n  PKL saved: {os.path.getsize(PKL)/1024:.0f} KB  ({len(feature_cols)} features)")
log("\nDone.")
