"""
Hold-out evaluation — looks up the 16 games from the raw CSV to get
genres/categories/publishers/developers, then applies all encodings
from the saved PKL before predicting.
"""
import ast, pickle, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PKL = r"C:\Users\hp\Downloads\Steam_Rating\steam_rating_model.pkl"
CSV = r"C:\Users\hp\Downloads\Steam_Rating\steam_all_games.csv"
HO  = r"C:\Users\hp\Downloads\Steam_Rating\manual_test_games.csv"

# ── Load model payload ──────────────────────────────────────────────────────
with open(PKL, 'rb') as f:
    p = pickle.load(f)

best_model      = p['model']
feature_cols    = p['feature_cols']
imputer         = p['imputer']
mlb_genre       = p['genre_mlb']
genre_cols      = p['genre_cols']
mlb_cat         = p['cat_mlb']
cat_cols        = p['cat_cols']
publisher_means = p['publisher_means']
developer_means = p['developer_means']
global_mean     = p['global_mean']

THRESHOLDS = {
    'Live Service': {'Exceptional': 9.0, 'Strong': 8.0, 'Promising': 7.0},
    'AAA':          {'Exceptional': 9.5, 'Strong': 8.5, 'Promising': 7.5},
    'AA':           {'Exceptional': 9.0, 'Strong': 8.0, 'Promising': 7.0},
    'Indie':        {'Exceptional': 9.0, 'Strong': 7.5, 'Promising': 6.5},
}

def categorize_rating(rating, tier):
    t = THRESHOLDS[tier]
    if rating >= t['Exceptional']: return 'Exceptional'
    elif rating >= t['Strong']:    return 'Strong'
    elif rating >= t['Promising']: return 'Promising'
    return 'At Risk'

# ── Parsing helpers ─────────────────────────────────────────────────────────
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

# ── Load hold-out game names + metadata from saved CSV ─────────────────────
df_ho = pd.read_csv(HO)
holdout_names = df_ho['name'].tolist()
print(f"Hold-out games: {len(holdout_names)}")

# ── Look up the 16 games in the raw CSV ────────────────────────────────────
print("Loading raw CSV to fetch genres/categories/publishers/developers...")
df_raw = pd.read_csv(CSV, low_memory=False)
df_raw = df_raw[df_raw['type'] == 'game'].copy()

df_lookup = df_raw[df_raw['name'].isin(holdout_names)].copy()
# Keep only first match per name (in case of duplicates)
df_lookup = df_lookup.drop_duplicates(subset='name').set_index('name')

# Merge raw text columns back into hold-out
df_ho['genres']           = df_ho['name'].map(df_lookup.get('genres',   pd.Series(dtype=str)) if 'genres'     in df_lookup.columns else pd.Series(dtype=str))
df_ho['categories']       = df_ho['name'].map(df_lookup['categories']   if 'categories'   in df_lookup.columns else pd.Series(dtype=str))
df_ho['publishers']       = df_ho['name'].map(df_lookup['publishers']   if 'publishers'   in df_lookup.columns else pd.Series(dtype=str))
df_ho['developers']       = df_ho['name'].map(df_lookup['developers']   if 'developers'   in df_lookup.columns else pd.Series(dtype=str))

df_ho['_genre_list']       = df_ho['genres'].apply(_parse_steam_list)
df_ho['_category_list']    = df_ho['categories'].apply(_parse_steam_list)
df_ho['publisher_primary'] = df_ho['publishers'].apply(_get_primary)
df_ho['developer_primary'] = df_ho['developers'].apply(_get_primary)

# ── Apply encodings ─────────────────────────────────────────────────────────
# Multi-hot genre
genre_arr = mlb_genre.transform(df_ho['_genre_list'])
df_ho = pd.concat([df_ho, pd.DataFrame(genre_arr, columns=genre_cols, index=df_ho.index)], axis=1)

# Multi-hot category
cat_arr = mlb_cat.transform(df_ho['_category_list'])
df_ho = pd.concat([df_ho, pd.DataFrame(cat_arr, columns=cat_cols, index=df_ho.index)], axis=1)

# Mean target encoding
df_ho['publisher_rating_mean'] = df_ho['publisher_primary'].map(publisher_means).fillna(global_mean)
df_ho['developer_rating_mean'] = df_ho['developer_primary'].map(developer_means).fillna(global_mean)

print("Sample developer lookups:")
for _, r in df_ho[['name','developer_primary','developer_rating_mean']].iterrows():
    print(f"  {r['name'][:35]:<35} dev={r['developer_primary'][:25]:<25} mean={r['developer_rating_mean']:.3f}")

# ── Feature engineering ─────────────────────────────────────────────────────
df_test = df_ho.copy()
df_test['metacritic_score'] = df_test['metacritic_score'].fillna(0)
df_test['price_usd'] = df_test['steamspy_initialprice'].fillna(0) / 100
df_test['price_log'] = np.log1p(df_test['price_usd'])
df_test = df_test.drop(columns=['steamspy_initialprice'], errors='ignore')

X_rows = []
for _, row in df_test.iterrows():
    r = {col: 0 for col in feature_cols}
    for col in feature_cols:
        if col in row.index:
            r[col] = row[col]
    X_rows.append(r)

X_hold = pd.DataFrame(X_rows)[feature_cols]
X_hold = pd.DataFrame(imputer.transform(X_hold), columns=feature_cols)
raw_preds = np.clip(best_model.predict(X_hold), 0, 10)

# ── Build results ───────────────────────────────────────────────────────────
df_results = df_ho[['name','game_scale','Rating','Rating_class']].copy()
df_results['pred_rating']       = [round(float(p), 2) for p in raw_preds]
df_results['error']             = (df_results['pred_rating'] - df_results['Rating']).round(2)
df_results['predicted_outlook'] = [
    categorize_rating(float(p), str(s))
    for p, s in zip(raw_preds, df_ho['game_scale'])
]

TIER_ORDER = {'Indie':0,'AA':1,'AAA':2,'Live Service':3}
df_results['_ts'] = df_results['game_scale'].map(TIER_ORDER)
df_results = df_results.sort_values('_ts').drop(columns='_ts').reset_index(drop=True)

# ── Print table ─────────────────────────────────────────────────────────────
print()
print('=' * 95)
print('  HOLD-OUT TEST RESULTS  ([OK]=|err|<=0.5  [~ ]=|err|<=1.0  [ERR]=|err|>1.0)')
print('=' * 95)
print(f"{'#':>2}  {'Game':<40} {'Tier':<14} {'Actual':>7} {'Predicted':>9} {'Error':>7}  {'Steam Label':<24} {'Pred Outlook'}")
print('-' * 95)
for i, row in df_results.iterrows():
    err  = abs(row['error'])
    flag = '[OK ]' if err <= 0.5 else '[~  ]' if err <= 1.0 else '[ERR]'
    print(f"{i+1:>2}  {str(row['name']):<40} {str(row['game_scale']):<14} "
          f"{row['Rating']:>7.2f} {row['pred_rating']:>9.2f} "
          f"{row['error']:>+7.2f}  {flag} {str(row['Rating_class']):<20} {row['predicted_outlook']}")
print('=' * 95)

# ── Metrics ─────────────────────────────────────────────────────────────────
h_rmse    = float(np.sqrt(((df_results['pred_rating'] - df_results['Rating'])**2).mean()))
h_mae     = float((df_results['pred_rating'] - df_results['Rating']).abs().mean())
within_05 = int((df_results['error'].abs() <= 0.5).sum())
within_10 = int((df_results['error'].abs() <= 1.0).sum())

print()
print('  METRICS')
print(f'  Games tested        : {len(df_results)}')
print(f'  RMSE (hold-out)     : {h_rmse:.4f}')
print(f'  MAE  (hold-out)     : {h_mae:.4f}')
print(f'  Within 0.5 pts      : {within_05}/{len(df_results)}')
print(f'  Within 1.0 pts      : {within_10}/{len(df_results)}')
print(f'  (Test-set RMSE      : {p["metrics"]["rmse"]}  -- on {p["test_size"]:,} games)')

print()
print('  Predicted Outlook distribution:')
for label in ['Exceptional','Strong','Promising','At Risk']:
    n = int((df_results['predicted_outlook'] == label).sum())
    print(f'    {label:<12} {"#"*n}  {n}')

print()
print('  RMSE by tier:')
for tier in ['Indie','AA','AAA','Live Service']:
    grp = df_results[df_results['game_scale'] == tier]
    if grp.empty: continue
    print(f'    {tier:<14} RMSE={np.sqrt(((grp["pred_rating"]-grp["Rating"])**2).mean()):.3f}'
          f'  MAE={(grp["pred_rating"]-grp["Rating"]).abs().mean():.3f}')
