"""
Fetches additional SteamSpy tag presence for games in our dataset.
Uses request=tag endpoint (returns all AppIDs with that tag).
Creates binary 0/1 features: 1 = game appears in tag results.
Saves steamspy_extra_tags.csv with AppID + one binary column per tag.
"""
import sys, time, requests, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"C:\Users\hp\Downloads\Steam_Rating\steamspy_extra_tags.csv"
CSV = r"C:\Users\hp\Downloads\Steam_Rating\steam_all_games.csv"

TAGS_TO_FETCH = [
    "RPG", "Puzzle", "Horror", "Adventure", "Simulation", "Platformer",
    "Roguelike", "Roguelite", "Visual Novel", "Anime", "JRPG",
    "Open World", "Sandbox", "Tower Defense", "Metroidvania",
    "Racing", "2D", "Singleplayer", "Co-op", "Atmospheric",
]

print("Loading AppIDs from CSV...")
df_raw = pd.read_csv(CSV, low_memory=False)
df_raw = df_raw[df_raw["type"] == "game"].copy()
all_appids = set(df_raw["AppID"].dropna().astype(int).tolist())
print(f"  {len(all_appids):,} game AppIDs in dataset")

# tag_appids[tag] = set of AppIDs that have this tag
tag_appids = {}

for tag in TAGS_TO_FETCH:
    url = f"https://steamspy.com/api.php?request=tag&tag={requests.utils.quote(tag)}"
    print(f"  Fetching: {tag!r}")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        games = r.json()
        appids = {int(k) for k in games.keys()}
        matched = appids & all_appids
        tag_appids[tag] = matched
        print(f"    {len(games):,} returned from API, {len(matched):,} matched our dataset")
    except Exception as e:
        print(f"    ERROR: {e}")
        tag_appids[tag] = set()
    time.sleep(1.1)

# Build output DataFrame: one row per AppID
print("\nBuilding output DataFrame...")
rows = []
for appid in all_appids:
    row = {"AppID": appid}
    for tag in TAGS_TO_FETCH:
        row[f"steamspy_tags_{tag}"] = 1 if appid in tag_appids.get(tag, set()) else 0
    rows.append(row)

df_out = pd.DataFrame(rows)
df_out["AppID"] = df_out["AppID"].astype(int)

# Report coverage
print("\nCoverage (games tagged with each label):")
for tag in TAGS_TO_FETCH:
    col = f"steamspy_tags_{tag}"
    n = int((df_out[col] > 0).sum())
    print(f"  {tag:<20} {n:>6,} games ({n/len(df_out)*100:.1f}%)")

df_out.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}  ({len(df_out):,} rows, {len(TAGS_TO_FETCH)} tags)")
