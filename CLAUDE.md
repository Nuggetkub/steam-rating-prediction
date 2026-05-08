# Steam Rating Prediction — Project Guide

## NotebookLM Integration via CLI

**IMPORTANT:** NotebookLM is accessible through the `notebooklm` CLI tool that is already installed on this machine (`notebooklm-py` v0.3.4, installed via pip). You do NOT need a browser, and you do NOT need to log in — authentication is already saved at `C:\Users\hp\.notebooklm\storage_state.json`.

To interact with NotebookLM, run `notebooklm` commands directly in the terminal using the Bash or PowerShell tool. It works like any other CLI (e.g., git, npm).

**Always activate the project notebook first:**
```
notebooklm use ef9430cc-9733-41fb-8a47-8eec2e7dc6d6
```

**Notebook:** Steam Rating Prediction  
**Notebook ID:** `ef9430cc-9733-41fb-8a47-8eec2e7dc6d6`

### Common commands (run these in PowerShell/Bash)

| Task | Command |
|------|---------|
| Check status | `notebooklm status` |
| Ask a research question | `notebooklm ask "your question"` |
| Create a note | `notebooklm note create "title" "content"` |
| Add a source URL | `notebooklm source add https://...` |
| Add a local file | `notebooklm source add ./file.md` |
| List sources | `$env:PYTHONIOENCODING="utf-8"; notebooklm source list --json` |
| Generate a report | `notebooklm generate report` |
| Download latest report | `notebooklm download report` |
| Generate audio podcast | `notebooklm generate audio` |

> Always prefix with `$env:PYTHONIOENCODING="utf-8";` on Windows when output may contain emoji/unicode to avoid encoding errors.

### Full example workflow
```powershell
# 1. Activate notebook
notebooklm use ef9430cc-9733-41fb-8a47-8eec2e7dc6d6

# 2. Ask a question
notebooklm ask "What are the most important features in the model?"

# 3. Save findings as a note
notebooklm note create "Feature Importance" "developer_rating_mean is the top feature at 36.4%"
```

## Project Overview

Steam game rating predictor using XGBoost. Predicts a game's rating (0–10) from metadata (price, genres, tags, developer/publisher history). Trained on 22,097 games. R²=0.81, RMSE=0.6372.

Key files:
- `Rating_Over_Value.ipynb` — main EDA and training notebook (runs on Google Colab)
- `train_xgboost.py` — standalone local training script
- `add_encodings.py` — feature engineering helpers
- `export_results.py` — result export utilities
- `manual_test_games.csv` — hold-out test set (16 games)
