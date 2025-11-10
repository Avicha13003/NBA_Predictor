# nba_ml_predictor.py
import logging, os, datetime, pytz
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

# Filenames
TODAY_CSV = "nba_today_stats.csv"          # your existing daily combined file
HISTORY_LOG = "player_game_log.csv"        # appended nightly by fetch_boxscores.py
OUT_TOP10_30 = "top10_hit30.csv"
OUT_TOP10_4THREES = "top10_hit4threes.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------- Helper: recent streak features from HISTORY_LOG ----------
def build_recent_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DF with one row per (PLAYER_NAME), including rolling averages:
      - r3_pts, r5_pts, r3_fg3m, r5_fg3m
    Computed from the most recent games in HISTORY_LOG.
    """
    if history.empty:
        return pd.DataFrame(columns=["PLAYER_NAME","r3_pts","r5_pts","r3_fg3m","r5_fg3m"])
    df = history.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.sort_values(["PLAYER_NAME","GAME_DATE"], inplace=True)
    # For rolling we need per-player windows excluding today; but history is always past games.
    r3 = df.groupby("PLAYER_NAME")["PTS"].rolling(3).mean().reset_index(level=0, drop=True)
    r5 = df.groupby("PLAYER_NAME")["PTS"].rolling(5).mean().reset_index(level=0, drop=True)
    t3 = df.groupby("PLAYER_NAME")["FG3M"].rolling(3).mean().reset_index(level=0, drop=True)
    t5 = df.groupby("PLAYER_NAME")["FG3M"].rolling(5).mean().reset_index(level=0, drop=True)
    df["r3_pts"] = r3
    df["r5_pts"] = r5
    df["r3_fg3m"] = t3
    df["r5_fg3m"] = t5
    # Keep latest row per player to summarize “current trend state”
    latest = df.sort_values("GAME_DATE").groupby("PLAYER_NAME").tail(1)
    keep = latest[["PLAYER_NAME","r3_pts","r5_pts","r3_fg3m","r5_fg3m"]].copy()
    # Fill if player has <3 games history
    for c in ["r3_pts","r5_pts","r3_fg3m","r5_fg3m"]:
        keep[c] = keep[c].fillna(0.0)
    return keep

# ---------- Helper: training set from HISTORY_LOG ----------
def prepare_training_sets(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build X/y for both tasks:
      - y30 = didHit30
      - y4 = didHit4Threes
    X uses per-game stats (PTS, FG3M, REB, AST, MIN) from history **last game**,
    plus player rolling trends from the preceding window (we approximate by using the
    rolling means already stored for that row).
    """
    df = history.copy()
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), pd.Series(dtype=int)

    # ensure numeric
    for c in ["PTS","FG3M","REB","AST","MIN"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Build trend columns per row (like in build_recent_features but at row level)
    df = df.sort_values(["PLAYER_NAME","GAME_DATE"]).copy()
    df["r3_pts"] = df.groupby("PLAYER_NAME")["PTS"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    df["r5_pts"] = df.groupby("PLAYER_NAME")["PTS"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    df["r3_fg3m"] = df.groupby("PLAYER_NAME")["FG3M"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    df["r5_fg3m"] = df.groupby("PLAYER_NAME")["FG3M"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)

    feature_cols = ["PTS","FG3M","REB","AST","MIN","r3_pts","r5_pts","r3_fg3m","r5_fg3m"]
    X = df[feature_cols].fillna(0.0)
    y30 = df["didHit30"].astype(int)
    y4 = df["didHit4Threes"].astype(int)
    return X, y30, X.copy(), y4

def make_model():
    num_cols = ["PTS","FG3M","REB","AST","MIN","r3_pts","r5_pts","r3_fg3m","r5_fg3m"]
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=True, with_std=True), num_cols)],
        remainder="drop"
    )
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

# ---------- Main daily predictor ----------
def main():
    # 1) Read today's combined CSV (your file)
    try:
        today_df = pd.read_csv(TODAY_CSV)
    except FileNotFoundError:
        logging.error(f"{TODAY_CSV} not found. Run your daily fetch first.")
        return

    # Normalize column names we’ll use
    # Expect columns: PLAYER, TEAM, PTS, FG3M, REB, AST, MIN, (ARENA, CITY, STATE, GAME_DATE, etc.)
    needed = ["PLAYER","TEAM","PTS","FG3M","REB","AST","MIN"]
    for c in needed:
        if c not in today_df.columns:
            logging.error(f"{TODAY_CSV} missing required column: {c}")
            return

    # 2) Load history (for training + trends)
    try:
        history = pd.read_csv(HISTORY_LOG)
    except FileNotFoundError:
        logging.warning(f"{HISTORY_LOG} not found. First run will use heuristics.")
        history = pd.DataFrame(columns=[
            "GAME_ID","PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION",
            "MIN","PTS","FG3M","REB","AST","GAME_DATE","didHit30","didHit4Threes"
        ])

    # 3) Build per-player recent trend snapshot to merge into today
    recent = build_recent_features(history)  # columns: PLAYER_NAME, r3_pts, r5_pts, r3_fg3m, r5_fg3m
    recent = recent.rename(columns={"PLAYER_NAME":"PLAYER"})
    today = today_df.copy()
    today = today.merge(recent, on="PLAYER", how="left")
    for c in ["r3_pts","r5_pts","r3_fg3m","r5_fg3m"]:
        today[c] = today[c].fillna(0.0)

    # 4) Train/update models from full historical dataset (if enough data)
    X30 = y30 = X4 = y4 = pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), pd.Series(dtype=int)
    have_models = False
    if not history.empty and history["didHit30"].sum() >= 25 and history["didHit4Threes"].sum() >= 25:
        X30, y30, X4, y4 = prepare_training_sets(history)
        if not X30.empty and y30.nunique() == 2 and not X4.empty and y4.nunique() == 2:
            model30 = make_model()
            model4 = make_model()
            model30.fit(X30, y30)
            model4.fit(X4, y4)
            have_models = True
            # (Optional) quick diagnostics
            try:
                p30 = model30.predict_proba(X30)[:,1]
                p4 = model4.predict_proba(X4)[:,1]
                logging.info(f"AUC(30): {roc_auc_score(y30, p30):.3f} | AUC(4threes): {roc_auc_score(y4, p4):.3f}")
            except Exception:
                pass
        else:
            logging.warning("Not enough diverse history to train; using heuristics.")
    else:
        logging.warning("History insufficient for training; using heuristics this run.")

    # 5) Predict today
    feat_cols = ["PTS","FG3M","REB","AST","MIN","r3_pts","r5_pts","r3_fg3m","r5_fg3m"]
    X_today = today[feat_cols].copy()
    # heuristic backups
    if have_models:
        today["prob_hit30"] = np.clip(make_model().fit(X30, y30).predict_proba(X_today)[:,1], 0, 1)
        today["prob_hit4threes"] = np.clip(make_model().fit(X4, y4).predict_proba(X_today)[:,1], 0, 1)
    else:
        # Simple heuristic: blend season stats & recent trends
        # Normalize roughly to [0,1] with soft thresholds
        today["prob_hit30"] = np.clip( (today["PTS"]*0.6 + today["r5_pts"]*0.4 - 20) / 20, 0, 1)
        today["prob_hit4threes"] = np.clip( (today["FG3M"]*0.6 + today["r5_fg3m"]*0.4 - 2) / 4, 0, 1)

        # 6) Save full prediction set
    out_all = today.copy()
    out_all = out_all.sort_values("prob_hit30", ascending=False)
    out_all.to_csv("nba_predictions_today.csv", index=False)
    logging.info(f"✅ Saved full daily predictions to nba_predictions_today.csv ({len(out_all)} players)")

    # 7) Save Top 10 lists (include useful context)
    keep_cols = ["PLAYER","TEAM","TEAM_FULL","TEAM_SIDE","PTS","FG3M","REB","AST","MIN",
                 "r3_pts","r5_pts","r3_fg3m","r5_fg3m","ARENA","CITY","STATE","GAME_DATE",
                 "prob_hit30","prob_hit4threes"]

    # --- Top 10 for 30+ points ---
    top30 = out_all.sort_values("prob_hit30", ascending=False).head(10)
    top30 = top30[[c for c in keep_cols if c in top30.columns]].copy()
    top30.to_csv(OUT_TOP10_30, index=False)

    # --- Top 10 for 4+ threes ---
    top43 = out_all.sort_values("prob_hit4threes", ascending=False).head(10)
    top43 = top43[[c for c in keep_cols if c in top43.columns]].copy()
    top43.to_csv(OUT_TOP10_4THREES, index=False)

    logging.info(f"✅ Wrote {OUT_TOP10_30}, {OUT_TOP10_4THREES}, and nba_predictions_today.csv")

if __name__ == "__main__":
    main()