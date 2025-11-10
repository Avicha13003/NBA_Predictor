# 4_nba_ml_predictor.py — final version (balanced positions + sanity check)
import logging, os, requests, time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from nba_api.stats.endpoints import CommonAllPlayers

# ==== Files ====
TODAY_CSV    = "nba_today_stats.csv"
HISTORY_LOG  = "player_game_log.csv"
ROLLING_ALL  = "rolling_metrics.csv"
TEAM_CONTEXT = "team_context.csv"
POS_CACHE    = "player_positions_cache.csv"

OUT_TOP10_30       = "top10_hit30.csv"
OUT_TOP10_4THREES  = "top10_hit4threes.csv"
OUT_ALL            = "nba_predictions_today.csv"

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_TEAMS_URL    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------- Utility ----------
def safe_read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            logging.warning(f"Could not read {path}: {e}")
    return pd.DataFrame()

def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
        else:
            if out[c].dtype == object or str(out[c].dtype).startswith("string"):
                out[c] = out[c].fillna("")
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def add_basic_today_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    for c in ["PTS", "REB", "AST", "FG3M", "MIN"]:
        val = t[c] if c in t.columns else pd.Series([0]*len(t))
        t[c] = pd.to_numeric(val, errors="coerce").fillna(0.0)
    t["USAGE_LITE"]  = ((t["PTS"] + t["REB"] + t["AST"]) / t["MIN"].replace(0, np.nan)).fillna(0) * 100
    t["THREES_RATE"] = (t["FG3M"] / t["PTS"].replace(0, np.nan)).fillna(0)
    t["IS_HOME"]     = (t.get("TEAM_SIDE", "") == "Home").astype(int)
    return t

def _espn_json(url, retries=3, sleep=0.3):
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(sleep)
    return {}

# ---------- Build positions cache ----------
def build_positions_cache(path: str = POS_CACHE) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            logging.info(f"Loaded positions cache: {path} ({len(df)} rows)")
            return df

    logging.info("Fetching player positions from ESPN team rosters...")
    team_js = _espn_json(ESPN_TEAMS_URL)
    team_items = team_js.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    rows = []

    for team in team_items:
        tinfo = team.get("team", {}) or {}
        tid = tinfo.get("id")
        if not tid:
            continue
        roster_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/roster"
        rjs = _espn_json(roster_url)
        for ath in rjs.get("athletes", []):
            name = ath.get("displayName", "")
            pos = (ath.get("position", {}) or {}).get("displayName", "")
            if not name:
                continue
            rows.append({"PLAYER": name.strip(), "POSITION": pos.strip().upper()})
        time.sleep(0.2)

    df = pd.DataFrame(rows).drop_duplicates(subset=["PLAYER"])

    # --- Normalize ESPN labels ---
    pos_map = {
        "POINT GUARD": "PG",
        "SHOOTING GUARD": "SG",
        "GUARD": "SG",        # generic guard = SG
        "G": "SG",
        "SMALL FORWARD": "SF",
        "POWER FORWARD": "PF",
        "FORWARD": "SF",      # generic forward = SF
        "F": "SF",
        "CENTER": "C"
    }
    df["POSITION"] = df["POSITION"].str.upper().map(pos_map).fillna(df["POSITION"])

    for pos in ["PG","SG","SF","PF","C"]:
        df[f"POS_{pos}"] = (df["POSITION"] == pos).astype(int)
    df["POS_WING"] = df["POSITION"].isin(["SG","SF"]).astype(int)
    df["POS_BIG"]  = df["POSITION"].isin(["PF","C"]).astype(int)

    # --- Sanity check ---
    counts = df["POSITION"].value_counts().to_dict()
    missing = df.loc[~df["POSITION"].isin(["PG","SG","SF","PF","C"]), "PLAYER"].tolist()
    logging.info(f"Position distribution: {counts}")
    if missing:
        logging.warning(f"Unmapped positions for {len(missing)} players (e.g. {missing[:5]})")

    df.to_csv(path, index=False)
    logging.info(f"✅ Saved position cache → {path} ({len(df)} players)")
    return df

# ---------- ESPN Injuries ----------
def load_injury_board() -> pd.DataFrame:
    try:
        r = requests.get(ESPN_INJURIES_URL, timeout=12)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        logging.warning(f"ESPN injuries request failed: {e}")
        return pd.DataFrame(columns=["PLAYER"])
    rows = []
    for team in js.get("injuries", []):
        for item in team.get("players", []):
            ath = item.get("athlete", {}) or {}
            status = (item.get("status", {}) or {}).get("type", "") or item.get("status", {}).get("status", "")
            desc = (item.get("status", {}) or {}).get("detail", "") or item.get("status", {}).get("description", "") or ""
            name = ath.get("displayName", "")
            if not name:
                continue
            s_lower = str(status).lower()
            out = int("out" in s_lower or "suspended" in s_lower)
            doubtful = int("doubt" in s_lower)
            questionable = int("questionable" in s_lower)
            probable = int("probable" in s_lower)
            severity = 1.0 if out else (0.75 if doubtful else (0.5 if questionable else (0.25 if probable else 0.0)))
            rows.append({
                "PLAYER": name.strip(),
                "INJ_STATUS": status or "",
                "INJ_DESC": desc or "",
                "INJ_OUT": out,
                "INJ_DOUBTFUL": doubtful,
                "INJ_QUESTIONABLE": questionable,
                "INJ_PROBABLE": probable,
                "INJ_SEVERITY": severity
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["PLAYER"])
    if "PLAYER" not in df.columns:
        df["PLAYER"] = ""
    logging.info(f"Pulled ESPN injuries for {len(df)} players")
    return df

# ---------- Main ----------
def main():
    today_df = safe_read_csv(TODAY_CSV)
    if today_df.empty:
        logging.error("No nba_today_stats.csv found or empty.")
        return

    history = safe_read_csv(HISTORY_LOG)
    rolling = safe_read_csv(ROLLING_ALL)
    context = safe_read_csv(TEAM_CONTEXT)
    pos_df = build_positions_cache(POS_CACHE)
    inj_df = load_injury_board()

    if "PLAYER_NAME" in rolling.columns and "PLAYER" not in rolling.columns:
        rolling.rename(columns={"PLAYER_NAME": "PLAYER"}, inplace=True)
    rolling_keep = ["PLAYER","r3_pts","r5_pts","r3_fg3m","r5_fg3m"]
    rolling = rolling[[c for c in rolling_keep if c in rolling.columns]].copy()

    today = today_df.copy().merge(rolling, on="PLAYER", how="left")

    if not context.empty:
        ctx = context.copy()
        if "TEAM" in ctx.columns and "TEAM_ABBREVIATION" not in ctx.columns:
            ctx.rename(columns={"TEAM": "TEAM_ABBREVIATION"}, inplace=True)
        ctx["IS_B2B_TEXT"] = ctx["IS_B2B"].astype(str)
        ctx["IS_B2B_NUM"]  = ctx["IS_B2B_TEXT"].str.lower().map({"yes":1,"no":0}).fillna(0).astype(int)
        today = today.merge(
            ctx[["TEAM_ABBREVIATION","TEAM_SIDE","DEF_RATING_Z","OPP_DEF_RATING_Z","OPP_OPP_PTS_Z","OPP_PTS_Z",
                 "DAYS_REST","IS_B2B_TEXT","IS_B2B_NUM","TRAVEL_MILES","TRAVEL_FATIGUE"]],
            left_on=["TEAM","TEAM_SIDE"],
            right_on=["TEAM_ABBREVIATION","TEAM_SIDE"],
            how="left"
        )
        today.drop(columns=["TEAM_ABBREVIATION"], inplace=True, errors="ignore")

    today = today.merge(pos_df, on="PLAYER", how="left")
    today = today.merge(inj_df, on="PLAYER", how="left")

    for c in ["INJ_OUT","INJ_DOUBTFUL","INJ_QUESTIONABLE","INJ_PROBABLE","INJ_SEVERITY"]:
        val = today[c] if c in today.columns else pd.Series([0]*len(today))
        today[c] = pd.to_numeric(val, errors="coerce").fillna(0.0)

    today = add_basic_today_features(today)

    base_feats = ["PTS","FG3M","REB","AST","MIN","r3_pts","r5_pts","USAGE_LITE","THREES_RATE","IS_HOME"]
    context_feats = ["DEF_RATING_Z","OPP_DEF_RATING_Z","OPP_OPP_PTS_Z","OPP_PTS_Z","DAYS_REST","IS_B2B_NUM","TRAVEL_MILES","TRAVEL_FATIGUE"]
    pos_feats = ["POS_PG","POS_SG","POS_SF","POS_PF","POS_C","POS_WING","POS_BIG"]
    inj_feats = ["INJ_OUT","INJ_DOUBTFUL","INJ_QUESTIONABLE","INJ_PROBABLE","INJ_SEVERITY"]

    feature_cols = base_feats + context_feats + pos_feats + inj_feats
    today = ensure_columns(today, feature_cols)

    # --- Safe roll values ---
    r3_fg3m = today["r3_fg3m"] if "r3_fg3m" in today else 0
    r5_fg3m = today["r5_fg3m"] if "r5_fg3m" in today else 0
    r3_pts  = today["r3_pts"]  if "r3_pts"  in today else 0
    r5_pts  = today["r5_pts"]  if "r5_pts"  in today else 0

    # --- Heuristic model ---
    today["prob_hit30"] = np.clip(
        (0.55*today["PTS"] + 0.30*r5_pts + 0.15*r3_pts
         + 0.20*today["DEF_RATING_Z"] - 0.25*today["IS_B2B_NUM"]
         - 0.03*today["TRAVEL_MILES"]/500 - 0.10*today["TRAVEL_FATIGUE"]
         - 0.45*today["INJ_SEVERITY"] - 22)/20, 0, 1)

    today["prob_hit4threes"] = np.clip(
        (0.55*today["FG3M"] + 0.25*r5_fg3m + 0.20*r3_fg3m
         - 0.20*today["OPP_DEF_RATING_Z"] - 0.20*today["IS_B2B_NUM"]
         - 0.03*today["TRAVEL_MILES"]/500 - 0.08*today["TRAVEL_FATIGUE"]
         - 0.40*today["INJ_SEVERITY"] - 2.5)/3.5, 0, 1)

    today.loc[today["INJ_OUT"] >= 1, ["prob_hit30","prob_hit4threes"]] *= 0.02

    today["prob_hit30_pct"] = (today["prob_hit30"]*100).round(1).astype(str) + "%"
    today["prob_hit4threes_pct"] = (today["prob_hit4threes"]*100).round(1).astype(str) + "%"

    today.to_csv(OUT_ALL, index=False)
    logging.info(f"✅ Saved predictions → {OUT_ALL} ({len(today)} players)")
    today.sort_values("prob_hit30", ascending=False).head(10).to_csv(OUT_TOP10_30, index=False)
    today.sort_values("prob_hit4threes", ascending=False).head(10).to_csv(OUT_TOP10_4THREES, index=False)
    logging.info("🏀 Exported Top 10s")

if __name__ == "__main__":
    main()