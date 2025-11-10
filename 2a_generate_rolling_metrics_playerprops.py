# generate_rolling_metrics.py — keeps existing rolling outputs + logs prop outcomes for training
import os, logging, datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

LOG_PATH = "player_game_log.csv"
ROLLING_ALL = "rolling_metrics.csv"
ROLLING_DAILY = "rolling_metrics_daily.csv"

# --- New props files ---
PROPS_TODAY      = "props_today.csv"        # produced by your props fetcher (Odds API script)
PROPS_HISTORY    = "props_history.csv"      # appended by this script
PROPS_TRAIN_LOG  = "props_training_log.csv" # one row per player/market/game with DidHitOver

# Market mapping → which game-log column to compare vs LINE
MARKET_TO_STAT = {
    "PTS": "PTS",
    "3PM": "FG3M",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
}

def safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def write_csv(df, path):
    df.to_csv(path, index=False)
    logging.info(f"✅ Wrote {path} ({len(df)} rows)")

def compute_pct(made, att):
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = (made / att) * 100
    return np.where(np.isfinite(pct), pct, 0)

def add_efficiency(df):
    df = df.copy()
    for col in ["FG3A", "FTM", "FTA", "FGA", "FGM", "FG3M"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Effective FG%
    df["EFG_PCT"] = np.where(df["FGA"] > 0,
                             ((df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]) * 100,
                             0)

    # True Shooting %
    denom = (df["FGA"] + 0.44 * df["FTA"]) * 2
    df["TS_PCT"] = np.where(denom > 0, (df["PTS"] / denom) * 100, 0)
    return df

def compute_rolling(df):
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df.dropna(subset=["GAME_DATE"], inplace=True)
    df.sort_values(["PLAYER", "GAME_DATE"], inplace=True)
    grouped = df.groupby("PLAYER")

    # These names mirror what you already used downstream
    df["r3_PTS"] = grouped["PTS"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df["r5_PTS"] = grouped["PTS"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df["r3_FG3_PCT"] = grouped["FG3_PCT"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df["r5_FG3_PCT"] = grouped["FG3_PCT"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df["r3_EFG"] = grouped["EFG_PCT"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df["r5_EFG"] = grouped["EFG_PCT"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df["r3_TS"]  = grouped["TS_PCT"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df["r5_TS"]  = grouped["TS_PCT"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)

    latest = df.sort_values("GAME_DATE").groupby("PLAYER").tail(1)
    keep_cols = ["PLAYER", "TEAM", "TEAM_FULL", "GAME_DATE",
                 "r3_PTS","r5_PTS","r3_FG3_PCT","r5_FG3_PCT","r3_EFG","r5_EFG","r3_TS","r5_TS"]
    return latest[keep_cols]

def append_props_history():
    """Append today's props into props_history.csv (de-duplicated)."""
    today_props = safe_read_csv(PROPS_TODAY)
    if today_props.empty:
        logging.info("No props_today.csv found — skipping props history append.")
        return

    # Normalize columns: PLAYER, TEAM(optional), MARKET, LINE, GAME_DATE
    p = today_props.copy()
    p.columns = p.columns.str.strip()
    # Coerce common names
    rename = {}
    for c in p.columns:
        lc = c.lower()
        if lc in ["player_name","name","player"]:
            rename[c] = "PLAYER"
        elif lc in ["team_abbr","team","team_code"]:
            rename[c] = "TEAM"
        elif lc in ["market","prop","stat","bettype"]:
            rename[c] = "MARKET"
        elif lc in ["line","threshold","value","odds_line"]:
            rename[c] = "LINE"
        elif lc in ["event_date","date","game_date","commence_date","commence_time","ct_date"]:
            rename[c] = "GAME_DATE"
    if rename:
        p = p.rename(columns=rename)

    if "PLAYER" not in p.columns or "MARKET" not in p.columns or "LINE" not in p.columns:
        logging.warning("props_today.csv missing required columns (PLAYER, MARKET, LINE) — skip append.")
        return

    p["PLAYER"] = p["PLAYER"].astype(str).str.strip()
    if "TEAM" in p.columns:
        p["TEAM"] = p["TEAM"].astype(str).str.strip().str.upper()
    p["MARKET"] = p["MARKET"].astype(str).str.strip().str.upper()
    p["LINE"]   = pd.to_numeric(p["LINE"], errors="coerce")

    # Map common aliases
    alias = {"POINTS":"PTS","PTS":"PTS","THREES":"3PM","3PM":"3PM","3PT MADE":"3PM","REBOUNDS":"REB","REB":"REB",
             "ASSISTS":"AST","AST":"AST","STEALS":"STL","STL":"STL"}
    p["MARKET"] = p["MARKET"].replace(alias)

    # Filter to supported markets
    p = p[p["MARKET"].isin(MARKET_TO_STAT.keys())]
    p = p.dropna(subset=["LINE"])

    # Ensure GAME_DATE (YYYY-MM-DD). If missing, assume today (local).
    if "GAME_DATE" not in p.columns or p["GAME_DATE"].isna().all():
        today_str = datetime.date.today().isoformat()
        p["GAME_DATE"] = today_str
    else:
        # parse to date
        p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"], errors="coerce").dt.date.astype(str)

    hist = safe_read_csv(PROPS_HISTORY)
    combined = pd.concat([hist, p], ignore_index=True)
    # De-dup by player/team/market/date; keep last line seen
    subset = ["PLAYER","MARKET","GAME_DATE"] + (["TEAM"] if "TEAM" in combined.columns else [])
    combined = combined.dropna(subset=["PLAYER","MARKET","GAME_DATE"])
    combined = combined.drop_duplicates(subset=subset, keep="last")
    write_csv(combined, PROPS_HISTORY)

def build_props_outcomes_yesterday(game_log: pd.DataFrame):
    """
    From yesterday's games in player_game_log.csv and props_history.csv,
    create per-player per-market rows with DidHitOver.
    Append to props_training_log.csv (de-dup).
    """
    if game_log.empty:
        return

    props_hist = safe_read_csv(PROPS_HISTORY)
    if props_hist.empty:
        logging.info("No props_history.csv yet — skip outcomes.")
        return

    # Normalize
    g = game_log.copy()
    g["PLAYER"] = g.get("PLAYER", g.get("PLAYER_NAME", "")).astype(str).str.strip()
    g["TEAM"]   = g.get("TEAM", "").astype(str).str.strip().str.upper()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"], errors="coerce").dt.date.astype(str)

    p = props_hist.copy()
    p["PLAYER"] = p["PLAYER"].astype(str).str.strip()
    if "TEAM" in p.columns:
        p["TEAM"] = p["TEAM"].astype(str).str.strip().str.upper()
    p["MARKET"] = p["MARKET"].astype(str).str.strip().str.upper()
    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"], errors="coerce").dt.date.astype(str)
    p["LINE"] = pd.to_numeric(p["LINE"], errors="coerce")

    # Yesterday only
    yday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    g_y = g[g["GAME_DATE"] == yday].copy()
    p_y = p[p["GAME_DATE"] == yday].copy()
    if g_y.empty or p_y.empty:
        logging.info(f"No yesterday outcomes to compute (yday={yday}).")
        return

    # Merge props with game stats
    keys = ["PLAYER","GAME_DATE"]
    if "TEAM" in p_y.columns and "TEAM" in g_y.columns:
        keys = ["PLAYER","TEAM","GAME_DATE"]
    m = p_y.merge(g_y, on=keys, how="left", suffixes=("_PROP",""))

    # Determine actual stat and DidHitOver
    rows = []
    for _, r in m.iterrows():
        market = r["MARKET"]
        if market not in MARKET_TO_STAT:
            continue
        stat_col = MARKET_TO_STAT[market]
        actual = pd.to_numeric(r.get(stat_col, np.nan), errors="coerce")
        line   = pd.to_numeric(r.get("LINE", np.nan), errors="coerce")
        if pd.isna(actual) or pd.isna(line):
            continue
        did_over = int(actual >= line)
        rows.append({
            "PLAYER": r["PLAYER"],
            "TEAM": r.get("TEAM", ""),
            "GAME_DATE": r["GAME_DATE"],
            "MARKET": market,
            "LINE": float(line),
            "ACTUAL": float(actual),
            "DidHitOver": did_over,
            # a few useful features for later (non-leak features will be built in trainer)
            "PTS": r.get("PTS", np.nan),
            "FG3M": r.get("FG3M", np.nan),
            "REB": r.get("REB", np.nan),
            "AST": r.get("AST", np.nan),
            "STL": r.get("STL", np.nan),
            "MIN": r.get("MIN", np.nan),
            "TEAM_SIDE": r.get("TEAM_SIDE", ""),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        logging.info("No prop outcomes produced for yesterday.")
        return

    # Append de-duplicated to props_training_log.csv
    existing = safe_read_csv(PROPS_TRAIN_LOG)
    combined = pd.concat([existing, out], ignore_index=True)
    combined = combined.drop_duplicates(subset=["PLAYER","TEAM","GAME_DATE","MARKET"], keep="last")
    write_csv(combined, PROPS_TRAIN_LOG)

def main():
    if not os.path.exists(LOG_PATH):
        logging.error(f"{LOG_PATH} not found.")
        return

    df = safe_read_csv(LOG_PATH)
    for col in ["PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "REB", "AST", "STL", "MIN"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Compute base FG3%
    df["FG3_PCT"] = compute_pct(df["FG3M"], df["FG3A"])
    df = add_efficiency(df)

    # Rolling outputs (unchanged)
    rolled = compute_rolling(df)
    write_csv(rolled, ROLLING_ALL)

    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    rolled_daily = rolled[rolled["GAME_DATE"] == yesterday]
    write_csv(rolled_daily, ROLLING_DAILY)

    # New: props history append + yesterday outcomes
    append_props_history()
    build_props_outcomes_yesterday(df)

if __name__ == "__main__":
    main()