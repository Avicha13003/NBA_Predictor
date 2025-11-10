# 4a_nba_ml_predictor.py — Adds props “best overs” (PTS, FG3M, REB, AST, STL)
import logging, os
import pandas as pd
import numpy as np

TODAY_CSV      = "nba_today_stats.csv"
TEAM_CONTEXT   = "team_context.csv"
GAME_LOG_CSV   = "player_game_log.csv"   # your historical per-game logs
PROPS_TODAY    = "props_today.csv"       # created by fetch_props_dk.py

OUT_ALL               = "nba_predictions_today.csv"
OUT_TOP_OVER_PTS      = "top10_over_points.csv"
OUT_TOP_OVER_3PM      = "top10_over_threes.csv"
OUT_TOP_OVER_REB      = "top10_over_rebounds.csv"
OUT_TOP_OVER_AST      = "top10_over_assists.csv"
OUT_TOP_OVER_STL      = "top10_over_steals.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def safe_read(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            logging.info(f"✅ Loaded {path} ({len(df)} rows, {len(df.columns)} cols)")
            return df
        except Exception as e:
            logging.warning(f"Could not read {path}: {e}")
    else:
        logging.warning(f"Missing {path}")
    return pd.DataFrame()

def ensure_cols(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            if df[c].dtype == object or str(df[c].dtype).startswith("string"):
                df[c] = df[c].fillna("")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def add_basic_today_features(df):
    t = df.copy()
    for c in ["PTS","REB","AST","FG3M","MIN"]:
        t[c] = pd.to_numeric(t.get(c, 0), errors="coerce").fillna(0.0)
    t["USAGE_LITE"] = ((t["PTS"] + t["REB"] + t["AST"]) / t["MIN"].replace(0, np.nan)).fillna(0)*100
    t["THREES_RATE"] = (t["FG3M"] / t["PTS"].replace(0, np.nan)).fillna(0)
    t["IS_HOME"] = (t.get("TEAM_SIDE","").str.strip().str.lower()=="home").astype(int)
    return t

def empirical_over_prob(series, line, min_games=4, bias=1.0):
    """Probability stat > line from recent samples with tiny Laplace smoothing."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return np.nan
    overs = (x > line).sum()
    # Laplace smoothing (add bias to both classes)
    return (overs + bias) / (n + 2*bias)

def compute_over_probs(today, props_df, logs):
    # Map from our MARKET codes to stat columns in logs
    market_to_col = {"PTS": "PTS", "FG3M": "FG3M", "REB": "REB", "AST": "AST", "STL": "STL"}
    want_markets = list(market_to_col.keys())
    props = props_df[props_df["MARKET"].isin(want_markets)].copy()

    # Prepare logs
    logs = logs.copy()
    if "PLAYER_NAME" in logs.columns and "PLAYER" not in logs.columns:
        logs.rename(columns={"PLAYER_NAME":"PLAYER"}, inplace=True)
    for stat in market_to_col.values():
        if stat not in logs.columns:
            logs[stat] = np.nan
        logs[stat] = pd.to_numeric(logs[stat], errors="coerce")

    logs["GAME_DATE"] = pd.to_datetime(logs.get("GAME_DATE"), errors="coerce")
    logs.sort_values(["PLAYER","GAME_DATE"], inplace=True)

    results = []
    for (player, team, market), grp in props.groupby(["PLAYER","TEAM","MARKET"]):
        stat_col = market_to_col[market]
        line = float(np.median(grp["LINE"]))  # use median if multiple books
        # last 10 games for this player
        plog = logs[logs["PLAYER"]==player].dropna(subset=[stat_col]).tail(10)
        prob = empirical_over_prob(plog[stat_col], line, min_games=4, bias=1.0)
        n_used = len(plog)

        # Fallback: season average proxy if no logs
        if np.isnan(prob):
            # Use today’s season per-game (from today df)
            row = today[(today["PLAYER"]==player) & (today["TEAM"]==team)].head(1)
            season_val = float(row[stat_col].iloc[0]) if not row.empty else np.nan
            if pd.isna(season_val):
                prob = np.nan
            else:
                # simple softness around difference vs line
                prob = float(1.0/(1.0+np.exp(-(season_val - line))))
            n_used = 0

        results.append({
            "PLAYER": player,
            "TEAM": team,
            "MARKET": market,
            "LINE": line,
            "over_prob": round(float(prob), 4) if not np.isnan(prob) else np.nan,
            "samples_used": int(n_used)
        })

    return pd.DataFrame(results)

def main():
    today = safe_read(TODAY_CSV)
    context = safe_read(TEAM_CONTEXT)  # optional here; keep for future
    props  = safe_read(PROPS_TODAY)
    logs   = safe_read(GAME_LOG_CSV)

    if today.empty or props.empty:
        logging.error("Need TODAY stats and PROPS to proceed.")
        return

    # Normalize casing
    today["PLAYER"] = today["PLAYER"].astype(str).str.replace("  "," ").str.strip()
    today["TEAM"]   = today["TEAM"].astype(str).str.upper().str.strip()
    props["PLAYER"] = props["PLAYER"].astype(str).str.replace("  "," ").str.strip()
    props["TEAM"]   = props["TEAM"].astype(str).str.upper().str.strip()

    # Build baseline predictions (same balanced heuristic you liked)
    today = add_basic_today_features(today)
    base_keep = [
        "PLAYER","TEAM","TEAM_FULL","TEAM_SIDE","PTS","FG3M","REB","AST","MIN",
        "USAGE_LITE","THREES_RATE","IS_HOME"
    ]
    today = ensure_cols(today, base_keep)[base_keep]

    # Compute over probabilities vs prop lines
    over_df = compute_over_probs(today, props, logs)

    # Join some display fields
    merged = over_df.merge(
        today[["PLAYER","TEAM","TEAM_FULL","TEAM_SIDE","PTS","FG3M","REB","AST","MIN"]],
        on=["PLAYER","TEAM"], how="left"
    )

    # Save a wide “all” table merged with today
    # Pivot over_df to columns per market
    wide = (over_df.pivot_table(index=["PLAYER","TEAM"], columns="MARKET", values=["LINE","over_prob"], aggfunc="first")
            .sort_index(axis=1, level=0))
    wide.columns = [f"{a}_{b}" for a,b in wide.columns]  # e.g., LINE_AST, over_prob_AST
    all_out = (today.merge(wide, on=["PLAYER","TEAM"], how="left")
                    .sort_values(["TEAM","PLAYER"]))
    all_out.to_csv(OUT_ALL, index=False)
    logging.info(f"✅ Saved {OUT_ALL} ({len(all_out)} players)")

    def dump_top10(market, out_path):
        sub = merged[merged["MARKET"]==market].copy()
        if sub.empty:
            logging.warning(f"No props found for {market}")
            return
        sub["over_pct"] = (sub["over_prob"]*100).round(1).astype(str) + "%"
        top = (sub.sort_values(["over_prob","samples_used","LINE"], ascending=[False, False, True])
                  .head(10)[["PLAYER","TEAM","TEAM_FULL","TEAM_SIDE","MARKET","LINE","over_prob","over_pct","samples_used"]])
        top.to_csv(out_path, index=False)
        logging.info(f"🏀 Top 10 {market} overs → {out_path}")

    dump_top10("PTS",  OUT_TOP_OVER_PTS)
    dump_top10("FG3M", OUT_TOP_OVER_3PM)
    dump_top10("REB",  OUT_TOP_OVER_REB)
    dump_top10("AST",  OUT_TOP_OVER_AST)
    dump_top10("STL",  OUT_TOP_OVER_STL)

if __name__ == "__main__":
    main()