# 5_props_predictor.py — Prop-line over probabilities with context & injuries (safe + Top-3 summary)
import os, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ==== Inputs ====
TODAY_CSV          = "nba_today_stats.csv"
TEAM_CONTEXT_CSV   = "team_context.csv"
INJURIES_CSV       = "rotowire_injuries.csv"
PROPS_CSV          = "props_today.csv"
GAME_LOG_CSV       = "player_game_log.csv"

# ==== Outputs ====
OUT_ALL            = "nba_prop_predictions_today.csv"
OUT_TOP_PTS        = "top10_over_pts.csv"
OUT_TOP_3PM        = "top10_over_3pm.csv"
OUT_TOP_REB        = "top10_over_reb.csv"
OUT_TOP_AST        = "top10_over_ast.csv"
OUT_TOP_STL        = "top10_over_stl.csv"

# ==== Config ====
RECENT_N           = 8
MIN_RECENT_GMS     = 3
MAX_Z_CLAMP        = 2.5

SUPPORTED_MARKETS = {
    "PTS":  ("PTS",  "Points"),
    "3PM":  ("FG3M", "3-Pointers Made"),
    "REB":  ("REB",  "Rebounds"),
    "AST":  ("AST",  "Assists"),
    "STL":  ("STL",  "Steals"),
}

# ---------- Utility ----------
def safe_read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            logging.info(f"✅ Loaded {path} ({len(df)} rows, {len(df.columns)} cols)")
            return df
        except Exception as e:
            logging.warning(f"Could not read {path}: {e}")
    else:
        logging.warning(f"Missing file: {path}")
    return pd.DataFrame()

def ensure_cols(df: pd.DataFrame, numeric_cols=None, fill=0.0):
    numeric_cols = numeric_cols or []
    out = df.copy()
    for c in numeric_cols:
        if c not in out.columns:
            out[c] = fill
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(fill)
    return out

def normalize_names(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def normalize_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def fmt_pct(x: float) -> str:
    if pd.isna(x) or x < 0.0001:
        return "~0%"
    return f"{float(x)*100:.1f}%"

def safe_int(val, default=0):
    """Safely convert a value to int, treating NaN or None as default."""
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default

# ---------- Injury Processing ----------
def injury_postprocess(df: pd.DataFrame) -> pd.DataFrame:
    inj = df.copy()
    inj.columns = inj.columns.str.strip()
    for c in ["Player", "Team", "Status"]:
        if c not in inj.columns:
            inj[c] = ""

    inj["Player"] = normalize_names(inj["Player"])
    inj["Team"]   = normalize_team(inj["Team"])
    inj["Status"] = inj["Status"].astype(str).str.strip().str.title()

    if "INJURY_CATEGORY" not in inj.columns:
        inj["INJURY_CATEGORY"] = np.where(
            inj["Status"].str.contains("Out|Doubtful", case=False, regex=True), "Out",
            np.where(inj["Status"].str.contains("Probable|Questionable|GTD", case=False, regex=True),
                     "Limited", "Active")
        )
    if "INJURY_FLAG" not in inj.columns:
        inj["INJURY_FLAG"] = inj["INJURY_CATEGORY"].map({"Out":1.0,"Limited":0.5,"Active":0.0}).fillna(0.0)
    if "IS_OUT" not in inj.columns:
        inj["IS_OUT"] = (inj["INJURY_CATEGORY"] == "Out").astype(int)
    if "IS_LIMITED" not in inj.columns:
        inj["IS_LIMITED"] = (inj["INJURY_CATEGORY"] == "Limited").astype(int)

    keep = ["Player","Team","Status","INJURY_CATEGORY","INJURY_FLAG","IS_OUT","IS_LIMITED"]
    extra = [c for c in inj.columns if c not in keep]
    inj = inj[keep + extra]
    return inj

def attach_injuries(df_today: pd.DataFrame, inj_csv: str) -> pd.DataFrame:
    inj = safe_read_csv(inj_csv)
    if inj.empty:
        df = df_today.copy()
        df["INJURY_CATEGORY"] = "Active"
        df["INJURY_FLAG"] = 0.0
        df["IS_OUT"] = 0
        df["IS_LIMITED"] = 0
        return df

    inj = injury_postprocess(inj)
    inj = inj.rename(columns={"Player":"PLAYER","Team":"TEAM","Status":"INJ_Status"})
    inj["PLAYER"] = normalize_names(inj["PLAYER"])
    inj["TEAM"]   = normalize_team(inj["TEAM"])

    merged = df_today.merge(
        inj[["PLAYER","TEAM","INJ_Status","INJURY_CATEGORY","INJURY_FLAG","IS_OUT","IS_LIMITED"]],
        on=["PLAYER","TEAM"],
        how="left"
    )
    merged["INJURY_CATEGORY"] = merged["INJURY_CATEGORY"].fillna("Active")
    merged["INJURY_FLAG"]     = pd.to_numeric(merged["INJURY_FLAG"], errors="coerce").fillna(0.0)
    merged["IS_OUT"]          = pd.to_numeric(merged["IS_OUT"], errors="coerce").fillna(0).astype(int)
    merged["IS_LIMITED"]      = pd.to_numeric(merged["IS_LIMITED"], errors="coerce").fillna(0).astype(int)
    merged["INJ_Status"]      = merged["INJ_Status"].fillna("Active")
    return merged

def injury_adjustment(row):
    try:
        is_out = safe_int(row.get("IS_OUT", 0))
        is_limited = safe_int(row.get("IS_LIMITED", 0))
    except Exception:
        is_out = 0
        is_limited = 0
    if is_out == 1:
        return "out"
    if is_limited == 1:
        return -0.15
    return 0.0

# ---------- Model Helpers ----------
def recent_over_prob(game_log, player, stat_col, line_val):
    if game_log.empty or stat_col not in game_log.columns:
        return (np.nan, 0, np.nan)
    df = game_log[game_log["PLAYER"].astype(str).str.strip() == player].copy()
    if df.empty:
        return (np.nan, 0, np.nan)
    df[stat_col] = pd.to_numeric(df[stat_col], errors="coerce")
    df["GAME_DATE"] = pd.to_datetime(df.get("GAME_DATE"), errors="coerce")
    df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").tail(RECENT_N)
    n = len(df)
    if n == 0:
        return (np.nan, 0, np.nan)
    hits = (df[stat_col] >= float(line_val)).sum()
    mean_minus_line = df[stat_col].mean() - float(line_val)
    return (hits/n, n, mean_minus_line)

def base_over_model(row, market_key, recent_prob, n_recent, mean_minus_line):
    stat_col, _ = SUPPORTED_MARKETS[market_key]
    season_val = float(row.get(stat_col, 0.0))
    line_val   = float(row["LINE"])
    scale = {"PTS":10,"3PM":2,"REB":4,"AST":4,"STL":2}[market_key]
    s1 = 0.5 + np.clip((season_val - line_val)/scale, -0.5, 0.5)
    if np.isnan(recent_prob) or n_recent < MIN_RECENT_GMS:
        s2, w_recent = 0.5, 0.35
    else:
        s2, w_recent = float(recent_prob), 0.55
    s3 = 0.5 + np.clip(mean_minus_line/scale, -0.3, 0.3)
    return float(np.clip((0.45*s1 + w_recent*s2 + 0.10*s3), 0.01, 0.99))

def context_adjustment(row, market_key):
    def c(x):
        try:
            return float(np.clip(x, -MAX_Z_CLAMP, MAX_Z_CLAMP))
        except:
            return 0.0
    defz = c(row.get("DEF_RATING_Z",0))
    opppts = c(row.get("OPP_PTS_Z",0))
    b2b = float(row.get("IS_B2B_NUM",0))
    tf = float(row.get("TRAVEL_FATIGUE",0))
    if market_key in ["PTS","3PM"]:
        boost = 0.04*defz + 0.02*opppts
    elif market_key == "REB":
        boost = 0.015*defz
    elif market_key == "AST":
        boost = 0.02*defz
    else:
        boost = 0.0
    penalty = 0.06*b2b + 0.04*tf
    return boost - penalty

# ---------- Main ----------
def main():
    today     = safe_read_csv(TODAY_CSV)
    context   = safe_read_csv(TEAM_CONTEXT_CSV)
    injuries  = safe_read_csv(INJURIES_CSV)
    props     = safe_read_csv(PROPS_CSV)
    game_log  = safe_read_csv(GAME_LOG_CSV)
    if today.empty or context.empty or props.empty:
        logging.error("Required inputs missing (today/context/props). Aborting.")
        return

    today = attach_injuries(today, INJURIES_CSV)

    # ---- Safe normalization for PLAYER and TEAM ----
    props["PLAYER"] = normalize_names(props.get("PLAYER", pd.Series("", index=props.index)))
    if "TEAM" in props.columns:
        props["TEAM"] = normalize_team(props["TEAM"])
    else:
        props["TEAM"] = ""

    merged = props.merge(today, on=["PLAYER","TEAM"], how="left")
    merged = ensure_cols(merged, ["LINE","PTS","FG3M","REB","AST","STL"], 0.0)

    if not game_log.empty:
        gl = game_log.copy()
        gl["PLAYER"] = normalize_names(gl.get("PLAYER", gl.get("PLAYER_NAME","")))
        for stat in ["PTS","FG3M","REB","AST","STL"]:
            if stat in gl.columns:
                gl[stat] = pd.to_numeric(gl[stat], errors="coerce")
    else:
        gl = pd.DataFrame()

    rows = []
    for _, r in merged.iterrows():
        market = r["MARKET"]
        if market not in SUPPORTED_MARKETS:
            continue
        stat_col, pretty = SUPPORTED_MARKETS[market]
        rp, n_recent, mean_minus_line = recent_over_prob(gl, r["PLAYER"], stat_col, r["LINE"])
        base = base_over_model(r, market, rp, n_recent, mean_minus_line)
        adj = base + context_adjustment(r, market)
        adj = float(np.clip(adj, 0.01, 0.99))
        inj_adj = injury_adjustment(r)
        final = 0.0 if inj_adj == "out" else float(np.clip(adj * (1.0 + inj_adj), 0.0, 1.0))

        rows.append({
            "PLAYER": r["PLAYER"], "TEAM": r.get("TEAM",""),
            "MARKET": market, "PROP_NAME": pretty, "LINE": float(r["LINE"]),
            "SEASON_VAL": float(r.get(stat_col, 0.0)),
            "RECENT_OVER_PROB": rp, "RECENT_N": n_recent,
            "BASE_PROB": base, "ADJ_PROB_BEFORE_INJ": adj,
            "FINAL_OVER_PROB": final,
            "INJ_Status": r.get("INJ_Status","Active"),
            "IS_OUT": safe_int(r.get("IS_OUT",0)),
            "IS_LIMITED": safe_int(r.get("IS_LIMITED",0)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        logging.error("No prop predictions produced.")
        return

    out["FINAL_OVER_PROB_PCT"] = out["FINAL_OVER_PROB"].apply(fmt_pct)
    out = out.sort_values(["MARKET","FINAL_OVER_PROB"], ascending=[True,False])
    out.to_csv(OUT_ALL, index=False)
    logging.info(f"✅ Saved all prop predictions → {OUT_ALL} ({len(out)} rows)")

    # Save top 10 CSVs
    def top_for(m):
        return out[out["MARKET"] == m].sort_values("FINAL_OVER_PROB", ascending=False).head(10)
    top_for("PTS").to_csv(OUT_TOP_PTS, index=False)
    top_for("3PM").to_csv(OUT_TOP_3PM, index=False)
    top_for("REB").to_csv(OUT_TOP_REB, index=False)
    top_for("AST").to_csv(OUT_TOP_AST, index=False)
    top_for("STL").to_csv(OUT_TOP_STL, index=False)
    logging.info("🏀 Exported Top 10s → top10_over_pts.csv, top10_over_3pm.csv, top10_over_reb.csv, top10_over_ast.csv, top10_over_stl.csv")

    # ---- Terminal summary: Top 3 per prop ----
    print("\n================ TOP 3 OVERS BY PROP ================")
    for m, (_, name) in SUPPORTED_MARKETS.items():
        subset = out[out["MARKET"] == m].sort_values("FINAL_OVER_PROB", ascending=False).head(3)
        if subset.empty:
            continue
        print(f"\n{name}:")
        for _, row in subset.iterrows():
            print(f"  {row['PLAYER']:<25}  o{row['LINE']:<4}  → {row['FINAL_OVER_PROB_PCT']}")
    print("=====================================================\n")

if __name__ == "__main__":
    main()