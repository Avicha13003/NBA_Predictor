# 4a_nba_ml_predictor.py — Heuristic + Rotowire Injury Integration (fixed merge)
import logging, os, unicodedata
import pandas as pd
import numpy as np
from difflib import get_close_matches

# ==== Files ====
TODAY_CSV    = "nba_today_stats.csv"
TEAM_CONTEXT = "team_context.csv"
ROTOWIRE_CSV = "rotowire_injuries.csv"

OUT_TOP10_30       = "top10_hit30.csv"
OUT_TOP10_4THREES  = "top10_hit4threes.csv"
OUT_ALL            = "nba_predictions_today.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------- Helpers ----------
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
        t[c] = pd.to_numeric(t.get(c, 0), errors="coerce").fillna(0.0)
    t["USAGE_LITE"]  = ((t["PTS"] + t["REB"] + t["AST"]) / t["MIN"].replace(0, np.nan)).fillna(0) * 100
    t["THREES_RATE"] = (t["FG3M"] / t["PTS"].replace(0, np.nan)).fillna(0)
    t["IS_HOME"]     = (t.get("TEAM_SIDE", "").str.strip().str.lower() == "home").astype(int)
    return t

def fmt_pct(x: float) -> str:
    if pd.isna(x) or x < 0.0001:
        return "~0%"
    return f"{x*100:.1f}%"

# ---------- Injury Handling ----------
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.replace(".", "").replace(",", "")
    for suf in [" Jr", " Sr", " II", " III"]:
        name = name.replace(suf, "")
    return name.strip().lower()

def load_and_merge_injuries(today: pd.DataFrame) -> pd.DataFrame:
    inj = safe_read_csv(ROTOWIRE_CSV)
    if inj.empty:
        today["INJ_Status"] = "Active"
        today["INJ_Injury"] = ""
        today["INJURY_CATEGORY"] = "Active"
        today["INJURY_FLAG"] = 0.0
        today["IS_OUT"] = 0
        today["IS_LIMITED"] = 0
        logging.warning("Injury CSV missing/empty — treating all players as Active.")
        return today

    # Normalize column names
    inj.columns = inj.columns.str.strip()
    inj["_norm_name"] = inj["Player"].apply(normalize_name)
    today["_norm_name"] = today["PLAYER"].apply(normalize_name)

    # If derived columns are missing, infer them from Status
    if "INJURY_CATEGORY" not in inj.columns:
        inj["Status"] = inj["Status"].astype(str).str.strip().str.title()
        inj["INJURY_CATEGORY"] = inj["Status"].apply(
            lambda x: "Out" if "Out" in x or "Doubtful" in x else
                      "Limited" if "Questionable" in x or "Probable" in x else
                      "Active"
        )
        inj["IS_OUT"] = inj["INJURY_CATEGORY"].eq("Out").astype(int)
        inj["IS_LIMITED"] = inj["INJURY_CATEGORY"].eq("Limited").astype(int)
        inj["INJURY_FLAG"] = inj["INJURY_CATEGORY"].map({"Out": 1.0, "Limited": 0.5, "Active": 0.0})
        logging.info("🧩 Derived injury category columns from Status text")

    # Merge by normalized name
    merged = today.merge(inj, on="_norm_name", how="left", suffixes=("", "_inj"))

    # Fill defaults
    merged["Status"] = merged["Status"].fillna("Active")
    merged["INJURY_CATEGORY"] = merged["INJURY_CATEGORY"].fillna("Active")
    merged["INJURY_FLAG"] = pd.to_numeric(merged["INJURY_FLAG"], errors="coerce").fillna(0.0)
    merged["IS_OUT"] = pd.to_numeric(merged["IS_OUT"], errors="coerce").fillna(0).astype(int)
    merged["IS_LIMITED"] = pd.to_numeric(merged["IS_LIMITED"], errors="coerce").fillna(0).astype(int)
    merged["Injury"] = merged.get("Injury", "").fillna("")

    n_listed = int((merged["Status"] != "Active").sum())
    n_out = int(merged["IS_OUT"].sum())
    n_lim = int(merged["IS_LIMITED"].sum())
    logging.info(f"🩺 Injury merge: {n_listed} listed | OUT={n_out} | LIMITED={n_lim}")

    merged.rename(columns={
        "Status": "INJ_Status",
        "Injury": "INJ_Injury"
    }, inplace=True)
    merged.drop(columns=["_norm_name"], inplace=True, errors="ignore")
    return merged

# ---------- Main ----------
def main():
    today_df = safe_read_csv(TODAY_CSV)
    context = safe_read_csv(TEAM_CONTEXT)
    if today_df.empty or context.empty:
        logging.error("Missing input files — aborting.")
        return

    context.columns = context.columns.str.strip()
    today_df.columns = today_df.columns.str.strip()
    context["TEAM_ABBREVIATION"] = context["TEAM_ABBREVIATION"].astype(str).str.strip().str.upper()
    today_df["TEAM"] = today_df["TEAM"].astype(str).str.strip().str.upper()
    context.rename(columns={c: "TEAM_SIDE" if c.replace(" ", "") == "TEAM_SIDE" else c for c in context.columns}, inplace=True)
    context["TEAM_SIDE"] = context["TEAM_SIDE"].astype(str).str.strip().str.title()
    today_df["TEAM_SIDE"] = today_df["TEAM_SIDE"].astype(str).str.strip().str.title()

    # Convert IS_B2B
    if "IS_B2B" in context.columns and "IS_B2B_NUM" not in context.columns:
        context["IS_B2B_NUM"] = context["IS_B2B"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

    context_feats = [
        "DEF_RATING_Z","OPP_DEF_RATING_Z","OPP_OPP_PTS_Z","OPP_PTS_Z",
        "DAYS_REST","IS_B2B_NUM","TRAVEL_MILES","TRAVEL_FATIGUE"
    ]
    context = ensure_columns(context, ["TEAM_ABBREVIATION","TEAM_SIDE"] + context_feats)

    today = today_df.merge(
        context[["TEAM_ABBREVIATION","TEAM_SIDE"] + context_feats],
        left_on=["TEAM","TEAM_SIDE"],
        right_on=["TEAM_ABBREVIATION","TEAM_SIDE"],
        how="left"
    ).drop(columns=["TEAM_ABBREVIATION"], errors="ignore")

    today = add_basic_today_features(today)
    today = ensure_columns(today, context_feats)

    # Merge injuries
    today = load_and_merge_injuries(today)

    # --- Heuristic mode ---
    logging.warning("⚙️ Using HEURISTIC MODE (balanced scoring + injuries)")

    raw30 = np.clip(
        (0.65*today["PTS"] + 0.25*today["REB"] + 0.10*today["AST"]
         + 0.10*today["DEF_RATING_Z"]
         - 0.05*today["OPP_DEF_RATING_Z"]
         - 0.15*today["IS_B2B_NUM"]
         - 0.02*today["TRAVEL_FATIGUE"]) / 40, 0, 1
    )
    raw43 = np.clip(
        (0.70*today["FG3M"] + 0.30*today["THREES_RATE"]
         - 0.15*today["OPP_DEF_RATING_Z"]
         - 0.10*today["IS_B2B_NUM"]
         - 0.03*today["TRAVEL_FATIGUE"]) / 6, 0, 1
    )

    # Injury multiplier
    inj_mult = (1.0
                - 0.80*today["IS_OUT"]
                - 0.40*today["IS_LIMITED"]
                - 0.25*today["INJURY_FLAG"]).clip(0, 1)

    today["prob_hit30"] = (raw30 * inj_mult).where(today["IS_OUT"] == 0, 0.0)
    today["prob_hit4threes"] = (raw43 * inj_mult).where(today["IS_OUT"] == 0, 0.0)

    today["prob_hit30_pct"] = today["prob_hit30"].apply(fmt_pct)
    today["prob_hit4threes_pct"] = today["prob_hit4threes"].apply(fmt_pct)
    today["PREDICTION_MODE"] = "Heuristic"

    keep_cols = [
        "PLAYER","TEAM","TEAM_FULL","TEAM_SIDE","PTS","FG3M","REB","AST","MIN",
        "USAGE_LITE","THREES_RATE","IS_HOME",
        "DEF_RATING_Z","OPP_DEF_RATING_Z","OPP_OPP_PTS_Z","OPP_PTS_Z","DAYS_REST","IS_B2B_NUM",
        "TRAVEL_MILES","TRAVEL_FATIGUE",
        "INJ_Status","INJ_Injury","INJURY_CATEGORY","INJURY_FLAG","IS_OUT","IS_LIMITED",
        "prob_hit30","prob_hit30_pct","prob_hit4threes","prob_hit4threes_pct","PREDICTION_MODE"
    ]
    today_out = ensure_columns(today, keep_cols)[keep_cols]
    today_out.to_csv(OUT_ALL, index=False)
    logging.info(f"✅ Saved predictions → {OUT_ALL} ({len(today_out)} players)")

    top30 = today_out.sort_values("prob_hit30", ascending=False).head(10)
    top43 = today_out.sort_values("prob_hit4threes", ascending=False).head(10)
    top30.to_csv(OUT_TOP10_30, index=False)
    top43.to_csv(OUT_TOP10_4THREES, index=False)
    logging.info("🏀 Exported Top 10s")

if __name__ == "__main__":
    main()