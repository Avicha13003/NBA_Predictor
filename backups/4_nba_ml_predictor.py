# 4_nba_ml_predictor.py — loads trained models + colorful top-5 summary
import logging, os, joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ==== Files ====
TODAY_CSV    = "nba_today_stats.csv"
HISTORY_LOG  = "player_game_log.csv"
ROLLING_ALL  = "rolling_metrics.csv"
TEAM_CONTEXT = "team_context.csv"

MODEL_HIT30       = "best_model_hit30.pkl"
MODEL_HIT4THREES  = "best_model_hit4threes.pkl"
TRAIN_SUMMARY     = "training_summary.csv"

OUT_TOP10_30       = "top10_hit30.csv"
OUT_TOP10_4THREES  = "top10_hit4threes.csv"
OUT_ALL            = "nba_predictions_today.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------- Color helpers ----------
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"

def color_pct(p: float) -> str:
    """Return colored % string based on probability."""
    if p >= 0.7:
        return f"{GREEN}{p*100:.1f}%{RESET}"
    elif p >= 0.45:
        return f"{YELLOW}{p*100:.1f}%{RESET}"
    else:
        return f"{RED}{p*100:.1f}%{RESET}"

# ---------- Utilities ----------
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
    for c in ["PTS","REB","AST","FG3M","MIN"]:
        t[c] = pd.to_numeric(t.get(c,0), errors="coerce").fillna(0.0)
    t["USAGE_LITE"]  = ((t["PTS"]+t["REB"]+t["AST"])/t["MIN"].replace(0,np.nan)).fillna(0)*100
    t["THREES_RATE"] = (t["FG3M"]/t["PTS"].replace(0,np.nan)).fillna(0)
    t["IS_HOME"]     = (t.get("TEAM_SIDE","")=="Home").astype(int)
    return t

def fmt_pct(x: float) -> str:
    if pd.isna(x) or x < 0.0001:
        return "~0%"
    return f"{x*100:.1f}%"

# ---------- Load Trained Models ----------
def load_models():
    model30 = model4 = None
    auc30 = auc4 = None
    if os.path.exists(MODEL_HIT30):
        try:
            model30 = joblib.load(MODEL_HIT30)
        except Exception as e:
            logging.warning(f"Could not load {MODEL_HIT30}: {e}")
    if os.path.exists(MODEL_HIT4THREES):
        try:
            model4 = joblib.load(MODEL_HIT4THREES)
        except Exception as e:
            logging.warning(f"Could not load {MODEL_HIT4THREES}: {e}")
    if os.path.exists(TRAIN_SUMMARY):
        try:
            summary = pd.read_csv(TRAIN_SUMMARY)
            auc30 = float(summary.loc[(summary["target"]=="didHit30") & (summary["best"]==True),"AUC"].max())
            auc4  = float(summary.loc[(summary["target"]=="didHit4Threes") & (summary["best"]==True),"AUC"].max())
        except Exception:
            pass
    return model30, model4, auc30, auc4

# ---------- Main ----------
def main():
    today_df = safe_read_csv(TODAY_CSV)
    if today_df.empty:
        logging.error("No nba_today_stats.csv found or empty.")
        return

    history = safe_read_csv(HISTORY_LOG)
    rolling = safe_read_csv(ROLLING_ALL)
    context = safe_read_csv(TEAM_CONTEXT)

    if "PLAYER_NAME" in rolling.columns and "PLAYER" not in rolling.columns:
        rolling.rename(columns={"PLAYER_NAME":"PLAYER"}, inplace=True)
    rolling_keep = [c for c in ["PLAYER","r3_pts","r5_pts","r3_fg3m","r5_fg3m"] if c in rolling.columns]
    rolling = rolling[rolling_keep]
    today = today_df.merge(rolling, on="PLAYER", how="left")

    if not context.empty:
        ctx = context.copy()
        if "TEAM" in ctx.columns and "TEAM_ABBREVIATION" not in ctx.columns:
            ctx.rename(columns={"TEAM":"TEAM_ABBREVIATION"}, inplace=True)
        ctx_expected = [
            "TEAM_ABBREVIATION","TEAM_SIDE","DEF_RATING_Z","OPP_DEF_RATING_Z",
            "OPP_OPP_PTS_Z","OPP_PTS_Z","DAYS_REST","IS_B2B","TRAVEL_MILES","TRAVEL_FATIGUE"
        ]
        ctx = ensure_columns(ctx, ctx_expected)
        ctx["IS_B2B_TEXT"] = ctx["IS_B2B"].astype(str)
        ctx["IS_B2B_NUM"]  = ctx["IS_B2B_TEXT"].str.lower().map({"yes":1,"no":0}).fillna(0).astype(int)
        today = today.merge(
            ctx[["TEAM_ABBREVIATION","TEAM_SIDE","DEF_RATING_Z","OPP_DEF_RATING_Z",
                 "OPP_OPP_PTS_Z","OPP_PTS_Z","DAYS_REST","IS_B2B_NUM","TRAVEL_MILES","TRAVEL_FATIGUE"]],
            left_on=["TEAM","TEAM_SIDE"], right_on=["TEAM_ABBREVIATION","TEAM_SIDE"], how="left"
        )
        today.drop(columns=["TEAM_ABBREVIATION"], inplace=True, errors="ignore")

    today = add_basic_today_features(today)

    base_feats = [
        "PTS","FG3M","REB","AST","MIN","r3_pts","r5_pts","r3_fg3m","r5_fg3m",
        "USAGE_LITE","THREES_RATE","IS_HOME"
    ]
    context_feats = [
        "DEF_RATING_Z","OPP_DEF_RATING_Z","OPP_OPP_PTS_Z","OPP_PTS_Z",
        "DAYS_REST","IS_B2B_NUM","TRAVEL_MILES","TRAVEL_FATIGUE"
    ]
    feature_cols = base_feats + context_feats
    today = ensure_columns(today, feature_cols)

    # Load trained models
    model30, model4, auc30, auc4 = load_models()
    have_models = model30 is not None and model4 is not None

    if have_models:
        logging.info(f"🧠 Loaded trained models successfully!")
        logging.info(f"📊 Model Summary → 30pt AUC={auc30:.3f if auc30 else 0.0} | 4Threes AUC={auc4:.3f if auc4 else 0.0}")
    else:
        logging.warning("⚠️ Could not load trained models — using heuristic predictions.")

    # Predict
    X_today = today[feature_cols].fillna(0.0)
    if have_models:
        today["prob_hit30"] = np.clip(model30.predict_proba(X_today)[:,1],0,1)
        today["prob_hit4threes"] = np.clip(model4.predict_proba(X_today)[:,1],0,1)
    else:
        today["prob_hit30"] = np.clip(
            (0.55*today["PTS"] + 0.30*today["r5_pts"] + 0.15*today["r3_pts"]
             + 0.20*today["DEF_RATING_Z"] - 0.25*today["IS_B2B_NUM"]
             - 0.03*today["TRAVEL_MILES"]/500.0 - 0.10*today["TRAVEL_FATIGUE"] - 22) / 20, 0, 1)
        today["prob_hit4threes"] = np.clip(
            (0.55*today["FG3M"] + 0.25*today["r5_fg3m"] + 0.20*today["r3_fg3m"]
             - 0.20*today["OPP_DEF_RATING_Z"] - 0.20*today["IS_B2B_NUM"]
             - 0.03*today["TRAVEL_MILES"]/500.0 - 0.08*today["TRAVEL_FATIGUE"] - 2.5) / 3.5, 0, 1)

    today.to_csv(OUT_ALL, index=False)
    logging.info(f"✅ Saved all predictions → {OUT_ALL} ({len(today)} players)")

    top30 = today.sort_values("prob_hit30", ascending=False).head(10)
    top43 = today.sort_values("prob_hit4threes", ascending=False).head(10)
    top30.to_csv(OUT_TOP10_30, index=False)
    top43.to_csv(OUT_TOP10_4THREES, index=False)
    logging.info(f"🏀 Top 10 exported → {OUT_TOP10_30}, {OUT_TOP10_4THREES}")

    # --- Colorful summary
    print(f"\n{BOLD}{CYAN}=== 🔥 TOP 5 PREDICTED 30+ POINT GAMES ==={RESET}")
    for _, r in top30.head(5).iterrows():
        print(f"{BOLD}{r['PLAYER']:<25}{RESET}{r['TEAM']:<5}{color_pct(r['prob_hit30'])}")

    print(f"\n{BOLD}{CYAN}=== 🎯 TOP 5 PREDICTED 4+ THREE-POINTER GAMES ==={RESET}")
    for _, r in top43.head(5).iterrows():
        print(f"{BOLD}{r['PLAYER']:<25}{RESET}{r['TEAM']:<5}{color_pct(r['prob_hit4threes'])}")

if __name__ == "__main__":
    main()