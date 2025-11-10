# nba_train_props_model.py — trains 1 model per market to predict DidHitOver using leak-free rolling features
import os, logging, json, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ===== Inputs =====
PROPS_TRAIN_LOG = "props_training_log.csv"  # built by your pipeline (props + boxscore outcomes)

# ===== Outputs =====
OUT_MODELS = {
    "PTS": "model_over_pts.pkl",
    "3PM": "model_over_3pm.pkl",
    "REB": "model_over_reb.pkl",
    "AST": "model_over_ast.pkl",
    "STL": "model_over_stl.pkl",
}
SUMMARY_OUT = "training_summary_props.csv"
FEATURE_MAP_OUT = "trained_feature_columns.json"  # which features were used per market

# ===== Config =====
STAT_COLS = ["PTS", "FG3M", "REB", "AST", "STL", "MIN"]  # numeric stat columns expected in training log
MIN_CLASS_POSITIVES = 20  # require at least this many positives per market before training
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Optional context columns—used if present in props_training_log.csv (already leak-free per row if logged correctly)
OPTIONAL_CONTEXT_COLS = [
    "DEF_RATING_Z", "OPP_DEF_RATING_Z", "OPP_PTS_Z",
    "DAYS_REST", "IS_B2B_NUM", "TRAVEL_MILES", "TRAVEL_FATIGUE",
    "INJURY_FLAG", "IS_OUT", "IS_LIMITED"
]

def safe_read_csv(p: str) -> pd.DataFrame:
    if not os.path.exists(p):
        logging.error(f"{p} not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        logging.info(f"✅ Loaded {p} ({len(df)} rows, {len(df.columns)} cols)")
        return df
    except Exception as e:
        logging.error(f"Could not read {p}: {e}")
        return pd.DataFrame()

def eval_metrics(y_true, y_prob, y_pred):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    return auc, acc, f1

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Basic text fields
    for c in ["PLAYER", "MARKET", "GAME_DATE", "TEAM"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()
    # Numerics
    for c in STAT_COLS + ["LINE", "ACTUAL", "DidHitOver"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # Normalize MARKET into our five keys
    if "MARKET" in d.columns:
        d["MARKET"] = d["MARKET"].str.upper().replace({
            "POINTS": "PTS", "PTS": "PTS",
            "THREES": "3PM", "3PT MADE": "3PM", "3PTM": "3PM", "3-PT MADE": "3PM",
            "REBOUNDS": "REB", "REB": "REB",
            "ASSISTS": "AST", "AST": "AST",
            "STEALS": "STL", "STL": "STL",
        })
    return d

def build_leakfree_features(dm: pd.DataFrame, stat_col: str):
    """
    dm: rows for a single MARKET with columns:
      PLAYER, GAME_DATE, LINE, ACTUAL, DidHitOver, STAT_COLS..., [OPTIONAL_CONTEXT_COLS...]
    We compute per-player rolling means up to the previous game (shifted) to avoid leakage.
    Returns: (features_df, feature_cols)
    """
    d = dm.copy()
    # Dates & sort
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"], errors="coerce")
    d = d.dropna(subset=["GAME_DATE"])
    d = d.sort_values(["PLAYER", "GAME_DATE"])

    # Ensure numeric for base stats and LINE/ACTUAL
    for c in STAT_COLS + ["LINE", "ACTUAL", "DidHitOver"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Group for shifting
    g = d.groupby("PLAYER", group_keys=False)

    # Rolling means of the target-driving stat & MIN, all shifted by 1 (no leakage)
    for col in [stat_col, "MIN"]:
        if col in d.columns:
            d[f"r3_prev_{col}"] = g[col].rolling(3, min_periods=1).mean().shift(1)
            d[f"r5_prev_{col}"] = g[col].rolling(5, min_periods=1).mean().shift(1)

    # Simple usage proxy from previous game
    if {"PTS", "REB", "AST", "MIN"}.issubset(d.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            usage = ( (d["PTS"] + d["REB"] + d["AST"]) / d["MIN"] ) * 100.0
        d["USAGE_LITE_PREV"] = g["MIN"].apply(lambda s: s.shift(1))  # build an index for shift
        d["USAGE_LITE_PREV"] = ((g["PTS"].shift(1) + g["REB"].shift(1) + g["AST"].shift(1)) /
                                 g["MIN"].shift(1) * 100.0)
        d["USAGE_LITE_PREV"] = d["USAGE_LITE_PREV"].replace([np.inf, -np.inf], np.nan)

    # Threes rate previous (helps 3PM)
    if {"FG3M", "PTS"}.issubset(d.columns):
        d["THREES_RATE_PREV"] = g["FG3M"].shift(1) / g["PTS"].shift(1)
        d["THREES_RATE_PREV"] = d["THREES_RATE_PREV"].replace([np.inf, -np.inf], np.nan)

    # Line itself is a strong signal + distance from previous rolling mean
    d["LINE"] = pd.to_numeric(d["LINE"], errors="coerce")
    if f"r5_prev_{stat_col}" in d.columns:
        d["roll5_minus_line"] = d[f"r5_prev_{stat_col}"] - d["LINE"]

    # Bring in optional context (already dated per-row if your log contains them)
    # We just ensure numeric & fillna later
    present_context = [c for c in OPTIONAL_CONTEXT_COLS if c in d.columns]
    for c in present_context:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Targets
    d = d.dropna(subset=["DidHitOver", "LINE"])
    d["DidHitOver"] = d["DidHitOver"].astype(int)

    # Build feature list
    feature_cols = []
    for c in d.columns:
        if (c.startswith("r3_prev_") or c.startswith("r5_prev_") or
            c in ["USAGE_LITE_PREV", "THREES_RATE_PREV", "LINE", "roll5_minus_line"] or
            c in present_context):
            feature_cols.append(c)

    # Final clean
    for c in feature_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    return d, feature_cols

def main():
    df = safe_read_csv(PROPS_TRAIN_LOG)
    if df.empty:
        logging.error("No props_training_log.csv yet — run your daily pipeline for a few days to collect outcomes.")
        return

    df = normalize(df)

    required_cols = {"PLAYER", "MARKET", "GAME_DATE", "LINE", "ACTUAL", "DidHitOver"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.error(f"props_training_log.csv missing required columns: {missing}")
        return

    summary_rows = []
    feature_map = {}
    markets = ["PTS", "3PM", "REB", "AST", "STL"]

    # Candidate models per market
    candidates = {
        "LogReg": LogisticRegression(max_iter=600, solver="lbfgs", class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=350, max_depth=12, random_state=RANDOM_STATE, class_weight="balanced"),
        "XGB": XGBClassifier(
            n_estimators=450, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE
        )
    }

    for market in markets:
        if market not in df["MARKET"].unique():
            logging.info(f"⏭️  Skipping {market} — no rows yet.")
            continue

        stat_col = {"PTS": "PTS", "3PM": "FG3M", "REB": "REB", "AST": "AST", "STL": "STL"}[market]
        dm = df[df["MARKET"] == market].copy()

        # Safety: ensure stat columns exist (even if all NaN → filled later)
        for c in STAT_COLS:
            if c not in dm.columns:
                dm[c] = np.nan

        dm, feat_cols = build_leakfree_features(dm, stat_col)

        pos = int(dm["DidHitOver"].sum())
        neg = int((1 - dm["DidHitOver"]).sum())
        n   = len(dm)
        logging.info(f"🔹 {market}: samples={n}, positives={pos}, negatives={neg}, features={len(feat_cols)}")

        if n < 100 or pos < MIN_CLASS_POSITIVES or dm["DidHitOver"].nunique() < 2:
            logging.info(f"⏭️  {market}: not enough data yet (need ≥{MIN_CLASS_POSITIVES} positives & class balance).")
            continue

        X = dm[feat_cols]
        y = dm["DidHitOver"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        best_auc, best_name, best_model = -1, None, None
        for name, mdl in candidates.items():
            mdl.fit(X_train, y_train)
            if hasattr(mdl, "predict_proba"):
                y_prob = mdl.predict_proba(X_test)[:, 1]
            else:
                # For models without predict_proba, fallback to decision or label
                try:
                    y_prob = mdl.decision_function(X_test)
                    # map to 0-1 via sigmoid-like transform
                    y_prob = 1 / (1 + np.exp(-y_prob))
                except Exception:
                    y_prob = mdl.predict(X_test).astype(float)

            y_pred = (y_prob >= 0.5).astype(int)
            auc, acc, f1 = eval_metrics(y_test, y_prob, y_pred)
            summary_rows.append({"market": market, "model": name, "AUC": auc, "ACC": acc, "F1": f1, "n": int(n), "pos": int(pos), "neg": int(neg)})

            if (not np.isnan(auc)) and auc > best_auc:
                best_auc, best_name, best_model = auc, name, mdl

        # Save best model
        if best_model is not None:
            out_path = OUT_MODELS[market]
            joblib.dump({"model": best_model, "features": feat_cols}, out_path)
            logging.info(f"✅ Saved best {market} model → {out_path} ({best_name}, AUC={best_auc:.3f})")
            feature_map[market] = feat_cols
        else:
            logging.info(f"❌ No model saved for {market} (insufficient/unstable data).")

    # Summary & feature map
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)
        logging.info(f"📄 Wrote {SUMMARY_OUT} ({len(summary_rows)} rows)")
    else:
        logging.info("No summary to write yet — collect more prop outcomes.")

    if feature_map:
        with open(FEATURE_MAP_OUT, "w", encoding="utf-8") as f:
            json.dump(feature_map, f, indent=2)
        logging.info(f"🧭 Saved feature columns per market → {FEATURE_MAP_OUT}")

if __name__ == "__main__":
    main()