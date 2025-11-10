# nba_train_model.py — trains multiple models and selects the best (verbose console)
import os, logging, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===== Files =====
HISTORY_LOG = "player_game_log.csv"
MODEL_HIT30 = "best_model_hit30.pkl"
MODEL_HIT4THREES = "best_model_hit4threes.pkl"
SUMMARY_OUT = "training_summary.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ===== Utility =====
def safe_read_csv(path):
    if not os.path.exists(path):
        logging.error(f"{path} not found.")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.error(f"Could not read {path}: {e}")
        return pd.DataFrame()

def evaluate_model(name, model, X_test, y_test):
    """Return metrics dictionary"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    return {"model": name, "AUC": auc, "ACC": acc, "F1": f1}

def feature_importance(model, feature_names):
    """Extract feature importance or coefficients"""
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        vals = np.zeros(len(feature_names))
    return dict(zip(feature_names, vals))

# ===== Main =====
def main():
    df = safe_read_csv(HISTORY_LOG)
    if df.empty:
        logging.error("No player_game_log.csv found or empty.")
        return

    # Basic normalization
    for c in ["PTS", "FG3M", "REB", "AST", "MIN"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)

    if "PLAYER_NAME" in df.columns and "PLAYER" not in df.columns:
        df.rename(columns={"PLAYER_NAME": "PLAYER"}, inplace=True)

    # --- Define features ---
    base_feats = [
        "PTS","FG3M","REB","AST","MIN",
        "r3_pts","r5_pts","r3_fg3m","r5_fg3m",
        "USAGE_LITE","THREES_RATE","IS_HOME",
        "DEF_RATING_Z","OPP_DEF_RATING_Z","OPP_OPP_PTS_Z","OPP_PTS_Z",
        "DAYS_REST","IS_B2B_NUM","TRAVEL_MILES","TRAVEL_FATIGUE"
    ]
    # Make sure missing columns exist
    for c in base_feats:
        if c not in df.columns:
            df[c] = 0.0

    df["didHit30"] = df["didHit30"].fillna(0).astype(int)
    df["didHit4Threes"] = df["didHit4Threes"].fillna(0).astype(int)

    logging.info(f"Samples: {len(df)} | {df['didHit30'].sum()} >=30pts | {df['didHit4Threes'].sum()} >=4threes")

    # Drop missing targets
    df = df.dropna(subset=["didHit30","didHit4Threes"])
    X = df[base_feats].fillna(0.0)

    # Train/test split
    X_train, X_test, y30_train, y30_test = train_test_split(X, df["didHit30"], test_size=0.25, random_state=42, stratify=df["didHit30"])
    _, _, y4_train, y4_test = train_test_split(X, df["didHit4Threes"], test_size=0.25, random_state=42, stratify=df["didHit4Threes"])

    # --- Define candidate models ---
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, solver="lbfgs"),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.07, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", use_label_encoder=False, random_state=42
        )
    }

    all_metrics = []
    best_auc_30, best_auc_4 = -1, -1
    best_model_30, best_model_4 = None, None
    best_name_30, best_name_4 = None, None

    # --- Train all models for both targets ---
    for name, model in models.items():
        logging.info(f"🔹 Training {name} for didHit30...")
        model.fit(X_train, y30_train)
        m30 = evaluate_model(name, model, X_test, y30_test)
        logging.info(f"→ AUC={m30['AUC']:.3f} | ACC={m30['ACC']:.3f} | F1={m30['F1']:.3f}")

        fi30 = feature_importance(model, base_feats)
        all_metrics.append({**m30, "target": "didHit30", "best": False})

        if m30["AUC"] > best_auc_30:
            best_auc_30, best_model_30, best_name_30 = m30["AUC"], model, name

        logging.info(f"🔹 Training {name} for didHit4Threes...")
        model.fit(X_train, y4_train)
        m4 = evaluate_model(name, model, X_test, y4_test)
        logging.info(f"→ AUC={m4['AUC']:.3f} | ACC={m4['ACC']:.3f} | F1={m4['F1']:.3f}")

        fi4 = feature_importance(model, base_feats)
        all_metrics.append({**m4, "target": "didHit4Threes", "best": False})

        if m4["AUC"] > best_auc_4:
            best_auc_4, best_model_4, best_name_4 = m4["AUC"], model, name

    # --- Save best models ---
    if best_model_30 is not None:
        joblib.dump(best_model_30, MODEL_HIT30)
        logging.info(f"✅ Saved best model for 30pt → {MODEL_HIT30} ({best_name_30}, AUC={best_auc_30:.3f})")

    if best_model_4 is not None:
        joblib.dump(best_model_4, MODEL_HIT4THREES)
        logging.info(f"✅ Saved best model for 4+Threes → {MODEL_HIT4THREES} ({best_name_4}, AUC={best_auc_4:.3f})")

    # Mark best ones in summary
    for m in all_metrics:
        if m["target"] == "didHit30" and m["model"] == best_name_30:
            m["best"] = True
        if m["target"] == "didHit4Threes" and m["model"] == best_name_4:
            m["best"] = True

    # --- Save summary CSV ---
    pd.DataFrame(all_metrics).to_csv(SUMMARY_OUT, index=False)
    logging.info(f"📄 Wrote {SUMMARY_OUT} ({len(all_metrics)} rows)")

    # --- Log feature correlations ---
    corrs = X.corrwith(df["didHit30"]).to_frame("corr_hit30").join(X.corrwith(df["didHit4Threes"]).to_frame("corr_hit4threes"))
    corrs.to_csv("feature_correlations.csv")
    logging.info("📈 Saved feature_correlations.csv")

if __name__ == "__main__":
    main()