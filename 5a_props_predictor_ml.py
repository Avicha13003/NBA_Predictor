# 5_props_predictor.py — Prop-line over probabilities with injuries + ML blending
import os, logging, joblib
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ==== Inputs ====
TODAY_CSV        = "nba_today_stats.csv"
TEAM_CONTEXT_CSV = "team_context.csv"
INJURIES_CSV     = "rotowire_injuries.csv"
PROPS_CSV        = "props_today.csv"
GAME_LOG_CSV     = "player_game_log.csv"

# ==== Models ====
MODELS = {
    "PTS": "model_over_pts.pkl",
    "3PM": "model_over_3pm.pkl",
    "REB": "model_over_reb.pkl",
    "AST": "model_over_ast.pkl",
    "STL": "model_over_stl.pkl",
}

# ==== Outputs ====
OUT_ALL      = "nba_prop_predictions_today.csv"
OUT_TOP_PTS  = "top10_over_pts.csv"
OUT_TOP_3PM  = "top10_over_3pm.csv"
OUT_TOP_REB  = "top10_over_reb.csv"
OUT_TOP_AST  = "top10_over_ast.csv"
OUT_TOP_STL  = "top10_over_stl.csv"

# ==== Config ====
RECENT_N, MIN_RECENT_GMS, MAX_Z_CLAMP = 8, 3, 2.5
ML_WEIGHT = 0.65  # how heavily to trust the model vs heuristic

SUPPORTED_MARKETS = {
    "PTS": ("PTS",  "Points"),
    "3PM": ("FG3M", "3-Pointers Made"),
    "REB": ("REB",  "Rebounds"),
    "AST": ("AST",  "Assists"),
    "STL": ("STL",  "Steals"),
}

# ---------- Utilities ----------
def safe_read_csv(p):
    try:
        return pd.read_csv(p)
    except: return pd.DataFrame()

def ensure_cols(df, cols, fill=0.0):
    for c in cols:
        df[c] = pd.to_numeric(df.get(c, fill), errors="coerce").fillna(fill)
    return df

def normalize_names(s): return s.astype(str).str.strip()
def normalize_team(s): return s.astype(str).str.strip().str.upper()
def fmt_pct(x): return f"{x*100:.1f}%" if pd.notna(x) else "~0%"
def safe_int(v, d=0): return int(v) if pd.notna(v) else d

# ---------- Injury logic ----------
def attach_injuries(today, inj_path):
    inj = safe_read_csv(inj_path)
    if inj.empty:
        today["INJ_Status"], today["IS_OUT"], today["IS_LIMITED"] = "Active", 0, 0
        return today
    inj["Player"] = normalize_names(inj.get("Player", ""))
    inj["Team"] = normalize_team(inj.get("Team", ""))
    inj["Status"] = inj.get("Status", "").astype(str)
    inj["IS_OUT"] = inj["Status"].str.contains("Out", case=False).astype(int)
    inj["IS_LIMITED"] = inj["Status"].str.contains("Questionable|GTD|Probable", case=False).astype(int)
    inj = inj.rename(columns={"Player": "PLAYER", "Team": "TEAM", "Status": "INJ_Status"})
    return today.merge(inj[["PLAYER", "TEAM", "INJ_Status", "IS_OUT", "IS_LIMITED"]], on=["PLAYER","TEAM"], how="left").fillna({"INJ_Status":"Active","IS_OUT":0,"IS_LIMITED":0})

def injury_adj(row):
    if safe_int(row.get("IS_OUT")) == 1: return "out"
    if safe_int(row.get("IS_LIMITED")) == 1: return -0.15
    return 0.0

# ---------- Base model ----------
def recent_over_prob(gl, player, stat_col, line):
    df = gl[gl["PLAYER"]==player]
    if df.empty: return (np.nan,0,np.nan)
    df[stat_col] = pd.to_numeric(df[stat_col], errors="coerce")
    df = df.sort_values("GAME_DATE").tail(RECENT_N)
    if df.empty: return (np.nan,0,np.nan)
    n = len(df)
    hits = (df[stat_col] >= line).sum()
    return (hits/n, n, df[stat_col].mean()-line)

def base_over_model(row, mkt, recent_prob, n_recent, mean_minus_line):
    stat,_=SUPPORTED_MARKETS[mkt]
    line = float(row["LINE"])
    val  = float(row.get(stat,0))
    s1=0.5+np.clip((val-line)/{"PTS":10,"3PM":2,"REB":4,"AST":4,"STL":2}[mkt],-0.5,0.5)
    s2,s3=(recent_prob if not np.isnan(recent_prob) else 0.5, 0.5+np.clip(mean_minus_line/10,-0.3,0.3))
    return np.clip(0.45*s1+0.45*s2+0.10*s3,0.01,0.99)

def context_adj(row,mkt):
    c=lambda x:np.clip(float(x),-MAX_Z_CLAMP,MAX_Z_CLAMP) if pd.notna(x) else 0
    defz,oppz=c(row.get("DEF_RATING_Z",0)),c(row.get("OPP_PTS_Z",0))
    b2b,tf=float(row.get("IS_B2B_NUM",0)),float(row.get("TRAVEL_FATIGUE",0))
    boost={"PTS":0.04*defz+0.02*oppz,"3PM":0.04*defz+0.02*oppz,"REB":0.015*defz,"AST":0.02*defz}.get(mkt,0)
    return boost - (0.06*b2b+0.04*tf)

# ---------- ML helpers ----------
def load_models():
    loaded={}
    for k,p in MODELS.items():
        if os.path.exists(p):
            loaded[k]=joblib.load(p)
            logging.info(f"✅ Loaded ML model for {k} ({p})")
        else:
            logging.warning(f"⚠️ Missing model: {p}")
    return loaded

def predict_ml_prob(model, row, mkt):
    # Use the same feature set as training (rolling features + line)
    features=[]
    for c in ["LINE","PTS","FG3M","REB","AST","STL","MIN"]:
        features.append(float(row.get(c,0)))
    X=np.array([features])
    try:
        if hasattr(model,"predict_proba"):
            return model.predict_proba(X)[:,1][0]
        return float(model.predict(X)[0])
    except Exception as e:
        logging.debug(f"ML prediction error for {row.get('PLAYER')}: {e}")
        return np.nan

# ---------- Main ----------
def main():
    today,context,inj,props,gl=[safe_read_csv(f) for f in [TODAY_CSV,TEAM_CONTEXT_CSV,INJURIES_CSV,PROPS_CSV,GAME_LOG_CSV]]
    if today.empty or context.empty or props.empty:
        logging.error("Missing required inputs.");return
    today=attach_injuries(today,INJURIES_CSV)
    today["PLAYER"]=normalize_names(today["PLAYER"])
    today["TEAM"]=normalize_team(today["TEAM"])
    props["PLAYER"]=normalize_names(props.get("PLAYER",pd.Series("",index=props.index)))
    props["TEAM"]=normalize_team(props.get("TEAM",pd.Series("",index=props.index)))

    merged=props.merge(today[["PLAYER","TEAM"]+[c for c in today.columns if c not in ["PLAYER","TEAM"]]],on="PLAYER",how="left",suffixes=("_PROP",""))
    merged["TEAM"]=merged["TEAM_PROP"].combine_first(merged["TEAM"])
    team_map=today[["PLAYER","TEAM"]].drop_duplicates().set_index("PLAYER")["TEAM"].to_dict()
    merged["TEAM"]=merged.apply(lambda r:r["TEAM"] if pd.notna(r["TEAM"]) and r["TEAM"]!="" else team_map.get(r["PLAYER"],""),axis=1)
    merged.drop(columns=["TEAM_PROP"],inplace=True,errors="ignore")
    merged.drop_duplicates(subset=["PLAYER","MARKET"],inplace=True)
    merged=ensure_cols(merged,["LINE","PTS","FG3M","REB","AST","STL"],0.0)

    gl["PLAYER"]=normalize_names(gl.get("PLAYER",gl.get("PLAYER_NAME","")))
    for s in ["PTS","FG3M","REB","AST","STL"]: gl[s]=pd.to_numeric(gl.get(s,0),errors="coerce")

    models=load_models()

    rows=[]
    for _,r in merged.iterrows():
        mkt=r["MARKET"]
        if mkt not in SUPPORTED_MARKETS: continue
        stat_col,_=SUPPORTED_MARKETS[mkt]
        rp,n_recent,diff=recent_over_prob(gl,r["PLAYER"],stat_col,r["LINE"])
        base=base_over_model(r,mkt,rp,n_recent,diff)
        adj=np.clip(base+context_adj(r,mkt),0.01,0.99)
        injadj=injury_adj(r)
        final=0.0 if injadj=="out" else np.clip(adj*(1.0+injadj),0.0,1.0)

        ml_prob=np.nan
        if mkt in models:
            ml_prob=predict_ml_prob(models[mkt],r,mkt)

        blended=np.clip((ML_WEIGHT*ml_prob + (1-ML_WEIGHT)*final) if not np.isnan(ml_prob) else final,0.0,1.0)

        rows.append({
            "PLAYER":r["PLAYER"],"TEAM":r.get("TEAM",""),
            "MARKET":mkt,"PROP_NAME":SUPPORTED_MARKETS[mkt][1],"LINE":r["LINE"],
            "FINAL_OVER_PROB":final,"ML_PROB":ml_prob,"BLENDED_PROB":blended,
            "BLENDED_PROB_PCT":fmt_pct(blended),"INJ_Status":r.get("INJ_Status","Active"),
            "IS_OUT":safe_int(r.get("IS_OUT",0)),"IS_LIMITED":safe_int(r.get("IS_LIMITED",0))
        })

    out=pd.DataFrame(rows)
    if out.empty: logging.error("No output.");return
    out.to_csv(OUT_ALL,index=False)
    for m,f in [("PTS",OUT_TOP_PTS),("3PM",OUT_TOP_3PM),("REB",OUT_TOP_REB),("AST",OUT_TOP_AST),("STL",OUT_TOP_STL)]:
        out[out["MARKET"]==m].sort_values("BLENDED_PROB",ascending=False).head(10).to_csv(f,index=False)
    logging.info(f"✅ Saved ML-blended prop predictions → {OUT_ALL}")

    print("\n===== TOP 3 ML-BLENDED OVERS =====")
    for m,(_,n) in SUPPORTED_MARKETS.items():
        sub=out[out["MARKET"]==m].sort_values("BLENDED_PROB",ascending=False).head(3)
        if sub.empty: continue
        print(f"\n{n}:")
        for _,row in sub.iterrows():
            print(f"  {row['PLAYER']:<25} {row['TEAM']:<4} o{row['LINE']} → {row['BLENDED_PROB_PCT']}")
    print("==================================\n")

if __name__=="__main__":
    main()