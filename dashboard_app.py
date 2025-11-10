<<<<<<< HEAD
# dashboard_app.py — NBA Player Props Dashboard (final patched version)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="NBA Player Props Dashboard", layout="wide")

# ---------- Helpers ----------
ALT_TEAM_MAP = {"GS": "GSW", "NO": "NOP", "SA": "SAS", "NY": "NYK", "PHO": "PHX"}

def norm_team(x: str) -> str:
    if not isinstance(x, str): return ""
    x = x.strip().upper()
    return ALT_TEAM_MAP.get(x, x)

def pct(x): 
    return f"{x*100:.1f}%" if pd.notna(x) else "—"

def color_for(val, hi_good=True):
    if pd.isna(val): return "#999999"
    if hi_good:
        return "#2ecc71" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#e74c3c")
    else:
        return "#e74c3c" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#2ecc71")

@st.cache_data
def load_csv(path, **kwargs):
    try:
        df = pd.read_csv(path, **kwargs)
        st.toast(f"Loaded {path} ({len(df)} rows)", icon="✅")
        return df
    except Exception:
        st.warning(f"⚠️ Could not load {path}")
        return pd.DataFrame()

@st.cache_data
def load_predictions():
    df = load_csv("nba_prop_predictions_today.csv")
    if df.empty: return df
    df["PLAYER"] = df["PLAYER"].astype(str).str.strip()
    df["TEAM"] = df["TEAM"].astype(str).map(norm_team)
    return df

@st.cache_data
def load_team_logos():
    logos = load_csv("team_logos.csv")
    if logos.empty: 
        return pd.DataFrame(columns=["TEAM","TEAM_FULL","LOGO_URL","PRIMARY_COLOR","SECONDARY_COLOR"])
    logos["TEAM"] = logos["TEAM"].astype(str).str.strip().str.upper()
    return logos

@st.cache_data
def load_headshots():
    heads = load_csv("player_headshots.csv")
    if heads.empty:
        return pd.DataFrame(columns=["PLAYER","PHOTO_URL"])
    col_player = "PLAYER" if "PLAYER" in heads.columns else "player"
    col_url = "PHOTO_URL" if "PHOTO_URL" in heads.columns else "image_url"
    heads = heads.rename(columns={col_player: "PLAYER", col_url: "PHOTO_URL"})
    heads["PLAYER_NORM"] = heads["PLAYER"].astype(str).str.lower().str.strip()
    return heads[["PLAYER","PLAYER_NORM","PHOTO_URL"]]

@st.cache_data
def load_game_log():
    return load_csv("player_game_log.csv")

@st.cache_data
def load_team_context():
    return load_csv("team_context.csv")

def compute_player_context(gl_all: pd.DataFrame, player: str, market: str, team_abbr: str):
    """Compute averages, volatility, last game, hit series safely (no caching)."""
    stat_map = {"PTS":"PTS","3PM":"FG3M","REB":"REB","AST":"AST","STL":"STL"}
    hit_col_map = {
        "PTS":"didHitOver_PTS","3PM":"didHitOver_FG3M","REB":"didHitOver_REB",
        "AST":"didHitOver_AST","STL":"didHitOver_STL",
    }
    out = {"last5_mean":np.nan,"last5_std":np.nan,"last_game":None,
           "series_list":[],"hit_rate_recent":np.nan,"n_recent":0}
    if gl_all.empty or stat_map[market] not in gl_all.columns:
        return out

    g = gl_all.copy()
    g["PLAYER"] = g["PLAYER"].astype(str).str.strip()
    g = g[g["PLAYER"] == player].copy()
    if g.empty: return out

    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"], errors="coerce")
    g = g.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")
    stat_col = stat_map[market]
    g[stat_col] = pd.to_numeric(g[stat_col], errors="coerce")

    last5 = g.tail(5)
    out["last5_mean"] = float(last5[stat_col].mean()) if len(last5) else np.nan
    out["last5_std"] = float(last5[stat_col].std(ddof=0)) if len(last5) else np.nan

    lg = g.tail(1)
    if not lg.empty:
        out["last_game"] = {
            "date": lg["GAME_DATE"].iloc[0].date().isoformat(),
            "team": str(lg["TEAM"].iloc[0]),
            "val": float(lg[stat_col].iloc[0]),
        }

    hit_col = hit_col_map.get(market)
    if hit_col and hit_col in g.columns:
        s = pd.to_numeric(g[hit_col], errors="coerce").fillna(0).tail(8)
        out["series_list"] = s.tolist()
        out["n_recent"] = int(len(s))
        out["hit_rate_recent"] = float(s.mean()) if len(s) else np.nan

    return out

def opponent_context(ctx_df: pd.DataFrame, team_abbr: str):
    if ctx_df.empty: 
        return None
    c = ctx_df.copy()
    c["TEAM_ABBREVIATION"] = c["TEAM_ABBREVIATION"].astype(str).map(norm_team)
    for cand in ["OPP_DEF_RATING_RANK","DEF_RATING_RANK","OPP_DEF_RANK"]:
        if cand in c.columns:
            row = c[c["TEAM_ABBREVIATION"] == team_abbr].tail(1)
            if not row.empty:
                return int(pd.to_numeric(row[cand], errors="coerce").fillna(0).iloc[0])
    return None

def sparkline(series_list, color="#2ecc71"):
    if not series_list or len(series_list)==0:
        return None
    data = pd.DataFrame({"idx": range(1, len(series_list)+1), "val": series_list})
    chart = alt.Chart(data).mark_line(point=False).encode(
        x=alt.X("idx:Q", axis=None),
        y=alt.Y("val:Q", axis=None, scale=alt.Scale(domain=[0,1])),
        color=alt.value(color)
    ).properties(height=30)
    return chart

# ---------- Load Data ----------
preds = load_predictions()
logos = load_team_logos()
heads = load_headshots()
gl = load_game_log()
ctx = load_team_context()

if preds.empty:
    st.stop()

preds["PLAYER_NORM"] = preds["PLAYER"].str.lower().str.strip()
if not heads.empty:
    preds = preds.merge(heads[["PLAYER_NORM","PHOTO_URL"]], on="PLAYER_NORM", how="left")
else:
    preds["PHOTO_URL"] = ""

if not logos.empty:
    preds = preds.merge(logos, on="TEAM", how="left")
else:
    preds[["TEAM_FULL","LOGO_URL","PRIMARY_COLOR","SECONDARY_COLOR"]] = ["","","",""]

preds["PHOTO_URL"] = preds["PHOTO_URL"].fillna("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png")
preds["LOGO_URL"] = preds["LOGO_URL"].fillna("")
preds["PRIMARY_COLOR"] = preds["PRIMARY_COLOR"].fillna("#333333")
preds["SECONDARY_COLOR"] = preds["SECONDARY_COLOR"].fillna("#777777")

# ---------- Sidebar Filters ----------
st.sidebar.title("🔎 Filters")
teams = ["All Teams"] + sorted([t for t in preds["TEAM"].dropna().unique().tolist() if t])
team_pick = st.sidebar.selectbox("Select Team", teams)

if team_pick != "All Teams":
    player_opts = ["All Players"] + sorted(preds.loc[preds["TEAM"]==team_pick,"PLAYER"].unique().tolist())
else:
    player_opts = ["All Players"] + sorted(preds["PLAYER"].unique().tolist())

player_pick = st.sidebar.selectbox("Select Player", player_opts)

sort_by = st.sidebar.selectbox(
    "Sort by",
    ["Prob Over (desc)","Line Edge (SEASON_VAL - LINE)","Recent Hit Rate","Volatility (std, asc)"]
)

# ---------- Filtering ----------
view = preds.copy()
if team_pick != "All Teams":
    view = view[view["TEAM"] == team_pick]
if player_pick != "All Players":
    view = view[view["PLAYER"] == player_pick]

# ---------- Header ----------
st.markdown("### 🏀 NBA Player Props Dashboard")
st.caption("Daily NBA Trends & Predictions — updated at least 2 hours before first tip.")

markets = view["MARKET"].dropna().unique().tolist()
if not markets:
    st.info("No props found.")
    st.stop()

tabs = st.tabs([m for m in markets])

# ---------- Per Market ----------
for tab, market in zip(tabs, markets):
    with tab:
        sub = view[view["MARKET"]==market].copy()
        if sub.empty:
            st.info("No data for this market.")
            continue

        ctx_rows = []
        for _, r in sub.iterrows():
            ctxp = compute_player_context(gl, r["PLAYER"], market, r.get("TEAM",""))
            ctx_rows.append({
                "PLAYER": r["PLAYER"],
                "last5_mean": ctxp["last5_mean"],
                "last5_std": ctxp["last5_std"],
                "hit_rate_recent": ctxp["hit_rate_recent"],
                "n_recent": ctxp["n_recent"],
                "last_game": ctxp["last_game"],
                "series_list": ctxp["series_list"],
            })
        ctx_df = pd.DataFrame(ctx_rows)

        sub = sub.merge(ctx_df, on="PLAYER", how="left")
        sub["line_edge"] = (pd.to_numeric(sub.get("SEASON_VAL",0), errors="coerce") - 
                            pd.to_numeric(sub.get("LINE",0), errors="coerce"))

        if sort_by == "Prob Over (desc)":
            sub = sub.sort_values("FINAL_OVER_PROB", ascending=False)
        elif sort_by == "Line Edge (SEASON_VAL - LINE)":
            sub = sub.sort_values("line_edge", ascending=False)
        elif sort_by == "Recent Hit Rate":
            sub = sub.sort_values("hit_rate_recent", ascending=False)
        elif sort_by == "Volatility (std, asc)":
            sub = sub.sort_values(sub["last5_std"].fillna(9e9), ascending=True)

        st.subheader(f"{market} · Top Overs")
        st.divider()

        for _, row in sub.head(10).iterrows():
            prim = row.get("PRIMARY_COLOR","#333333")
            sec  = row.get("SECONDARY_COLOR","#777777")

            with st.container(border=True):
                st.markdown(f"<div style='height:4px;background:linear-gradient(90deg,{prim},{sec});border-radius:4px;'></div>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns([1.0, 3.0, 2.2, 1.4])

                with c1:
                    if isinstance(row.get("PHOTO_URL",""), str) and row["PHOTO_URL"].startswith("http"):
                        st.image(row["PHOTO_URL"], width=86)
                    else:
                        st.image("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png", width=86)
                    if isinstance(row.get("LOGO_URL",""), str) and row["LOGO_URL"].startswith("http"):
                        st.image(row["LOGO_URL"], width=42)

                with c2:
                    st.markdown(f"#### {row['PLAYER']}")
                    st.markdown(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    opp_rank = opponent_context(ctx, row.get("TEAM",""))
                    team_badge = f"<span style='background:{prim};color:white;padding:2px 6px;border-radius:6px;font-size:0.8rem'>{row.get('TEAM','')}</span>"
                    inj = row.get("INJ_Status","Active")
                    st.markdown(f"Team: {team_badge} &nbsp;|&nbsp; Injury: **{inj}**", unsafe_allow_html=True)
                    if row.get("last_game"):
                        lg = row["last_game"]
                        st.caption(f"Last game ({lg['date']}): {lg['val']} {market} — Team: {lg['team']}{' | Opp Def Rank: ' + str(opp_rank) if opp_rank else ''}")

                with c3:
                    st.metric("Prob. Over", row.get("FINAL_OVER_PROB_PCT","—"))
                    hr = row.get("hit_rate_recent", np.nan)
                    n  = int(row.get("n_recent", 0) or 0)
                    st.markdown(
                        f"<div style='color:{color_for(hr)}'>Recent Hit Rate: {pct(hr)} ({n}g)</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"Line edge: {row['line_edge']:+.2f}")

                with c4:
                    chart = sparkline(row.get("series_list"), color=color_for(row.get("hit_rate_recent"), hi_good=True))
                    if chart is not None:
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.caption("No recent series")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.divider()

=======
# dashboard_app.py — NBA Player Props Dashboard (final patched version)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="NBA Player Props Dashboard", layout="wide")

# ---------- Helpers ----------
ALT_TEAM_MAP = {"GS": "GSW", "NO": "NOP", "SA": "SAS", "NY": "NYK", "PHO": "PHX"}

def norm_team(x: str) -> str:
    if not isinstance(x, str): return ""
    x = x.strip().upper()
    return ALT_TEAM_MAP.get(x, x)

def pct(x): 
    return f"{x*100:.1f}%" if pd.notna(x) else "—"

def color_for(val, hi_good=True):
    if pd.isna(val): return "#999999"
    if hi_good:
        return "#2ecc71" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#e74c3c")
    else:
        return "#e74c3c" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#2ecc71")

@st.cache_data
def load_csv(path, **kwargs):
    try:
        df = pd.read_csv(path, **kwargs)
        st.toast(f"Loaded {path} ({len(df)} rows)", icon="✅")
        return df
    except Exception:
        st.warning(f"⚠️ Could not load {path}")
        return pd.DataFrame()

@st.cache_data
def load_predictions():
    df = load_csv("nba_prop_predictions_today.csv")
    if df.empty: return df
    df["PLAYER"] = df["PLAYER"].astype(str).str.strip()
    df["TEAM"] = df["TEAM"].astype(str).map(norm_team)
    return df

@st.cache_data
def load_team_logos():
    logos = load_csv("team_logos.csv")
    if logos.empty: 
        return pd.DataFrame(columns=["TEAM","TEAM_FULL","LOGO_URL","PRIMARY_COLOR","SECONDARY_COLOR"])
    logos["TEAM"] = logos["TEAM"].astype(str).str.strip().str.upper()
    return logos

@st.cache_data
def load_headshots():
    heads = load_csv("player_headshots.csv")
    if heads.empty:
        return pd.DataFrame(columns=["PLAYER","PHOTO_URL"])
    col_player = "PLAYER" if "PLAYER" in heads.columns else "player"
    col_url = "PHOTO_URL" if "PHOTO_URL" in heads.columns else "image_url"
    heads = heads.rename(columns={col_player: "PLAYER", col_url: "PHOTO_URL"})
    heads["PLAYER_NORM"] = heads["PLAYER"].astype(str).str.lower().str.strip()
    return heads[["PLAYER","PLAYER_NORM","PHOTO_URL"]]

@st.cache_data
def load_game_log():
    return load_csv("player_game_log.csv")

@st.cache_data
def load_team_context():
    return load_csv("team_context.csv")

def compute_player_context(gl_all: pd.DataFrame, player: str, market: str, team_abbr: str):
    """Compute averages, volatility, last game, hit series safely (no caching)."""
    stat_map = {"PTS":"PTS","3PM":"FG3M","REB":"REB","AST":"AST","STL":"STL"}
    hit_col_map = {
        "PTS":"didHitOver_PTS","3PM":"didHitOver_FG3M","REB":"didHitOver_REB",
        "AST":"didHitOver_AST","STL":"didHitOver_STL",
    }
    out = {"last5_mean":np.nan,"last5_std":np.nan,"last_game":None,
           "series_list":[],"hit_rate_recent":np.nan,"n_recent":0}
    if gl_all.empty or stat_map[market] not in gl_all.columns:
        return out

    g = gl_all.copy()
    g["PLAYER"] = g["PLAYER"].astype(str).str.strip()
    g = g[g["PLAYER"] == player].copy()
    if g.empty: return out

    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"], errors="coerce")
    g = g.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")
    stat_col = stat_map[market]
    g[stat_col] = pd.to_numeric(g[stat_col], errors="coerce")

    last5 = g.tail(5)
    out["last5_mean"] = float(last5[stat_col].mean()) if len(last5) else np.nan
    out["last5_std"] = float(last5[stat_col].std(ddof=0)) if len(last5) else np.nan

    lg = g.tail(1)
    if not lg.empty:
        out["last_game"] = {
            "date": lg["GAME_DATE"].iloc[0].date().isoformat(),
            "team": str(lg["TEAM"].iloc[0]),
            "val": float(lg[stat_col].iloc[0]),
        }

    hit_col = hit_col_map.get(market)
    if hit_col and hit_col in g.columns:
        s = pd.to_numeric(g[hit_col], errors="coerce").fillna(0).tail(8)
        out["series_list"] = s.tolist()
        out["n_recent"] = int(len(s))
        out["hit_rate_recent"] = float(s.mean()) if len(s) else np.nan

    return out

def opponent_context(ctx_df: pd.DataFrame, team_abbr: str):
    if ctx_df.empty: 
        return None
    c = ctx_df.copy()
    c["TEAM_ABBREVIATION"] = c["TEAM_ABBREVIATION"].astype(str).map(norm_team)
    for cand in ["OPP_DEF_RATING_RANK","DEF_RATING_RANK","OPP_DEF_RANK"]:
        if cand in c.columns:
            row = c[c["TEAM_ABBREVIATION"] == team_abbr].tail(1)
            if not row.empty:
                return int(pd.to_numeric(row[cand], errors="coerce").fillna(0).iloc[0])
    return None

def sparkline(series_list, color="#2ecc71"):
    if not series_list or len(series_list)==0:
        return None
    data = pd.DataFrame({"idx": range(1, len(series_list)+1), "val": series_list})
    chart = alt.Chart(data).mark_line(point=False).encode(
        x=alt.X("idx:Q", axis=None),
        y=alt.Y("val:Q", axis=None, scale=alt.Scale(domain=[0,1])),
        color=alt.value(color)
    ).properties(height=30)
    return chart

# ---------- Load Data ----------
preds = load_predictions()
logos = load_team_logos()
heads = load_headshots()
gl = load_game_log()
ctx = load_team_context()

if preds.empty:
    st.stop()

preds["PLAYER_NORM"] = preds["PLAYER"].str.lower().str.strip()
if not heads.empty:
    preds = preds.merge(heads[["PLAYER_NORM","PHOTO_URL"]], on="PLAYER_NORM", how="left")
else:
    preds["PHOTO_URL"] = ""

if not logos.empty:
    preds = preds.merge(logos, on="TEAM", how="left")
else:
    preds[["TEAM_FULL","LOGO_URL","PRIMARY_COLOR","SECONDARY_COLOR"]] = ["","","",""]

preds["PHOTO_URL"] = preds["PHOTO_URL"].fillna("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png")
preds["LOGO_URL"] = preds["LOGO_URL"].fillna("")
preds["PRIMARY_COLOR"] = preds["PRIMARY_COLOR"].fillna("#333333")
preds["SECONDARY_COLOR"] = preds["SECONDARY_COLOR"].fillna("#777777")

# ---------- Sidebar Filters ----------
st.sidebar.title("🔎 Filters")
teams = ["All Teams"] + sorted([t for t in preds["TEAM"].dropna().unique().tolist() if t])
team_pick = st.sidebar.selectbox("Select Team", teams)

if team_pick != "All Teams":
    player_opts = ["All Players"] + sorted(preds.loc[preds["TEAM"]==team_pick,"PLAYER"].unique().tolist())
else:
    player_opts = ["All Players"] + sorted(preds["PLAYER"].unique().tolist())

player_pick = st.sidebar.selectbox("Select Player", player_opts)

sort_by = st.sidebar.selectbox(
    "Sort by",
    ["Prob Over (desc)","Line Edge (SEASON_VAL - LINE)","Recent Hit Rate","Volatility (std, asc)"]
)

# ---------- Filtering ----------
view = preds.copy()
if team_pick != "All Teams":
    view = view[view["TEAM"] == team_pick]
if player_pick != "All Players":
    view = view[view["PLAYER"] == player_pick]

# ---------- Header ----------
st.markdown("### 🏀 NBA Player Props Dashboard")
st.caption("Daily NBA Trends & Predictions — updated at least 2 hours before first tip.")

markets = view["MARKET"].dropna().unique().tolist()
if not markets:
    st.info("No props found.")
    st.stop()

tabs = st.tabs([m for m in markets])

# ---------- Per Market ----------
for tab, market in zip(tabs, markets):
    with tab:
        sub = view[view["MARKET"]==market].copy()
        if sub.empty:
            st.info("No data for this market.")
            continue

        ctx_rows = []
        for _, r in sub.iterrows():
            ctxp = compute_player_context(gl, r["PLAYER"], market, r.get("TEAM",""))
            ctx_rows.append({
                "PLAYER": r["PLAYER"],
                "last5_mean": ctxp["last5_mean"],
                "last5_std": ctxp["last5_std"],
                "hit_rate_recent": ctxp["hit_rate_recent"],
                "n_recent": ctxp["n_recent"],
                "last_game": ctxp["last_game"],
                "series_list": ctxp["series_list"],
            })
        ctx_df = pd.DataFrame(ctx_rows)

        sub = sub.merge(ctx_df, on="PLAYER", how="left")
        sub["line_edge"] = (pd.to_numeric(sub.get("SEASON_VAL",0), errors="coerce") - 
                            pd.to_numeric(sub.get("LINE",0), errors="coerce"))

        if sort_by == "Prob Over (desc)":
            sub = sub.sort_values("FINAL_OVER_PROB", ascending=False)
        elif sort_by == "Line Edge (SEASON_VAL - LINE)":
            sub = sub.sort_values("line_edge", ascending=False)
        elif sort_by == "Recent Hit Rate":
            sub = sub.sort_values("hit_rate_recent", ascending=False)
        elif sort_by == "Volatility (std, asc)":
            sub = sub.sort_values(sub["last5_std"].fillna(9e9), ascending=True)

        st.subheader(f"{market} · Top Overs")
        st.divider()

        for _, row in sub.head(10).iterrows():
            prim = row.get("PRIMARY_COLOR","#333333")
            sec  = row.get("SECONDARY_COLOR","#777777")

            with st.container(border=True):
                st.markdown(f"<div style='height:4px;background:linear-gradient(90deg,{prim},{sec});border-radius:4px;'></div>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns([1.0, 3.0, 2.2, 1.4])

                with c1:
                    if isinstance(row.get("PHOTO_URL",""), str) and row["PHOTO_URL"].startswith("http"):
                        st.image(row["PHOTO_URL"], width=86)
                    else:
                        st.image("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png", width=86)
                    if isinstance(row.get("LOGO_URL",""), str) and row["LOGO_URL"].startswith("http"):
                        st.image(row["LOGO_URL"], width=42)

                with c2:
                    st.markdown(f"#### {row['PLAYER']}")
                    st.markdown(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    opp_rank = opponent_context(ctx, row.get("TEAM",""))
                    team_badge = f"<span style='background:{prim};color:white;padding:2px 6px;border-radius:6px;font-size:0.8rem'>{row.get('TEAM','')}</span>"
                    inj = row.get("INJ_Status","Active")
                    st.markdown(f"Team: {team_badge} &nbsp;|&nbsp; Injury: **{inj}**", unsafe_allow_html=True)
                    if row.get("last_game"):
                        lg = row["last_game"]
                        st.caption(f"Last game ({lg['date']}): {lg['val']} {market} — Team: {lg['team']}{' | Opp Def Rank: ' + str(opp_rank) if opp_rank else ''}")

                with c3:
                    st.metric("Prob. Over", row.get("FINAL_OVER_PROB_PCT","—"))
                    hr = row.get("hit_rate_recent", np.nan)
                    n  = int(row.get("n_recent", 0) or 0)
                    st.markdown(
                        f"<div style='color:{color_for(hr)}'>Recent Hit Rate: {pct(hr)} ({n}g)</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"Line edge: {row['line_edge']:+.2f}")

                with c4:
                    chart = sparkline(row.get("series_list"), color=color_for(row.get("hit_rate_recent"), hi_good=True))
                    if chart is not None:
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.caption("No recent series")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.divider()

>>>>>>> a970c21 (Daily dashboard update)
st.caption("Daily NBA Trends & Predictions — powered by your pipeline • Mobile-friendly • Free on Streamlit Cloud")