# dashboard_app.py — NBA Player Props Dashboard (logos + optional headshots + rolling trend)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="NBA Player Props Dashboard", layout="wide")

# ------------------------------
# Data loaders (cached)
# ------------------------------
@st.cache_data
def load_today():
    try:
        df = pd.read_csv("nba_prop_predictions_today.csv")
        # standardize types
        for c in ["FINAL_OVER_PROB"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # keep only supported markets (defensive in case)
        return df
    except Exception:
        st.error("⚠️ Could not load nba_prop_predictions_today.csv")
        return pd.DataFrame()

@st.cache_data
def load_team_logos():
    try:
        logos = pd.read_csv("team_logos.csv")  # columns: TEAM,TEAM_FULL,LOGO_URL
        logos["TEAM"] = logos["TEAM"].astype(str).str.upper().str.strip()
        return logos
    except Exception:
        return pd.DataFrame(columns=["TEAM", "TEAM_FULL", "LOGO_URL"])

@st.cache_data
def load_headshots():
    try:
        hs = pd.read_csv("player_headshots.csv")  # columns: PLAYER,PHOTO_URL
        hs["PLAYER"] = hs["PLAYER"].astype(str).str.strip()
        return hs
    except Exception:
        return pd.DataFrame(columns=["PLAYER", "PHOTO_URL"])

@st.cache_data
def load_props_history():
    """
    Optional: props_training_log.csv used to render rolling trend sparkline.
    Expected cols: GAME_DATE, PLAYER, MARKET, LINE, ACTUAL, DidHitOver
    (from our earlier pipeline that logs prop outcomes post-game)
    """
    try:
        df = pd.read_csv("props_training_log.csv")
        # Normalize
        df["PLAYER"] = df.get("PLAYER", df.get("PLAYER_NAME", "")).astype(str).str.strip()
        df["MARKET"] = df["MARKET"].astype(str).str.upper().str.strip()
        # Parse date robustly
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        # Coerce DidHitOver
        if "DidHitOver" in df.columns:
            df["DidHitOver"] = pd.to_numeric(df["DidHitOver"], errors="coerce").fillna(0).astype(int)
        return df.dropna(subset=["GAME_DATE"])
    except Exception:
        return pd.DataFrame()

df_today = load_today()
team_logos = load_team_logos()
headshots = load_headshots()
df_hist = load_props_history()

if df_today.empty:
    st.stop()

# Build quick lookup maps
team_logo_map = {r["TEAM"]: r["LOGO_URL"] for _, r in team_logos.iterrows()} if not team_logos.empty else {}
headshot_map = {r["PLAYER"]: r["PHOTO_URL"] for _, r in headshots.iterrows()} if not headshots.empty else {}

# ------------------------------
# Helpers
# ------------------------------
def get_team_logo(team_abbr: str) -> str | None:
    if not team_abbr:
        return None
    return team_logo_map.get(str(team_abbr).upper().strip())

def get_player_photo(player_name: str) -> str | None:
    if not player_name:
        return None
    url = headshot_map.get(player_name.strip())
    return url if isinstance(url, str) and len(url) > 0 else None

def compute_rolling_trend(player: str, market: str, window: int = 5, horizon: int = 10):
    """
    Returns (dates_list, rolling_pct_list)
    - Take last `horizon` outcomes for given player/market from props_training_log.csv
    - Compute rolling mean over `window` (min_periods=1)
    If history missing, fall back to a flat line using today's RECENT_OVER_PROB if present.
    """
    if not df_hist.empty:
        sub = df_hist[(df_hist["PLAYER"] == player) & (df_hist["MARKET"] == market)].copy()
        if not sub.empty:
            sub = sub.sort_values("GAME_DATE").tail(horizon)
            if "DidHitOver" in sub.columns:
                roll = (
                    sub["DidHitOver"]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .tolist()
                )
                dts = [d.strftime("%Y-%m-%d") if isinstance(d, pd.Timestamp) else str(d) for d in sub["GAME_DATE"]]
                return dts, roll

    # fallback: use today RECENT_OVER_PROB to draw a flat mini-series (3 points)
    sub_today = df_today[(df_today["PLAYER"] == player) & (df_today["MARKET"] == market)]
    if not sub_today.empty and "RECENT_OVER_PROB" in sub_today.columns:
        val = float(sub_today.iloc[0]["RECENT_OVER_PROB"]) if pd.notna(sub_today.iloc[0]["RECENT_OVER_PROB"]) else 0.5
    else:
        val = 0.5
    return ["-3", "-2", "-1"], [val, val, val]

def render_trend_sparkline(player: str, market: str, key: str):
    dates, vals = compute_rolling_trend(player, market, window=5, horizon=10)
    if not vals:
        return
    # Color by slope
    color = "green" if vals[-1] >= vals[0] else "red"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(vals))),
        y=vals,
        mode="lines+markers",
        line=dict(width=2, color=color),
        marker=dict(size=6),
        hovertext=[f"{dates[i]}: {vals[i]*100:.1f}%" for i in range(len(vals))],
        hoverinfo="text"
    ))
    fig.update_layout(
        height=90,
        margin=dict(l=4, r=4, t=6, b=6),
        yaxis=dict(range=[0,1], showgrid=False, tickvals=[0,0.5,1.0], ticktext=["0%","50%","100%"]),
        xaxis=dict(visible=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# ------------------------------
# UI
# ------------------------------
st.title("🏀 NBA Player Props Dashboard")
st.caption("Top overs with injuries/context + rolling over% trend (last ~10 games).")

# Market tabs in consistent order
market_order = ["PTS","3PM","REB","AST","STL"]
markets = [m for m in market_order if m in set(df_today["MARKET"].unique())]
tabs = st.tabs([f"🔥 {m}" for m in markets])

for tab, market in zip(tabs, markets):
    with tab:
        subset = df_today[df_today["MARKET"] == market].copy()
        if subset.empty:
            st.info("No props available for this market today.")
            continue

        # Show top 10 by our final probability
        subset = subset.sort_values("FINAL_OVER_PROB", ascending=False).head(10)

        # Cards
        for i, (_, row) in enumerate(subset.iterrows()):
            player = str(row["PLAYER"])
            team = str(row.get("TEAM", "") or "")
            prop_name = f"{row['PROP_NAME']} o{row['LINE']}"
            prob_pct = f"{float(row['FINAL_OVER_PROB'])*100:.1f}%" if pd.notna(row["FINAL_OVER_PROB"]) else "—"
            rec_prob = row.get("RECENT_OVER_PROB")
            rec_n = int(row.get("RECENT_N", 0))
            rec_text = f"{float(rec_prob)*100:.1f}% ({rec_n}g)" if pd.notna(rec_prob) else "—"
            inj = str(row.get("INJ_Status","Active"))
            season_val = float(row.get("SEASON_VAL", 0.0))

            # Ensure unique element keys
            block_key = f"blk_{market}_{i}_{player.replace(' ','_')}"
            chart_key = f"chart_{market}_{i}_{player.replace(' ','_')}"

            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 4, 2])

                with c1:
                    # Player headshot (optional), then team logo
                    photo_url = get_player_photo(player)
                    if photo_url:
                        st.image(photo_url, width=90, caption=None)
                    logo_url = get_team_logo(team)
                    if logo_url:
                        st.image(logo_url, width=48)

                with c2:
                    st.subheader(player)
                    st.write(f"**{prop_name}**")
                    meta_bits = []
                    if team:
                        meta_bits.append(f"Team: `{team}`")
                    if inj:
                        meta_bits.append(f"Injury: {inj}")
                    st.write(" | ".join(meta_bits) if meta_bits else "\u00A0")
                    # Rolling trend sparkline
                    st.caption("Rolling over% (last ~10 games)")
                    render_trend_sparkline(player, market, key=chart_key)

                with c3:
                    st.metric("Prob. Over", prob_pct)
                    st.write(f"Recent Hit Rate: **{rec_text}**")
                    st.write(f"Season {row['PROP_NAME'].split()[0]}: **{season_val:.1f}**")

        st.divider()
        st.caption("Tip: rolling trend uses your `props_training_log.csv` if present; otherwise it falls back to today’s recent rate.")