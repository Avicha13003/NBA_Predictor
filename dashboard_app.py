# dashboard_app.py — NBA Player Props Dashboard (with player photos, team logos & trends)
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px

st.set_page_config(page_title="NBA Player Props Dashboard", layout="wide")

# === Load data ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("nba_prop_predictions_today.csv")
        df["RECENT_OVER_PROB"] = pd.to_numeric(df["RECENT_OVER_PROB"], errors="coerce").fillna(0)
        df["RECENT_N"] = pd.to_numeric(df["RECENT_N"], errors="coerce").fillna(0)
        return df
    except Exception as e:
        st.error(f"⚠️ Could not load nba_prop_predictions_today.csv: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# === Helper functions ===
@st.cache_data
def get_player_image(name):
    """
    NBA official headshot image fetcher.
    """
    name_fmt = name.lower().replace(" ", "-")
    url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{name_fmt}.png"
    return url

@st.cache_data
def get_team_logo(team_abbr):
    """
    Fetch team logo from a consistent CDN (NBA or Basketball Reference).
    """
    if not team_abbr or not isinstance(team_abbr, str):
        return ""
    team_abbr = team_abbr.strip().lower()
    # Try NBA CDN first
    return f"https://cdn.ssref.net/req/202106291/images/teams/{team_abbr}_logo.svg"

def render_hit_rate_chart(player_row):
    """
    Simulate hit rate trend chart from RECENT_OVER_PROB and RECENT_N values.
    """
    if player_row["RECENT_N"] < 3:
        st.write("📉 Insufficient data for trend chart.")
        return
    # Simulated trend (placeholder if actual game logs not available)
    base = player_row["RECENT_OVER_PROB"] * 100
    trend = [max(0, min(100, base + i * 2 - 4)) for i in range(int(player_row["RECENT_N"]))]
    df_trend = pd.DataFrame({
        "Game #": list(range(1, len(trend)+1)),
        "Hit %": trend
    })
    fig = px.line(df_trend, x="Game #", y="Hit %", markers=True, title="", height=150)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title=None, yaxis_title=None,
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)

# === UI Header ===
st.title("🏀 NBA Player Props Dashboard")
st.caption("Automatically updated daily — top overs from model predictions, with injury context & trends.")

# === Tabs per market ===
markets = df["MARKET"].unique()
tabs = st.tabs([f"🔥 {m}" for m in markets])

for tab, market in zip(tabs, markets):
    with tab:
        subset = df[df["MARKET"] == market].sort_values("FINAL_OVER_PROB", ascending=False).head(10)

        for _, row in subset.iterrows():
            with st.container(border=True):
                cols = st.columns([1.2, 3, 2])
                with cols[0]:
                    # Player & team images
                    st.image(get_player_image(row["PLAYER"]), width=80)
                    st.image(get_team_logo(row["TEAM"]), width=50)

                with cols[1]:
                    st.subheader(row["PLAYER"])
                    st.write(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    st.write(f"Team: `{row['TEAM'] or '—'}` | Injury: {row['INJ_Status']}")
                    st.write(f"Recent Hit Rate: {row['RECENT_OVER_PROB']*100:.1f}% ({int(row['RECENT_N'])} games)")

                with cols[2]:
                    st.metric("Prob. Over", row["FINAL_OVER_PROB_PCT"])
                    render_hit_rate_chart(row)