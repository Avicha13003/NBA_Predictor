# dashboard_app.py — NBA Player Props Dashboard (mobile + free)
import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.set_page_config(page_title="NBA Player Props Dashboard", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("nba_prop_predictions_today.csv")
        return df
    except Exception:
        st.error("⚠️ Could not load nba_prop_predictions_today.csv")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# --- Helper: player & team images ---
@st.cache_data
def get_player_image(name):
    name_fmt = name.lower().replace(" ", "-")
    url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{name_fmt}.png"
    return url

@st.cache_data
def get_team_logo(team_abbr):
    if not team_abbr:
        return ""
    url = f"https://cdn.ssref.net/req/202106291/images/teams/{team_abbr.lower()}_logo.svg"
    return url

# --- UI Header ---
st.title("🏀 NBA Player Props Dashboard")
st.caption("Automatically updated daily — showing top overs from player prop predictions.")

# --- Tabs per market ---
markets = df["MARKET"].unique()
tabs = st.tabs([f"🔥 {m}" for m in markets])

for tab, market in zip(tabs, markets):
    with tab:
        subset = df[df["MARKET"] == market].sort_values("FINAL_OVER_PROB", ascending=False).head(10)
        for _, row in subset.iterrows():
            with st.container(border=True):
                cols = st.columns([1, 4, 2])
                with cols[0]:
                    st.image(get_player_image(row["PLAYER"]), width=80)
                    st.image(get_team_logo(row["TEAM"]), width=50)
                with cols[1]:
                    st.subheader(row["PLAYER"])
                    st.write(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    st.write(f"Team: `{row['TEAM']}` | Injury: {row['INJ_Status']}")
                with cols[2]:
                    st.metric("Prob. Over", row["FINAL_OVER_PROB_PCT"])
                    st.write(f"Recent Hit Rate: {row['RECENT_OVER_PROB']*100:.1f}% ({int(row['RECENT_N'])}g)")