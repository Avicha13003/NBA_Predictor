# dashboard_app.py — NBA Player Props Dashboard (with filters, logos, headshots)
import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="NBA Player Props Dashboard", layout="wide")

# --- Load data ---
@st.cache_data
def load_predictions():
    try:
        df = pd.read_csv("nba_prop_predictions_today.csv")
        df["PLAYER"] = df["PLAYER"].astype(str).str.strip()
        df["TEAM"] = df["TEAM"].astype(str).str.strip()
        return df
    except Exception:
        st.error("⚠️ Could not load nba_prop_predictions_today.csv")
        return pd.DataFrame()

@st.cache_data
def load_team_logos():
    try:
        logos = pd.read_csv("team_logos.csv")
        logos["TEAM"] = logos["TEAM"].astype(str).str.strip().str.upper()
        return logos
    except Exception:
        st.warning("⚠️ team_logos.csv not found — skipping logos.")
        return pd.DataFrame(columns=["TEAM", "LOGO_URL"])

@st.cache_data
def load_headshots():
    try:
        heads = pd.read_csv("player_headshots.csv")
        heads["player_norm"] = heads["player"].astype(str).str.strip().str.lower()
        return heads
    except Exception:
        st.warning("⚠️ player_headshots.csv not found — skipping headshots.")
        return pd.DataFrame(columns=["player", "image_url", "player_norm"])

# --- Load files ---
df = load_predictions()
logos = load_team_logos()
headshots = load_headshots()

if df.empty:
    st.stop()

# --- Merge assets ---
df["player_norm"] = df["PLAYER"].str.lower().str.strip()
df["TEAM"] = df["TEAM"].fillna("").str.upper()

if not headshots.empty:
    df = df.merge(headshots[["player_norm", "image_url"]], on="player_norm", how="left")
else:
    df["image_url"] = ""

if not logos.empty:
    df = df.merge(logos, on="TEAM", how="left")
else:
    df["LOGO_URL"] = ""

# Fill fallback images
df["image_url"] = df["image_url"].fillna("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png")
df["LOGO_URL"] = df["LOGO_URL"].fillna("")

# --- Sidebar Filters ---
st.sidebar.title("🔍 Filters")

teams = sorted(df["TEAM"].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Select Team", ["All Teams"] + teams)

players = sorted(
    df["PLAYER"][df["TEAM"] == selected_team].unique().tolist()
) if selected_team != "All Teams" else sorted(df["PLAYER"].unique().tolist())

selected_player = st.sidebar.selectbox("Select Player", ["All Players"] + players)

if selected_team != "All Teams":
    df = df[df["TEAM"] == selected_team]
if selected_player != "All Players":
    df = df[df["PLAYER"] == selected_player]

# --- Header ---
st.title("🏀 NBA Player Props Dashboard")
st.caption("Automatically updated daily — top overs with probabilities, trends, and injuries.")

# --- Market Tabs ---
markets = df["MARKET"].dropna().unique()
tabs = st.tabs([f"{m}" for m in markets])

for tab, market in zip(tabs, markets):
    with tab:
        subset = df[df["MARKET"] == market].sort_values("FINAL_OVER_PROB", ascending=False).head(10)
        if subset.empty:
            st.info("No data for this market today.")
            continue

        for _, row in subset.iterrows():
            with st.container(border=True):
                cols = st.columns([1, 3, 2])
                with cols[0]:
                    st.image(row["image_url"], width=80)
                    if row["LOGO_URL"]:
                        st.image(row["LOGO_URL"], width=40)
                with cols[1]:
                    st.subheader(row["PLAYER"])
                    st.write(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    st.write(f"Team: `{row['TEAM']}` | Injury: {row['INJ_Status']}")
                with cols[2]:
                    st.metric("Prob. Over", row["FINAL_OVER_PROB_PCT"])
                    if not pd.isna(row.get("RECENT_OVER_PROB", None)):
                        pct = row["RECENT_OVER_PROB"] * 100
                        color = "green" if pct > 60 else "orange" if pct > 40 else "red"
                        st.markdown(f"<span style='color:{color}'>Recent Hit Rate: {pct:.1f}% ({int(row['RECENT_N'])}g)</span>", unsafe_allow_html=True)