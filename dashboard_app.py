# dashboard_app.py — NBA Player Props Dashboard (filters + logos + headshots + safe image handling)
import streamlit as st
import pandas as pd

st.set_page_config(page_title="NBA Player Props Dashboard", layout="wide")

# --- Cached loaders ---
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
    import os
    try:
        # Try multiple potential paths (Streamlit Cloud may run from /app/src)
        for path in ["player_headshots.csv", "./src/player_headshots.csv", "./data/player_headshots.csv"]:
            if os.path.exists(path):
                heads = pd.read_csv(path)

                # Handle either naming convention
                if "PLAYER" in heads.columns and "PHOTO_URL" in heads.columns:
                    heads = heads.rename(columns={"PLAYER": "player", "PHOTO_URL": "image_url"})
                elif "player" not in heads.columns or "image_url" not in heads.columns:
                    st.warning(f"⚠️ Columns not found in {path}. Expected PLAYER/PHOTO_URL or player/image_url.")
                    continue

                heads["player_norm"] = heads["player"].astype(str).str.strip().str.lower()
                st.success(f"✅ Loaded player_headshots.csv ({len(heads)} rows) from {path}")
                return heads

        st.warning("⚠️ player_headshots.csv not found in any expected path.")
        return pd.DataFrame(columns=["player", "image_url", "player_norm"])
    except Exception as e:
        st.error(f"❌ Error loading player_headshots.csv: {e}")
        return pd.DataFrame(columns=["player", "image_url", "player_norm"])

# --- Load Data ---
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

if selected_team != "All Teams":
    filtered_players = sorted(df[df["TEAM"] == selected_team]["PLAYER"].unique().tolist())
else:
    filtered_players = sorted(df["PLAYER"].unique().tolist())

selected_player = st.sidebar.selectbox("Select Player", ["All Players"] + filtered_players)

# Apply filters
filtered_df = df.copy()
if selected_team != "All Teams":
    filtered_df = filtered_df[filtered_df["TEAM"] == selected_team]
if selected_player != "All Players":
    filtered_df = filtered_df[filtered_df["PLAYER"] == selected_player]

# --- Header ---
st.title("🏀 NBA Player Props Dashboard")
st.caption("Automatically updated daily — top overs with probabilities, trends, and injuries.")

# --- Market Tabs ---
markets = filtered_df["MARKET"].dropna().unique()
tabs = st.tabs([f"{m}" for m in markets])

for tab, market in zip(tabs, markets):
    with tab:
        subset = filtered_df[filtered_df["MARKET"] == market].sort_values("FINAL_OVER_PROB", ascending=False).head(10)
        if subset.empty:
            st.info("No data for this market today.")
            continue

        for _, row in subset.iterrows():
            with st.container(border=True):
                cols = st.columns([1, 3, 2])
                with cols[0]:
                    # Safe image handling
                    img_url = row.get("image_url", "")
                    logo_url = row.get("LOGO_URL", "")
                    if isinstance(img_url, str) and img_url.startswith("http"):
                        st.image(img_url, width=80)
                    else:
                        st.image("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png", width=80)
                    if isinstance(logo_url, str) and logo_url.startswith("http"):
                        st.image(logo_url, width=40)

                with cols[1]:
                    st.subheader(row["PLAYER"])
                    st.write(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    st.write(f"Team: `{row['TEAM']}` | Injury: {row['INJ_Status']}")

                with cols[2]:
                    st.metric("Prob. Over", row["FINAL_OVER_PROB_PCT"])
                    if not pd.isna(row.get("RECENT_OVER_PROB", None)):
                        pct = row["RECENT_OVER_PROB"] * 100
                        color = "green" if pct > 60 else "orange" if pct > 40 else "red"
                        st.markdown(
                            f"<span style='color:{color}'>Recent Hit Rate: {pct:.1f}% ({int(row['RECENT_N'])}g)</span>",
                            unsafe_allow_html=True,
                        )