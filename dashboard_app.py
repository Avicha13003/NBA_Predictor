# dashboard_app.py — NBA Player Props Dashboard (fixed IDs + photos/logos)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="NBA Player Props Dashboard", layout="wide")

# === Load data ===
@st.cache_data
def load_data():
    try:
        props = pd.read_csv("nba_prop_predictions_today.csv")
        logs = pd.read_csv("player_game_log.csv")
        props["RECENT_OVER_PROB"] = pd.to_numeric(props["RECENT_OVER_PROB"], errors="coerce").fillna(0)
        props["RECENT_N"] = pd.to_numeric(props["RECENT_N"], errors="coerce").fillna(0)
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
        return props, logs
    except Exception as e:
        st.error(f"⚠️ Could not load CSVs: {e}")
        return pd.DataFrame(), pd.DataFrame()

props, logs = load_data()
if props.empty or logs.empty:
    st.stop()

# === Helpers ===
@st.cache_data
def get_player_image(name):
    """Try NBA headshot, fallback placeholder."""
    name_fmt = name.lower().replace(" ", "-")
    possible_urls = [
        f"https://cdn.nba.com/headshots/nba/latest/260x190/{name_fmt}.png",
        f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{name_fmt}.png"
    ]
    for u in possible_urls:
        return u  # Streamlit caches this regardless of load
    return "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"

@st.cache_data
def get_team_logo(team_abbr):
    if not team_abbr or not isinstance(team_abbr, str) or len(team_abbr.strip()) == 0:
        return "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"
    return f"https://cdn.ssref.net/req/202106291/images/teams/{team_abbr.lower()}_logo.svg"

def get_hit_column(market):
    return {
        "PTS": "didHitOver_PTS",
        "REB": "didHitOver_REB",
        "AST": "didHitOver_AST",
        "3PM": "didHitOver_FG3M",
        "STL": "didHitOver_STL"
    }.get(market)

def render_hit_rate_chart(player, market, unique_key):
    """Render small trend line with unique Streamlit key per player-market."""
    hit_col = get_hit_column(market)
    if not hit_col or hit_col not in logs.columns:
        return
    dfp = logs[logs["PLAYER"].astype(str).str.strip() == player].copy()
    if dfp.empty:
        return
    dfp = dfp.sort_values("GAME_DATE").tail(10)
    dfp["Hit"] = pd.to_numeric(dfp[hit_col], errors="coerce").fillna(0).astype(int)
    dfp["RollingRate"] = dfp["Hit"].rolling(5, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfp["GAME_DATE"], y=dfp["Hit"],
        mode="lines+markers",
        line=dict(color="green"), name="Hit (1=Over)"
    ))
    fig.add_trace(go.Scatter(
        x=dfp["GAME_DATE"], y=dfp["RollingRate"],
        mode="lines",
        line=dict(color="orange", dash="dot"),
        name="Rolling Avg"
    ))
    fig.update_layout(
        height=150, margin=dict(l=0, r=0, t=0, b=0),
        yaxis=dict(range=[-0.1, 1.1], tickvals=[0,1]),
        xaxis_title=None, yaxis_title=None, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{player}-{market}-{unique_key}")

# === UI Header ===
st.title("🏀 NBA Player Props Dashboard")
st.caption("Daily top player prop overs — with real hit trends, team logos, and injury context.")

# === Tabs ===
markets = props["MARKET"].unique()
tabs = st.tabs([f"🔥 {m}" for m in markets])

for tab, market in zip(tabs, markets):
    with tab:
        subset = props[props["MARKET"] == market].sort_values("FINAL_OVER_PROB", ascending=False).head(10)

        for i, (_, row) in enumerate(subset.iterrows()):
            with st.container(border=True):
                cols = st.columns([1.2, 3, 2])
                with cols[0]:
                    st.image(get_player_image(row["PLAYER"]), width=90)
                    st.image(get_team_logo(row["TEAM"]), width=50)

                with cols[1]:
                    st.subheader(row["PLAYER"])
                    st.write(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    st.write(f"Team: `{row['TEAM'] or '—'}` | Injury: {row['INJ_Status']}")
                    st.write(f"Recent Hit Rate: {row['RECENT_OVER_PROB']*100:.1f}% ({int(row['RECENT_N'])} games)")

                with cols[2]:
                    st.metric("Prob. Over", row["FINAL_OVER_PROB_PCT"])
                    render_hit_rate_chart(row["PLAYER"], market, i)