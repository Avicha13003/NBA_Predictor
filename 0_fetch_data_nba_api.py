import os
import pandas as pd
import pytz
import datetime
import logging
from nba_api.stats.endpoints import ScoreboardV2, LeagueDashPlayerStats
from nba_api.stats.static import teams

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# --- Arena location mapping (known NBA venues) ---
ARENA_LOCATIONS = {
    "State Farm Arena": {"city": "Atlanta", "state": "GA"},
    "TD Garden": {"city": "Boston", "state": "MA"},
    "Barclays Center": {"city": "Brooklyn", "state": "NY"},
    "Spectrum Center": {"city": "Charlotte", "state": "NC"},
    "United Center": {"city": "Chicago", "state": "IL"},
    "Rocket Mortgage FieldHouse": {"city": "Cleveland", "state": "OH"},
    "American Airlines Center": {"city": "Dallas", "state": "TX"},
    "Ball Arena": {"city": "Denver", "state": "CO"},
    "Little Caesars Arena": {"city": "Detroit", "state": "MI"},
    "Chase Center": {"city": "San Francisco", "state": "CA"},
    "Toyota Center": {"city": "Houston", "state": "TX"},
    "Gainbridge Fieldhouse": {"city": "Indianapolis", "state": "IN"},
    "Crypto.com Arena": {"city": "Los Angeles", "state": "CA"},
    "Kaseya Center": {"city": "Miami", "state": "FL"},
    "Fiserv Forum": {"city": "Milwaukee", "state": "WI"},
    "Target Center": {"city": "Minneapolis", "state": "MN"},
    "FedExForum": {"city": "Memphis", "state": "TN"},
    "Smoothie King Center": {"city": "New Orleans", "state": "LA"},
    "Madison Square Garden": {"city": "New York", "state": "NY"},
    "Paycom Center": {"city": "Oklahoma City", "state": "OK"},
    "Amway Center": {"city": "Orlando", "state": "FL"},
    "Wells Fargo Center": {"city": "Philadelphia", "state": "PA"},
    "Footprint Center": {"city": "Phoenix", "state": "AZ"},
    "Moda Center": {"city": "Portland", "state": "OR"},
    "Golden 1 Center": {"city": "Sacramento", "state": "CA"},
    "Frost Bank Center": {"city": "San Antonio", "state": "TX"},
    "Scotiabank Arena": {"city": "Toronto", "state": "ON"},
    "Delta Center": {"city": "Salt Lake City", "state": "UT"},
    "Capital One Arena": {"city": "Washington", "state": "DC"},
}

# --- Special VENUE fixes (international / alt venues) ---
VENUE_FIXES = {
    "Arena CDMX": {"city": "Mexico City", "state": "MX"},
    "Accor Arena": {"city": "Paris", "state": "FR"},
    "Etihad Arena": {"city": "Abu Dhabi", "state": "UAE"},
}

# --- Team city fallback map ---
TEAM_CITY_MAP = {
    "ATL": {"city": "Atlanta", "state": "GA"},
    "BOS": {"city": "Boston", "state": "MA"},
    "BKN": {"city": "Brooklyn", "state": "NY"},
    "CHA": {"city": "Charlotte", "state": "NC"},
    "CHI": {"city": "Chicago", "state": "IL"},
    "CLE": {"city": "Cleveland", "state": "OH"},
    "DAL": {"city": "Dallas", "state": "TX"},
    "DEN": {"city": "Denver", "state": "CO"},
    "DET": {"city": "Detroit", "state": "MI"},
    "GSW": {"city": "San Francisco", "state": "CA"},
    "HOU": {"city": "Houston", "state": "TX"},
    "IND": {"city": "Indianapolis", "state": "IN"},
    "LAC": {"city": "Los Angeles", "state": "CA"},
    "LAL": {"city": "Los Angeles", "state": "CA"},
    "MEM": {"city": "Memphis", "state": "TN"},
    "MIA": {"city": "Miami", "state": "FL"},
    "MIL": {"city": "Milwaukee", "state": "WI"},
    "MIN": {"city": "Minneapolis", "state": "MN"},
    "NOP": {"city": "New Orleans", "state": "LA"},
    "NYK": {"city": "New York", "state": "NY"},
    "OKC": {"city": "Oklahoma City", "state": "OK"},
    "ORL": {"city": "Orlando", "state": "FL"},
    "PHI": {"city": "Philadelphia", "state": "PA"},
    "PHX": {"city": "Phoenix", "state": "AZ"},
    "POR": {"city": "Portland", "state": "OR"},
    "SAC": {"city": "Sacramento", "state": "CA"},
    "SAS": {"city": "San Antonio", "state": "TX"},
    "TOR": {"city": "Toronto", "state": "ON"},
    "UTA": {"city": "Salt Lake City", "state": "UT"},
    "WAS": {"city": "Washington", "state": "DC"},
}

# --- Helper functions ---
def get_team_name(team_id: int) -> str:
    for t in teams.get_teams():
        if t["id"] == team_id:
            return t["full_name"]
    return f"Unknown ({team_id})"

def get_team_abbr(team_name: str) -> str:
    for t in teams.get_teams():
        if t["full_name"] == team_name:
            return t["abbreviation"]
    return "UNK"

def get_today_schedule():
    today = datetime.date.today()
    logging.info(f"Fetching NBA schedule for {today}...")
    try:
        scoreboard = ScoreboardV2(game_date=today.strftime("%Y-%m-%d"))
        games = scoreboard.game_header.get_data_frame()
    except Exception as e:
        logging.error(f"Failed to fetch schedule: {e}")
        return pd.DataFrame()

    if games.empty:
        logging.warning("No NBA games found for today.")
        return games

    games = games[["GAME_ID", "GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "ARENA_NAME"]]
    games = games.rename(columns={"GAME_DATE_EST": "GAME_DATE", "ARENA_NAME": "ARENA"})

    games["HOME_TEAM"] = games["HOME_TEAM_ID"].apply(get_team_name)
    games["AWAY_TEAM"] = games["VISITOR_TEAM_ID"].apply(get_team_name)

    # --- City/state logic ---
    city_list, state_list = [], []
    for _, g in games.iterrows():
        arena = g["ARENA"]
        home_team = g["HOME_TEAM"]

        city = ARENA_LOCATIONS.get(arena, {}).get("city", "Unknown")
        state = ARENA_LOCATIONS.get(arena, {}).get("state", "Unknown")

        # Special international / alternate venue override
        if arena in VENUE_FIXES:
            fix = VENUE_FIXES[arena]
            city, state = fix["city"], fix["state"]
            logging.info(f"🌍 Venue override applied for {arena}: {city}, {state}")

        # Fallback to home team city if still unknown
        if city == "Unknown" or not city:
            abbr = get_team_abbr(home_team)
            if abbr in TEAM_CITY_MAP:
                fallback = TEAM_CITY_MAP[abbr]
                city, state = fallback["city"], fallback["state"]
                logging.info(f"🏙️ Fallback to team city for {arena}: {city}, {state}")

        city_list.append(city)
        state_list.append(state)

    games["CITY"], games["STATE"] = city_list, state_list
    return games

def fetch_player_stats():
    """Fetch current season player averages."""
    try:
        stats = LeagueDashPlayerStats(per_mode_detailed="PerGame").get_data_frames()[0]
        return stats
    except Exception as e:
        logging.error(f"Failed to fetch player stats: {e}")
        return pd.DataFrame()

def build_daily_stats():
    games = get_today_schedule()
    if games.empty:
        logging.error("No games schedule fetched — aborting.")
        return pd.DataFrame()

    logging.info(f"✅ Found {len(games)} games for today")

    player_stats = fetch_player_stats()
    if player_stats.empty:
        logging.error("No player stats fetched.")
        return pd.DataFrame()

    all_rows = []
    for _, g in games.iterrows():
        game_dt_utc = pd.to_datetime(g["GAME_DATE"])
        game_date_str = game_dt_utc.strftime("%Y-%m-%d")

        for side, team_col in [("Home", "HOME_TEAM"), ("Away", "AWAY_TEAM")]:
            team_name = g[team_col]
            team_abbr = get_team_abbr(team_name)
            arena, city, state = g["ARENA"], g["CITY"], g["STATE"]

            players = player_stats[player_stats["TEAM_ABBREVIATION"] == team_abbr]
            if players.empty:
                logging.warning(f"No players found for {team_name} ({team_abbr})")
                continue

            logging.info(f"Processing: {g['AWAY_TEAM']} @ {g['HOME_TEAM']} ({arena}, {city}, {state})")

            for _, p in players.iterrows():
                all_rows.append({
                    "PLAYER": p["PLAYER_NAME"],
                    "TEAM": p["TEAM_ABBREVIATION"],
                    "TEAM_FULL": team_name,
                    "TEAM_SIDE": side,
                    "PTS": p["PTS"],
                    "REB": p["REB"],
                    "AST": p["AST"],
                    "FG3M": p["FG3M"],
                    "MIN": p["MIN"],
                    "ARENA": arena,
                    "CITY": city,
                    "STATE": state,
                    "GAME_DATE": game_date_str,
                })

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv("nba_today_stats.csv", index=False)
        logging.info(f"✅ Saved today's player + game data to nba_today_stats.csv")
    else:
        logging.warning("No player data compiled.")

    return df

if __name__ == "__main__":
    df = build_daily_stats()
    if not df.empty:
        print(df.head())