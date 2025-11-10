# src/fetch_data.py
import os
import logging
import time
from datetime import date

import pandas as pd
import requests
from nba_api.stats.endpoints import playergamelog, commonplayerinfo

from dotenv import load_dotenv
load_dotenv()

# Logging
logging.basicConfig(
    filename="../data/raw/fetch_data.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# Environment/config
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BASE_BDL = "https://www.balldontlie.io/api/v1"

# -----------------------------
# Helper: Safe request
# -----------------------------
def safe_get_json(url, params=None):
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            logging.warning(f"Request to {url} returned {resp.status_code}")
            return None
        return resp.json()
    except Exception as e:
        logging.error(f"Request to {url} failed: {e}")
        return None

# -----------------------------
# Fetch games schedule
# -----------------------------
def fetch_schedule_balldontlie(target_date):
    url = f"{BASE_BDL}/games"
    params = {"dates[]": target_date, "per_page": 100}
    js = safe_get_json(url, params=params)  # no headers
    if not js or "data" not in js or not js["data"]:
        logging.warning("BallDontLie schedule fetch returned no data.")
        return None
    df = pd.json_normalize(js["data"])
    logging.info(f"Fetched {len(df)} games via BallDontLie for {target_date}")
    return df

# -----------------------------
# Fetch all active players for a team via BDL
# -----------------------------
def fetch_team_players_bdl(team_abbrev):
    url = f"{BASE_BDL}/players"
    all_players = []
    page = 1
    while True:
        params = {"per_page": 100, "page": page}
        data = safe_get_json(url, params=params)  # no headers
        if not data or "data" not in data:
            break
        players = data["data"]
        for p in players:
            if p["team"]["abbreviation"] == team_abbrev:
                all_players.append(p)
        if page >= data["meta"]["total_pages"]:
            break
        page += 1
    return all_players

# -----------------------------
# Fetch recent NBA stats for a player
# -----------------------------
def fetch_recent_player_stats(player_id, num_games=10):
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25', season_type_all_star='Regular Season')
        df = gl.get_data_frames()[0].head(num_games)
        df2 = df[["GAME_ID", "PLAYER_ID", "PTS", "REB", "AST", "MIN"]].copy()
        df2["PLAYER_ID"] = player_id
        return df2
    except Exception as e:
        logging.warning(f"nba_api fetch_recent_player_stats failed for player {player_id}: {e}")
        return None

# -----------------------------
# Fetch season averages
# -----------------------------
def fetch_season_average(player_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df = info.get_data_frames()[1]
        avg = df.iloc[-1].to_dict() if not df.empty else {}
        return avg
    except Exception as e:
        logging.warning(f"nba_api fetch_season_average failed for player {player_id}: {e}")
        return {}

# -----------------------------
# Fetch Odds API props
# -----------------------------
def fetch_odds_today(target_date):
    url = ("https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
           f"?regions=us&markets=player_points,player_assists,player_rebounds&date={target_date}&apiKey={ODDS_API_KEY}")
    js = safe_get_json(url)
    if not js:
        logging.warning("Odds API returned no data.")
        return None
    df = pd.json_normalize(js)
    logging.info(f"Fetched odds data with {len(df)} events")
    return df

# -----------------------------
# Main
# -----------------------------
def main():
    target_date = str(date.today())
    logging.info(f"=== Starting fetch_data for {target_date} ===")

    games_df = fetch_schedule_balldontlie(target_date)
    if games_df is None or games_df.empty:
        logging.error("No games schedule fetched — aborting.")
        return

    all_players = []

    # Loop over games
    for _, g in games_df.iterrows():
        for team_type in ["home_team", "visitor_team"]:
            try:
                team_abbrev = g[f"{team_type}.abbreviation"]
                players = fetch_team_players_bdl(team_abbrev)
                if not players:
                    logging.warning(f"No players found for team {team_abbrev}")
                    continue

                for p in players:
                    player_name = f"{p['first_name']} {p['last_name']}"
                    pid = p["id"]
                    recent = fetch_recent_player_stats(pid)
                    if recent is not None:
                        season_avg = fetch_season_average(pid)
                        recent["season_pts_avg"] = season_avg.get("PTS", None)
                        recent["season_reb_avg"] = season_avg.get("REB", None)
                        recent["season_ast_avg"] = season_avg.get("AST", None)
                        recent["player_name"] = player_name
                        recent["team_abbrev"] = team_abbrev
                        all_players.append(recent)
                    time.sleep(0.2)

            except Exception as e:
                logging.warning(f"Error fetching players for {team_type}: {e}")

    if not all_players:
        logging.error("No player data fetched.")
        return

    players_df = pd.concat(all_players, ignore_index=True)
    logging.info(f"Fetched player stats for {players_df['PLAYER_ID'].nunique()} players")

    # Odds
    odds_df = fetch_odds_today(target_date)
    if odds_df is None:
        logging.warning("Proceeding without odds data.")

    # Merge
    out_df = players_df.copy()
    if odds_df is not None and "player_name" in odds_df.columns:
        out_df = out_df.merge(odds_df, how="left", on="player_name")

    # Save
    os.makedirs("../data/raw", exist_ok=True)
    out_path = f"../data/raw/today_{target_date}.csv"
    out_df.to_csv(out_path, index=False)
    logging.info(f"Saved merged data to {out_path}")
    logging.info(f"=== Finished fetch_data for {target_date} ===")

if __name__ == "__main__":
    main()