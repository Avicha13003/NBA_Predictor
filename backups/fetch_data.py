# src/fetch_data.py
import os
import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.live.nba.endpoints import scoreboard  # live games
from nba_api.stats.static import players as nba_players

# Configure logging
logging.basicConfig(
    filename="../data/raw/fetch_data.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# Environment/config
BDL_API_KEY = os.getenv("BDL_API_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BASE_BDL = "https://api.balldontlie.io/v1"

def safe_get_json(url, headers=None, params=None):
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            logging.warning(f"Request to {url} returned {resp.status_code}")
            return None
        return resp.json()
    except Exception as e:
        logging.error(f"Request to {url} failed: {e}")
        return None

def fetch_schedule_nbaapi(target_date):
    try:
        sb = scoreboard.ScoreBoard()  # No arguments
        games_df = sb.games.get_dict()  # Updated call structure
        games = pd.DataFrame(games_df['games'])
        logging.info(f"Fetched {len(games)} games via nba_api for {target_date}")
        return games
    except Exception as e:
        logging.warning(f"nba_api schedule fetch failed: {e}")
        return None

def fetch_schedule_balldontlie(target_date):
    url = f"{BASE_BDL}/games"
    params = {"dates[]": target_date, "per_page": 100}
    if BDL_API_KEY:
        headers = {"Authorization": BDL_API_KEY}
    else:
        headers = None
    js = safe_get_json(url, headers=headers, params=params)
    if not js or "data" not in js:
        logging.warning("BallDontLie schedule fetch returned no data.")
        return None
    df = pd.DataFrame(js["data"])
    logging.info(f"Fetched {len(df)} games via BallDontLie for {target_date}")
    return df

def fetch_recent_player_stats(player_id, num_games=10):
    """Fetch last N games for a player."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25', season_type_all_star='Regular Season')
        df = gl.get_data_frames()[0].head(num_games)
        # Keep only relevant columns
        df2 = df[["GAME_ID", "PLAYER_ID", "PTS", "REB", "AST", "MIN"]].copy()
        df2["PLAYER_ID"] = player_id
        return df2
    except Exception as e:
        logging.warning(f"nba_api fetch_recent_player_stats failed for player {player_id}: {e}")
        return None

def fetch_season_average(player_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df = info.get_data_frames()[1] # available seasons
        # Simplify: pick last season
        avg = df.iloc[-1].to_dict()
        return avg
    except Exception as e:
        logging.warning(f"nba_api fetch_season_average failed for player {player_id}: {e}")
        return {}

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

def main():
    target_date = str(date.today())
    logging.info(f"=== Starting fetch_data for {target_date} ===")

    # 1. Schedule
    games = fetch_schedule_nbaapi(target_date)
    if games is None or games.empty:
        games = fetch_schedule_balldontlie(target_date)
    if games is None or games.empty:
        logging.error("No games schedule fetched — aborting.")
        return

    # 2. Player stats pull
    all_players = []
    for idx, g in games.iterrows():
        # determine team/player list for home and away
        for team_key in ["HOME_TEAM_ID", "VISITOR_TEAM_ID"]:
            try:
                team_id = g[team_key]
                # fetch roster via nba_players static list
                roster = [p for p in nba_players.get_active_players() if p["team_id"] == team_id]
                for p in roster:
                    pid = p["id"]
                    recent = fetch_recent_player_stats(pid, num_games=10)
                    if recent is not None:
                        # add season avg
                        season_avg = fetch_season_average(pid)
                        recent["season_pts_avg"] = season_avg.get("PTS", None)
                        recent["season_reb_avg"] = season_avg.get("REB", None)
                        recent["season_ast_avg"] = season_avg.get("AST", None)
                        recent["player_name"] = p["full_name"]
                        recent["team_id"] = team_id
                        all_players.append(recent)
                    time.sleep(0.3)  # throttle
            except Exception as e:
                logging.warning(f"Error fetching players for team {team_id}: {e}")

    if not all_players:
        logging.error("No player data fetched.")
        return

    players_df = pd.concat(all_players, ignore_index=True)
    logging.info(f"Fetched player stats for {players_df['PLAYER_ID'].nunique()} players")

    # 3. Odds
    odds_df = fetch_odds_today(target_date)
    if odds_df is None:
        logging.warning("Proceeding without odds data.")

    # 4. Merge and save
    out_df = players_df.copy()
    # Note: you’ll likely need to join odds_df by player_name or event
    if odds_df is not None:
        out_df = out_df.merge(odds_df, how="left", on="player_name")

    # save
    os.makedirs("../data/raw", exist_ok=True)
    out_path = f"../data/raw/today_{target_date}.csv"
    out_df.to_csv(out_path, index=False)
    logging.info(f"Saved merged data to {out_path}")
    logging.info(f"=== Finished fetch_data for {target_date} ===")

if __name__ == "__main__":
    main()