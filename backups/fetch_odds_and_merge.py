import pandas as pd
import datetime
import logging
import requests

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# --- Paths ---
PLAYER_STATS_CSV = "nba_today_stats.csv"  # Combined stats + schedule

# --- Odds API ---
ODDS_API_KEY = "fcdd880206db31b09e9a1dc115a13a54"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"  # Example endpoint

def fetch_prop_odds():
    """
    Fetch player prop odds from your API key
    """
    logging.info("Fetching prop odds from API...")
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",          # US odds
        "markets": "player_points", # Player points props
        "oddsFormat": "decimal",  # Or "american"
        "dateFormat": "iso"
    }
    
    try:
        response = requests.get(ODDS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Flatten the API response into a dataframe
        odds_list = []
        for game in data:
            for market in game.get("bookmakers", []):
                for outcome in market.get("markets", []):
                    for player in outcome.get("outcomes", []):
                        odds_list.append({
                            "PLAYER": player.get("name"),
                            "TEAM": player.get("team"),
                            "PROP_TYPE": "PTS",
                            "LINE": player.get("point", None),
                            "ODDS": player.get("price", None)
                        })
        odds_df = pd.DataFrame(odds_list)
        logging.info(f"Fetched {len(odds_df)} player prop lines")
        return odds_df
    
    except Exception as e:
        logging.error(f"Failed to fetch odds: {e}")
        return pd.DataFrame()

# --- Feature engineering ---
def build_features(df):
    logging.info("Building features...")
    
    # Convert GAME_DATE to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Rest days
    df = df.sort_values(['PLAYER','GAME_DATE'])
    df['last_game_date'] = df.groupby('PLAYER')['GAME_DATE'].shift(1)
    df['rest_days'] = (df['GAME_DATE'] - df['last_game_date']).dt.days.fillna(0)
    
    # Recent 5-game averages
    for stat in ['PTS','REB','AST']:
        df[f'{stat}_recent_avg'] = df.groupby('PLAYER')[stat].rolling(5, min_periods=1).mean().reset_index(0,drop=True)
    
    # Placeholder matchup feature
    df['opponent_strength'] = 1.0
    return df

# --- Scoring ---
def score_over_picks(df):
    logging.info("Scoring over picks...")
    # Simple heuristic: compare recent avg to prop line
    df['score'] = df['PTS_recent_avg'] - df['LINE']
    return df

# --- Main pipeline ---
def main_pipeline():
    df = pd.read_csv(PLAYER_STATS_CSV)
    odds_df = fetch_prop_odds()
    
    # Merge odds into player stats
    merged_df = df.merge(odds_df, on=['PLAYER','TEAM'], how='left')
    
    # Build features
    merged_df = build_features(merged_df)
    
    # Score picks
    merged_df = score_over_picks(merged_df)
    
    # Top 10 over picks
    top10 = merged_df.sort_values('score', ascending=False).head(10)
    
    logging.info("✅ Top 10 Over Picks:")
    print(top10[['PLAYER','TEAM','PROP_TYPE','LINE','PTS_recent_avg','score']])
    
    top10.to_csv("top10_over_picks.csv", index=False)

if __name__ == "__main__":
    main_pipeline()