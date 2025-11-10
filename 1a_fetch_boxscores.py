import requests, datetime, logging, os
import pandas as pd

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DAYS_BACK = 1
LOG_PATH = "player_game_log.csv"
DAILY_DIR = "daily_boxscores"
PROPS_PATH = "props_today.csv"   # used for DidHitOver tracking
os.makedirs(DAILY_DIR, exist_ok=True)

def fetch_espn_scoreboard(date_str):
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    logging.info(f"Fetching ESPN scoreboard for {date_str}...")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("events", [])
    except Exception as e:
        logging.error(f"Failed to fetch ESPN scoreboard: {e}")
        return []

def fetch_espn_boxscore(event_id):
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"Error fetching boxscore for event {event_id}: {e}")
        return None

def parse_made_attempt(stat_str):
    made, att, pct = 0, 0, 0.0
    if stat_str and "-" in stat_str:
        try:
            made, att = map(int, stat_str.split("-"))
            pct = round((made / att) * 100, 1) if att > 0 else 0.0
        except ValueError:
            pass
    return made, att, pct

def parse_boxscore(json_data, date_str):
    if not json_data or "boxscore" not in json_data:
        return pd.DataFrame()

    players = []
    for team in json_data["boxscore"].get("players", []):
        team_info = team.get("team", {})
        team_name = team_info.get("displayName", "Unknown")
        team_abbr = team_info.get("abbreviation", "UNK")

        for stat_group in team.get("statistics", []):
            headers = stat_group.get("names", [])
            for athlete in stat_group.get("athletes", []):
                values = athlete.get("stats", [])
                if len(headers) != len(values):
                    continue
                stat_map = dict(zip(headers, values))

                fgm, fga, fg_pct = parse_made_attempt(stat_map.get("FG"))
                fg3m, fg3a, fg3_pct = parse_made_attempt(stat_map.get("3PT"))
                ftm, fta, ft_pct = parse_made_attempt(stat_map.get("FT"))

                efg = round(((fgm + 0.5 * fg3m) / fga) * 100, 1) if fga > 0 else 0.0
                ts = round((float(stat_map.get("PTS", 0)) / (2 * (fga + 0.44 * fta))) * 100, 1) if (fga + 0.44 * fta) > 0 else 0.0

                players.append({
                    "GAME_DATE": date_str,
                    "TEAM": team_abbr,
                    "TEAM_FULL": team_name,
                    "PLAYER": athlete["athlete"]["displayName"],
                    "PTS": float(stat_map.get("PTS", 0)),
                    "REB": float(stat_map.get("REB", 0)),
                    "AST": float(stat_map.get("AST", 0)),
                    "STL": float(stat_map.get("STL", 0)),
                    "FGM": fgm, "FGA": fga, "FG_PCT": fg_pct,
                    "FG3M": fg3m, "FG3A": fg3a, "FG3_PCT": fg3_pct,
                    "FTM": ftm, "FTA": fta, "FT_PCT": ft_pct,
                    "EFG_PCT": efg, "TS_PCT": ts,
                    "MIN": stat_map.get("MIN", "0")
                })

    return pd.DataFrame(players)

def add_did_hit_over_columns(df):
    """Compares actual boxscores vs. props_today.csv to flag DidHitOver columns."""
    if not os.path.exists(PROPS_PATH):
        logging.warning("⚠️ props_today.csv not found — skipping DidHitOver tagging.")
        for stat in ["PTS", "REB", "AST", "FG3M", "STL"]:
            df[f"didHitOver_{stat}"] = 0
        return df

    props = pd.read_csv(PROPS_PATH)
    if props.empty or "PLAYER" not in props.columns or "MARKET" not in props.columns:
        logging.warning("⚠️ Invalid props_today.csv format.")
        return df

    props["PLAYER"] = props["PLAYER"].astype(str).str.strip()

    for stat in ["PTS", "REB", "AST", "FG3M", "STL"]:
        stat_props = props[props["MARKET"] == stat]
        df[f"didHitOver_{stat}"] = 0

        for _, row in stat_props.iterrows():
            player, line = str(row["PLAYER"]).strip(), float(row["LINE"])
            mask = df["PLAYER"].astype(str).str.strip() == player
            if mask.any():
                df.loc[mask, f"didHitOver_{stat}"] = (df.loc[mask, stat] > line).astype(int)

    logging.info("✅ Added DidHitOver columns based on props_today.csv.")
    return df

def append_to_log(df):
    if df.empty:
        logging.warning("No new boxscore data to append.")
        return
    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH)
        combined = pd.concat([old, df], ignore_index=True)
        combined.drop_duplicates(subset=["GAME_DATE","PLAYER","TEAM"], keep="last", inplace=True)
    else:
        combined = df
    combined.to_csv(LOG_PATH, index=False)
    logging.info(f"✅ Updated {LOG_PATH} ({len(df)} new, {len(combined)} total rows).")

def add_rolling_features(df):
    if df.empty:
        return df

    df = df.sort_values(by=["PLAYER", "GAME_DATE"])
    grouped = df.groupby("PLAYER")

    df["r3_PTS"] = grouped["PTS"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df["r5_PTS"] = grouped["PTS"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df["r3_FG3_PCT"] = grouped["FG3_PCT"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df["r5_FG3_PCT"] = grouped["FG3_PCT"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    return df

def build_daily_boxscores():
    target_date = (datetime.date.today() - datetime.timedelta(days=DAYS_BACK))
    date_str = target_date.strftime("%Y-%m-%d")
    date_str_api = target_date.strftime("%Y%m%d")

    events = fetch_espn_scoreboard(date_str_api)
    if not events:
        logging.warning("No games found for that date.")
        return

    completed = [e for e in events if e.get("status", {}).get("type", {}).get("completed")]
    logging.info(f"Found {len(completed)} completed games for {date_str}.")

    all_dfs = []
    for e in completed:
        event_id = e["id"]
        matchup = e.get("shortName", "Unknown Matchup")
        logging.info(f"Fetching boxscore for {matchup} (event_id={event_id})...")
        df = parse_boxscore(fetch_espn_boxscore(event_id), date_str)
        if not df.empty:
            logging.info(f"Parsed {len(df)} player stats for {matchup}.")
            all_dfs.append(df)
        else:
            logging.warning(f"No stats for event {event_id}")

    if not all_dfs:
        logging.warning("No boxscore data available.")
        return

    all_data = pd.concat(all_dfs, ignore_index=True)
    all_data["didHit30"] = (all_data["PTS"] >= 30).astype(int)
    all_data["didHit4Threes"] = (all_data["FG3M"] >= 4).astype(int)

    all_data = add_did_hit_over_columns(all_data)
    all_data = add_rolling_features(all_data)

    daily_path = os.path.join(DAILY_DIR, f"espn_boxscores_{target_date.strftime('%Y%m%d')}.csv")
    all_data.to_csv(daily_path, index=False)
    logging.info(f"📅 Saved daily boxscores → {daily_path}")

    append_to_log(all_data)

if __name__ == "__main__":
    build_daily_boxscores()