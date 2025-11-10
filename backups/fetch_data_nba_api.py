from nba_api.stats.endpoints import ScoreboardV2, PlayerGameLog, CommonTeamRoster
from nba_api.stats.static import teams
from datetime import date
import pandas as pd
import time

def get_today_schedule():
    """Fetch today's NBA games."""
    today = date.today().strftime("%Y-%m-%d")
    print(f"Fetching NBA schedule for {today}...")
    scoreboard = ScoreboardV2(game_date=today)
    games = scoreboard.game_header.get_data_frame()

    if games.empty:
        print("No NBA games found for today.")
        return None

    games = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_STATUS_TEXT']]
    print(f"Found {len(games)} games today.")
    return games


def get_team_name(team_id):
    """Convert team ID to team name."""
    team_list = teams.get_teams()
    team = next((t for t in team_list if t['id'] == team_id), None)
    return team['full_name'] if team else "Unknown"


def get_team_roster(team_id):
    """Fetch a team's current roster."""
    time.sleep(1)  # avoid rate limits
    roster = CommonTeamRoster(team_id=team_id).get_data_frames()[0]
    return roster[['PLAYER', 'PLAYER_ID', 'POSITION']]


def get_player_recent_stats(player_id, last_n=5):
    """Fetch recent game stats for a player."""
    time.sleep(1)  # rate limit to be safe
    logs = PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]
    recent = logs.head(last_n)
    avg_stats = {
        'PTS': recent['PTS'].mean(),
        'REB': recent['REB'].mean(),
        'AST': recent['AST'].mean(),
        'FG3M': recent['FG3M'].mean(),
        'MIN': recent['MIN'].mean(),
    }
    return avg_stats


def build_daily_stats():
    games = get_today_schedule()
    if games is None:
        return

    all_rows = []

    for _, g in games.iterrows():
        home_team = get_team_name(g['HOME_TEAM_ID'])
        away_team = get_team_name(g['VISITOR_TEAM_ID'])
        print(f"\nProcessing: {away_team} @ {home_team}")

        for team_id, team_label in [(g['HOME_TEAM_ID'], "Home"), (g['VISITOR_TEAM_ID'], "Away")]:
            roster = get_team_roster(team_id)
            for _, player in roster.iterrows():
                try:
                    stats = get_player_recent_stats(player['PLAYER_ID'])
                    stats.update({
                        'PLAYER': player['PLAYER'],
                        'POSITION': player['POSITION'],
                        'TEAM': get_team_name(team_id),
                        'GAME_ID': g['GAME_ID'],
                        'TEAM_SIDE': team_label
                    })
                    all_rows.append(stats)
                except Exception as e:
                    print(f"  ⚠️ Could not fetch stats for {player['PLAYER']}: {e}")
                    continue

    df = pd.DataFrame(all_rows)
    df.to_csv("nba_today_stats.csv", index=False)
    print("\n✅ Saved today's player stats to nba_today_stats.csv")
    return df


if __name__ == "__main__":
    df = build_daily_stats()
    if df is not None:
        print(df.head(10))