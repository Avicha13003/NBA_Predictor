# generate_rolling_metrics.py
import pandas as pd
import numpy as np
import os, logging, datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

LOG_PATH = "player_game_log.csv"
ROLLING_ALL = "rolling_metrics.csv"
ROLLING_DAILY = "rolling_metrics_daily.csv"

def compute_pct(made, att):
    """Return shooting percentage."""
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = (made / att) * 100
    return np.where(np.isfinite(pct), pct, 0)

def add_efficiency(df):
    """Add EFG% and TS%."""
    df = df.copy()
    for col in ["FG3A", "FTM", "FTA"]:
        if col not in df.columns:
            df[col] = 0

    # Effective FG%
    df["EFG_PCT"] = np.where(df["FGA"] > 0,
                             ((df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]) * 100,
                             0)

    # True Shooting %
    df["TS_PCT"] = np.where((df["FGA"] + 0.44 * df["FTA"]) > 0,
                            (df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"]))) * 100,
                            0)
    return df

def compute_rolling(df):
    """Compute rolling 3- and 5-game averages per player."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df.dropna(subset=["GAME_DATE"], inplace=True)
    df.sort_values(["PLAYER", "GAME_DATE"], inplace=True)

    grouped = df.groupby("PLAYER")

    roll_cols = {
        "r3_PTS": grouped["PTS"].rolling(3, min_periods=1).mean().reset_index(0, drop=True),
        "r5_PTS": grouped["PTS"].rolling(5, min_periods=1).mean().reset_index(0, drop=True),
        "r3_FG3_PCT": grouped["FG3_PCT"].rolling(3, min_periods=1).mean().reset_index(0, drop=True),
        "r5_FG3_PCT": grouped["FG3_PCT"].rolling(5, min_periods=1).mean().reset_index(0, drop=True),
        "r3_EFG": grouped["EFG_PCT"].rolling(3, min_periods=1).mean().reset_index(0, drop=True),
        "r5_EFG": grouped["EFG_PCT"].rolling(5, min_periods=1).mean().reset_index(0, drop=True),
        "r3_TS": grouped["TS_PCT"].rolling(3, min_periods=1).mean().reset_index(0, drop=True),
        "r5_TS": grouped["TS_PCT"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    }

    for col, vals in roll_cols.items():
        df[col] = vals

    latest = df.sort_values("GAME_DATE").groupby("PLAYER").tail(1)
    keep_cols = ["PLAYER", "TEAM", "TEAM_FULL", "GAME_DATE"] + list(roll_cols.keys())
    return latest[keep_cols]

def main():
    if not os.path.exists(LOG_PATH):
        logging.error(f"{LOG_PATH} not found.")
        return

    df = pd.read_csv(LOG_PATH)
    for col in ["PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Compute base FG3%
    df["FG3_PCT"] = compute_pct(df["FG3M"], df["FG3A"])
    df = add_efficiency(df)

    # Compute rolling averages
    rolled = compute_rolling(df)

    # Write all-player rolling file
    rolled.to_csv(ROLLING_ALL, index=False)
    logging.info(f"✅ Wrote {ROLLING_ALL} ({len(rolled)} players)")

    # Write yesterday-only subset
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    rolled_daily = rolled[rolled["GAME_DATE"] == yesterday]
    rolled_daily.to_csv(ROLLING_DAILY, index=False)
    logging.info(f"📅 Wrote {ROLLING_DAILY} ({len(rolled_daily)} players for {yesterday})")

if __name__ == "__main__":
    main()