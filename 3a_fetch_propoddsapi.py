import os
import time
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

# --- Config ---
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba"
OUT_FILE = "props_today.csv"
LOCAL_TZ = ZoneInfo("America/Chicago")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def require_key():
    if not API_KEY:
        raise RuntimeError("Missing ODDS_API_KEY in your .env file")

def fetch_json(url):
    """Fetch JSON with simple retries + 429 backoff."""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                # The Odds API returns remaining-requests headers; log if present.
                remain = r.headers.get("x-requests-remaining")
                used   = r.headers.get("x-requests-used")
                if remain is not None and used is not None:
                    logging.info(f"OddsAPI quota → remaining={remain}, used={used}")
                return r.json()
            elif r.status_code == 429:
                wait = (attempt + 1) * 5
                logging.warning(f"Rate limit hit (429). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"HTTP {r.status_code}: {r.text[:200]}")
                break
        except Exception as e:
            logging.warning(f"Fetch error ({attempt+1}/3): {e}")
            time.sleep(3)
    return None

def local_day_utc_bounds(now_local=None):
    """Return (start_utc, end_utc) for the current local calendar day."""
    if now_local is None:
        now_local = datetime.now(LOCAL_TZ)
    start_local = datetime.combine(now_local.date(), dtime(0, 0, 0), tzinfo=LOCAL_TZ)
    end_local   = datetime.combine(now_local.date(), dtime(23, 59, 59), tzinfo=LOCAL_TZ)
    return start_local.astimezone(ZoneInfo("UTC")), end_local.astimezone(ZoneInfo("UTC"))

def fetch_events_for_local_day():
    """Fetch all NBA events and keep those whose UTC commence_time falls within the local (CT) day."""
    url = f"{BASE_URL}/events?apiKey={API_KEY}"
    data = fetch_json(url)
    if not data:
        logging.error("No events data fetched from The Odds API.")
        return []

    start_utc, end_utc = local_day_utc_bounds()
    logging.info(f"Filtering events for local day window (CT): {start_utc} → {end_utc} UTC")

    total = 0
    kept = []
    for e in data:
        total += 1
        ts = e.get("commence_time")
        if not ts:
            continue
        try:
            dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo("UTC"))
        except Exception:
            continue
        if start_utc <= dt_utc <= end_utc:
            kept.append(e)

    logging.info(f"✅ Found {len(kept)} NBA events in local day (from {total} total).")
    for ev in kept:
        logging.info(f" - {ev.get('away_team')} @ {ev.get('home_team')} | {ev.get('commence_time')}")
    return kept

def fetch_props_for_event(event_id):
    """Fetch player props for a single event (DraftKings only)."""
    markets = ",".join([
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_steals"
    ])
    url = f"{BASE_URL}/events/{event_id}/odds?apiKey={API_KEY}&regions=us&markets={markets}"
    data = fetch_json(url)
    if not data:
        return []

    rows = []
    bookmakers = data.get("bookmakers", [])
    for bk in bookmakers:
        if bk.get("key") != "draftkings":
            continue
        for market in bk.get("markets", []):
            mkey = (market.get("key") or "").replace("player_", "").upper()
            for outcome in market.get("outcomes", []):
                player = (outcome.get("description") or "").strip()
                line = outcome.get("point", None)
                if player and line is not None:
                    rows.append({
                        "PLAYER": player,
                        "MARKET": mkey,
                        "LINE": float(line),
                    })
    return rows

def normalize_props(rows, event):
    """Normalize and map market abbreviations; attach teams."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    mapping = {
        "POINTS": "PTS",
        "REBOUNDS": "REB",
        "ASSISTS": "AST",
        "THREES": "3PM",
        "STEALS": "STL"
    }
    df["MARKET"] = df["MARKET"].map(mapping).fillna(df["MARKET"])
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    df["HOME_TEAM"] = home
    df["AWAY_TEAM"] = away
    df["GAME"] = f"{away} @ {home}"
    return df[["PLAYER", "MARKET", "LINE", "GAME", "HOME_TEAM", "AWAY_TEAM"]]

def main():
    require_key()
    logging.info("Fetching today's NBA player props from The Odds API (DraftKings only)...")

    events = fetch_events_for_local_day()
    all_frames = []

    for e in events:
        eid = e.get("id")
        home, away = e.get("home_team"), e.get("away_team")
        logging.info(f"🏀 {away} @ {home} — fetching player props…")
        rows = fetch_props_for_event(eid)
        if rows:
            df = normalize_props(rows, e)
            all_frames.append(df)
        time.sleep(1.2)  # polite delay

    if not all_frames:
        logging.warning("No props found for the local day window.")
        pd.DataFrame(columns=["PLAYER","MARKET","LINE","GAME","HOME_TEAM","AWAY_TEAM"]).to_csv(OUT_FILE, index=False)
        return

    out = pd.concat(all_frames, ignore_index=True)
    out.to_csv(OUT_FILE, index=False)
    logging.info(f"✅ Saved player props to {OUT_FILE} ({len(out)} rows)")

if __name__ == "__main__":
    main()