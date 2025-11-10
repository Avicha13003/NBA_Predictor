# 3_build_team_context.py  — adds TRAVEL_MILES + TRAVEL_FATIGUE
import pandas as pd, numpy as np, requests, time, datetime, logging, math
from zoneinfo import ZoneInfo
from scipy.stats import zscore
from nba_api.stats.endpoints import TeamDashboardByGeneralSplits
from nba_api.stats.static import teams as nba_teams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_TEAM_SCHEDULE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule"
TEAM_CONTEXT_OUT = "team_context.csv"

CHI_TZ = ZoneInfo("America/Chicago")

# --- Arena coordinates (approximate, stable) ---
# If an arena changes in the future, just update here.
ARENA_COORDS = {
    # East
    "ATL": ("State Farm Arena", 33.7573, -84.3963),
    "BOS": ("TD Garden", 42.3663, -71.0620),
    "BKN": ("Barclays Center", 40.6826, -73.9754),
    "NYK": ("Madison Square Garden", 40.7505, -73.9934),
    "PHI": ("Wells Fargo Center", 39.9012, -75.1719),
    "TOR": ("Scotiabank Arena", 43.6435, -79.3791),
    "CLE": ("Rocket Mortgage FieldHouse", 41.4965, -81.6882),
    "CHI": ("United Center", 41.8807, -87.6742),
    "DET": ("Little Caesars Arena", 42.3410, -83.0550),
    "IND": ("Gainbridge Fieldhouse", 39.7653, -86.1555),
    "MIL": ("Fiserv Forum", 43.0451, -87.9172),
    "MIA": ("Kaseya Center", 25.7814, -80.1880),
    "ORL": ("Kia Center (Amway)", 28.5392, -81.3839),
    "WAS": ("Capital One Arena", 38.8981, -77.0209),
    "CHA": ("Spectrum Center", 35.2251, -80.8392),
    # West
    "GSW": ("Chase Center", 37.7680, -122.3877),
    "LAL": ("Crypto.com Arena", 34.0430, -118.2673),
    "LAC": ("Intuit Dome", 33.9581, -118.3417),
    "PHX": ("Footprint Center", 33.4457, -112.0712),
    "SAC": ("Golden 1 Center", 38.5803, -121.4997),
    "MEM": ("FedExForum", 35.1382, -90.0506),
    "DAL": ("American Airlines Center", 32.7905, -96.8104),
    "HOU": ("Toyota Center", 29.7508, -95.3621),
    "SAS": ("Frost Bank Center", 29.4369, -98.4375),
    "NOP": ("Smoothie King Center", 29.9490, -90.0815),
    "OKC": ("Paycom Center", 35.4634, -97.5149),
    "MIN": ("Target Center", 44.9795, -93.2761),
    "POR": ("Moda Center", 45.5316, -122.6668),
    "DEN": ("Ball Arena", 39.7487, -105.0077),
    "UTA": ("Delta Center", 40.7683, -111.9011),
    "SAC": ("Golden 1 Center", 38.5803, -121.4997),
}

# --- Helpers ---
def _requests_json(url, retries=3, sleep=0.35):
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logging.warning(f"Request failed {url}: {e}")
        time.sleep(sleep)
    return None

def espn_scoreboard_for_date(dt: datetime.date):
    url = f"{ESPN_SCOREBOARD}?dates={dt.strftime('%Y%m%d')}"
    js = _requests_json(url)
    return js.get("events", []) if js else []

def _to_ct_date(iso_ts: str) -> str:
    ts = pd.to_datetime(iso_ts, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.tz_convert(CHI_TZ).date().isoformat()

def parse_espm_games(events):
    games = []
    for ev in events:
        comps = ev.get("competitions", []) or []
        if not comps: continue
        comp = comps[0]
        venue = comp.get("venue", {}) or {}
        addr = venue.get("address", {}) or {}
        teams = comp.get("competitors", []) or []
        if len(teams) != 2: continue
        home = next((t for t in teams if t.get("homeAway") == "home"), None)
        away = next((t for t in teams if t.get("homeAway") == "away"), None)
        if not home or not away: continue
        games.append({
            "GAME_DATE": _to_ct_date(comp.get("date", "")),
            "ARENA": venue.get("fullName", ""),
            "CITY": addr.get("city", ""),
            "STATE": addr.get("state", ""),
            "HOME_ID": str(home.get("id") or ""),
            "HOME_TEAM": (home.get("team", {}) or {}).get("abbreviation", ""),
            "HOME_FULL": (home.get("team", {}) or {}).get("displayName", ""),
            "AWAY_ID": str(away.get("id") or ""),
            "AWAY_TEAM": (away.get("team", {}) or {}).get("abbreviation", ""),
            "AWAY_FULL": (away.get("team", {}) or {}).get("displayName", "")
        })
    return pd.DataFrame(games)

def espn_recent_games(team_id: str, asof_ct_date: datetime.date):
    js = _requests_json(ESPN_TEAM_SCHEDULE.format(team_id=team_id))
    if not js: return []
    rows, cutoff = [], pd.Timestamp(asof_ct_date, tz=CHI_TZ).date()
    for ev in js.get("events", []):
        comps = ev.get("competitions", []) or []
        if not comps: continue
        comp = comps[0]
        st = comp.get("status", {}).get("type", {}) or {}
        if not st.get("completed"): continue
        gdate_ct_str = _to_ct_date(comp.get("date", "")) or ""
        if not gdate_ct_str: continue
        gdate_ct = datetime.date.fromisoformat(gdate_ct_str)
        if gdate_ct > cutoff:  # skip future games / later same-UTC-day games
            continue
        sides = comp.get("competitors", []) or []
        if len(sides) != 2: continue
        t0, t1 = sides
        if str(t0.get("id")) == str(team_id):
            our, opp = t0, t1
        elif str(t1.get("id")) == str(team_id):
            our, opp = t1, t0
        else:
            continue
        def score_val(team):
            val = team.get("score")
            if isinstance(val, dict): val = val.get("value", 0)
            return int(val or 0)
        rows.append({"gameDate": gdate_ct_str, "teamScore": score_val(our), "oppScore": score_val(opp)})
    rows.sort(key=lambda x: x["gameDate"], reverse=True)
    return rows

def build_rest_and_opp_points(espm_ids, asof_ct_date: datetime.date):
    rows = []
    for tid in sorted(set(espm_ids)):
        rec = espn_recent_games(tid, asof_ct_date)
        last = rec[0]["gameDate"] if rec else None
        opp_pts = np.mean([g["oppScore"] for g in rec]) if rec else np.nan
        rows.append({"TEAM_ESPN_ID": str(tid), "LAST_GAME_DATE": last, "OPP_PTS": opp_pts})
        time.sleep(0.08)
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["LAST_GAME_DATE"] = pd.to_datetime(df["LAST_GAME_DATE"], format="%Y-%m-%d", errors="coerce")
    today_ct = pd.Timestamp(asof_ct_date)
    df["DAYS_REST"] = (today_ct - df["LAST_GAME_DATE"]).dt.days.clip(lower=0)
    df["IS_B2B"] = np.where(df["DAYS_REST"] == 1, "Yes", "No")
    df["OPP_PTS_Z"] = zscore(df["OPP_PTS"].fillna(df["OPP_PTS"].mean()))
    return df

def normalize_abbr(abbr):
    mapping = {"NO":"NOP","NY":"NYK","UTAH":"UTA","GS":"GSW","SA":"SAS","WSH":"WAS","PHO":"PHX"}
    return mapping.get(abbr, abbr)

def build_defense_snapshot():
    logging.info("Fetching league defense snapshot (TeamDashboardByGeneralSplits)...")
    rows = []
    for t in nba_teams.get_teams():
        tid, name, abbr = t["id"], t["full_name"], t["abbreviation"]
        def_rating = np.nan
        try:
            dash = TeamDashboardByGeneralSplits(
                team_id=tid,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                season_type_all_star="Regular Season"
            ).get_data_frames()[0]
            candidates = [c for c in dash.columns if "def" in c.lower() and "rating" in c.lower()]
            if candidates:
                def_rating = float(dash.loc[0, candidates[0]])
        except Exception as e:
            logging.warning(f"Could not fetch DEF_RATING for {name}: {e}")
        rows.append({"TEAM_NAME": name, "TEAM_ABBREVIATION": abbr, "DEF_RATING": def_rating})
        time.sleep(0.2)
    base = pd.DataFrame(rows)
    base["DEF_RATING_Z"] = zscore(base["DEF_RATING"].fillna(base["DEF_RATING"].mean()))
    return base

# --- Haversine miles ---
def haversine_miles(lat1, lon1, lat2, lon2):
    R_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return (R_km * c) * 0.621371  # km→miles

def travel_between(abbr_from, abbr_to):
    """Miles from FROM team’s home arena to TO team’s home arena."""
    if abbr_from not in ARENA_COORDS or abbr_to not in ARENA_COORDS:
        return np.nan
    _, lat1, lon1 = ARENA_COORDS[abbr_from]
    _, lat2, lon2 = ARENA_COORDS[abbr_to]
    return round(haversine_miles(lat1, lon1, lat2, lon2), 1)

def travel_fatigue(miles, days_rest):
    """
    Simple fatigue proxy:
      - travel term: miles/1000
      - rest term: max(0, 2 - days_rest) * 0.5
    """
    miles_term = (0 if pd.isna(miles) else miles / 1000.0)
    rest_term = max(0.0, 2.0 - float(days_rest))
    return round(miles_term + 0.5 * rest_term, 3)

# --- Main ---
def main():
    today_ct = datetime.datetime.now(CHI_TZ).date()
    events = espn_scoreboard_for_date(today_ct)
    sched = parse_espm_games(events)
    if sched.empty:
        logging.error("No ESPN games found for today.")
        return
    logging.info(f"Parsed {len(sched)} games for {today_ct}")

    # normalize abbreviations
    sched["HOME_TEAM"] = sched["HOME_TEAM"].apply(normalize_abbr)
    sched["AWAY_TEAM"] = sched["AWAY_TEAM"].apply(normalize_abbr)

    # defense snapshot & rest/opp points
    defense = build_defense_snapshot()
    rest_opp = build_rest_and_opp_points(
        list(sched["HOME_ID"].astype(str)) + list(sched["AWAY_ID"].astype(str)),
        today_ct
    )

    # HOME rows
    home = sched.merge(defense, left_on="HOME_TEAM", right_on="TEAM_ABBREVIATION", how="left")
    home = home.merge(rest_opp.rename(columns={"TEAM_ESPN_ID":"HOME_ID"}), on="HOME_ID", how="left")
    home = home.assign(
        TEAM_ABBREVIATION=sched["HOME_TEAM"],
        TEAM_NAME=sched["HOME_FULL"],
        TEAM_SIDE="Home",
        OPP_TEAM_FULL=sched["AWAY_FULL"],
        OPP_TEAM_ABBR=sched["AWAY_TEAM"],
        CITY=sched["CITY"],
        STATE=sched["STATE"]
    )
    # TRAVEL: home team is at home → 0 miles
    home["TRAVEL_MILES"] = 0.0
    home["TRAVEL_FATIGUE"] = home.apply(lambda r: travel_fatigue(r["TRAVEL_MILES"], r["DAYS_REST"]), axis=1)

    # AWAY rows
    away = sched.merge(defense, left_on="AWAY_TEAM", right_on="TEAM_ABBREVIATION", how="left")
    away = away.merge(rest_opp.rename(columns={"TEAM_ESPN_ID":"AWAY_ID"}), on="AWAY_ID", how="left")
    away = away.assign(
        TEAM_ABBREVIATION=sched["AWAY_TEAM"],
        TEAM_NAME=sched["AWAY_FULL"],
        TEAM_SIDE="Away",
        OPP_TEAM_FULL=sched["HOME_FULL"],
        OPP_TEAM_ABBR=sched["HOME_TEAM"],
        CITY=sched["CITY"],
        STATE=sched["STATE"]
    )
    # TRAVEL: away travels FROM their home arena TO today's home arena
    away["TRAVEL_MILES"] = away.apply(lambda r: travel_between(r["TEAM_ABBREVIATION"], r["OPP_TEAM_ABBR"]), axis=1)
    away["TRAVEL_FATIGUE"] = away.apply(lambda r: travel_fatigue(r["TRAVEL_MILES"], r["DAYS_REST"]), axis=1)

    teams = pd.concat([home, away], ignore_index=True)

    # Opponent mappings (defense & opp-opp-pts)
    opp_def_map = defense.set_index("TEAM_ABBREVIATION")["DEF_RATING_Z"].to_dict()
    opp_opp_pts_map = rest_opp.set_index("TEAM_ESPN_ID")["OPP_PTS_Z"].to_dict()
    abbr_to_espm = {normalize_abbr(r["HOME_TEAM"]): r["HOME_ID"] for _, r in sched.iterrows()}
    abbr_to_espm.update({normalize_abbr(r["AWAY_TEAM"]): r["AWAY_ID"] for _, r in sched.iterrows()})

    teams["OPP_DEF_RATING_Z"] = teams["OPP_TEAM_ABBR"].map(opp_def_map)
    teams["OPP_OPP_PTS_Z"] = teams["OPP_TEAM_ABBR"].map(lambda ab: opp_opp_pts_map.get(abbr_to_espm.get(ab, ""), np.nan))

    # Format LAST_GAME_DATE for readability
    teams["LAST_GAME_DATE"] = pd.to_datetime(teams["LAST_GAME_DATE"], errors="coerce").dt.strftime("%m/%d/%Y")

    keep = [
        "TEAM_ABBREVIATION","TEAM_NAME","TEAM_SIDE","CITY","STATE",
        "DEF_RATING","DEF_RATING_Z","LAST_GAME_DATE","DAYS_REST","IS_B2B",
        "TRAVEL_MILES","TRAVEL_FATIGUE",
        "OPP_PTS","OPP_PTS_Z","OPP_TEAM_FULL","OPP_DEF_RATING_Z","OPP_OPP_PTS_Z"
    ]
    out = teams[keep].drop_duplicates(subset=["TEAM_ABBREVIATION","TEAM_SIDE"]).reset_index(drop=True)
    out.to_csv(TEAM_CONTEXT_OUT, index=False)
    logging.info(f"✅ Wrote {TEAM_CONTEXT_OUT} ({len(out)} teams)")
    print(out.head(12))

if __name__ == "__main__":
    main()