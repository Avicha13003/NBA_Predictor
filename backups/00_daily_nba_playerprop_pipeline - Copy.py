# 00_daily_nba_pipeline_props.py — Full daily NBA workflow (including player props)
import subprocess, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

SCRIPTS = [
    "0_fetch_data_nba_api.py",        # daily player/game data
    "0a_fetch_nba_injuries_rotowire.py",        # daily injuries
    "1_fetch_boxscores.py",           # update boxscores
    "2_generate_rolling_metrics_playerprops.py",  # rolling averages, trends
    "3_build_team_context.py",        # defense, travel, rest
    "3a_fetch_propoddsapi.py",        # new: DraftKings/The Odds API props
    "5_props_predictor.py",           # new: player prop prediction engine
]

def run_script(name):
    logging.info(f"🚀 Running {name} ...")
    result = subprocess.run(["python", name], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"❌ {name} failed.\n{result.stderr}")
    else:
        logging.info(f"✅ {name} completed successfully.")
        if result.stdout.strip():
            logging.info(result.stdout.strip())

def main():
    logging.info("🏀 Starting full NBA daily pipeline (with player props)...")
    for s in SCRIPTS:
        run_script(s)
    logging.info("🎯 All scripts completed successfully! Daily pipeline finished.")

if __name__ == "__main__":
    main()