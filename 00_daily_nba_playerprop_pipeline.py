# 00_daily_nba_playerprop_pipeline.py — Full daily NBA workflow (UTF-8 safe)
import subprocess
import logging
import datetime
import sys
import os

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Ensure UTF-8 output (fixes Windows emoji encoding errors)
sys.stdout.reconfigure(encoding='utf-8')

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# === List of scripts to run ===
SCRIPTS = [
    "0_fetch_data_nba_api.py",
    "0a_fetch_nba_injuries_rotowire.py",
    "1a_fetch_boxscores.py",
    "2a_generate_rolling_metrics_playerprops.py",
    "3_build_team_context.py",
    "3a_fetch_propoddsapi.py",
    "5_props_predictor.py",
]

def run_script(name: str) -> bool:
    """Run a Python script and capture its output."""
    logging.info(f"🚀 Running {name} ...")
    result = subprocess.run(["python", name], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"❌ {name} failed with error:\n{result.stderr.strip()}")
        return False
    else:
        logging.info(f"✅ {name} completed successfully.")
        if result.stdout.strip():
            logging.info(result.stdout.strip())
        return True

def main():
    start = datetime.datetime.now()
    logging.info("🏀 Starting full NBA daily pipeline (with player props)...")

    for s in SCRIPTS:
        success = run_script(s)
        if not success:
            logging.error(f"⚠️ Halting pipeline due to error in {s}.")
            break

    elapsed = datetime.datetime.now() - start
    logging.info(f"🎯 All scripts completed. Total runtime: {elapsed.seconds // 60}m {elapsed.seconds % 60}s")

if __name__ == "__main__":
    main()