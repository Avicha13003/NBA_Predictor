# run_all_daily.py — orchestrates your full daily NBA prediction pipeline (Option 1: Stop on Failure)
import subprocess, datetime, sys, os

# ==== Configuration ====
SCRIPTS = [
    "0_fetch_data_nba_api.py",
    "1_fetch_boxscores.py",
    "2_generate_rolling_metrics.py",
    "3_build_team_context.py",
    "0a_fetch_nba_injuries_rotowire.py",
    "4a_nba_ml_predictor.py"
]

LOG_FILE = "daily_run_log.txt"

def run_script(script_name):
    """Run a script and stop execution if it fails."""
    print(f"\n🚀 Starting: {script_name}")
    start_time = datetime.datetime.now()
    try:
        subprocess.run([sys.executable, script_name], check=True)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"✅ Completed {script_name} in {duration:.1f}s\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {script_name} failed. Stopping pipeline.")
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.datetime.now()}] ❌ {script_name} FAILED — {e}\n")
        sys.exit(1)

def main():
    print("\n🏀 Starting NBA Prediction Daily Workflow")
    print("==========================================")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n=== Daily Run Started: {datetime.datetime.now()} ===\n")

    for script in SCRIPTS:
        run_script(script)

    try:
    print("🎯 All scripts completed successfully!")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"✅ Completed full daily run: {datetime.datetime.now()}\n")
except Exception as e:
    print(f"⚠️ Warning: Could not write final log entry ({e})")

if __name__ == "__main__":
    main()