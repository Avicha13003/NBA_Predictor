# fetch_nba_injuries_rotowire.py
import pandas as pd
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# === Config ===
INPUT_CSV = "rotowire_injuries.csv"        # your exported Rotowire injury report
OUT_CSV   = "nba_injuries.csv"
OUT_JSON  = "nba_injuries.json"

# Statuses that imply the player should be treated as OUT
OUT_STATUSES = ["Out", "Doubtful", "Suspended"]

# Statuses that imply partial limitation (useful if you later want weighting)
LIMITED_STATUSES = ["Questionable", "Probable", "Day-To-Day"]

def clean_injury_data():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"File not found: {INPUT_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(INPUT_CSV)
    df.columns = [c.strip().title() for c in df.columns]  # normalize headers

    # Ensure required columns exist
    expected = ["Player", "Team", "Pos", "Injury", "Status"]
    for col in expected:
        if col not in df.columns:
            logging.error(f"Missing expected column: {col}")
            return pd.DataFrame()

    # Clean up whitespace and casing
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Team"]   = df["Team"].astype(str).str.upper().str.strip()
    df["Pos"]    = df["Pos"].astype(str).str.upper().str.strip()
    df["Status"] = df["Status"].astype(str).str.title().str.strip()
    df["Injury"] = df["Injury"].fillna("").astype(str).str.strip()

    # Create numeric flags
    df["INJURY_FLAG"] = df["Status"].apply(
        lambda s: 1 if s in OUT_STATUSES else (0.5 if s in LIMITED_STATUSES else 0)
    )

    # Simplified status category
    df["INJURY_CATEGORY"] = df["Status"].apply(
        lambda s: "Out" if s in OUT_STATUSES
        else ("Limited" if s in LIMITED_STATUSES else "Active")
    )

    # Optional short flag (for merging into ML predictor)
    df["IS_OUT"] = (df["INJURY_CATEGORY"] == "Out").astype(int)
    df["IS_LIMITED"] = (df["INJURY_CATEGORY"] == "Limited").astype(int)

    # Reorder + clean output
    out_cols = ["Player", "Team", "Pos", "Injury", "Status",
                "INJURY_FLAG", "INJURY_CATEGORY", "IS_OUT", "IS_LIMITED"]
    df = df[out_cols].drop_duplicates(subset=["Player"])

    # Save outputs
    df.to_csv(OUT_CSV, index=False)
    df.to_json(OUT_JSON, orient="records", indent=2)
    logging.info(f"✅ Injury data cleaned and saved → {OUT_CSV} & {OUT_JSON} ({len(df)} players)")

    # Sample printout
    logging.info("Sample output:")
    print(df.head(10))
    return df

if __name__ == "__main__":
    clean_injury_data()