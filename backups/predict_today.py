import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# --- Load your combined CSV ---
CSV_FILE = "nba_today_stats.csv"
df = pd.read_csv(CSV_FILE)

# --- Example: create target variable ---
# For now, define "PTS_OVER_20" as target
df['PTS_OVER_20'] = (df['PTS'] > 20).astype(int)

# --- Feature engineering ---
# Simple examples: recent stats (here we just use season stats), team side, arena city/state
df['REB_AST'] = df['REB'] + df['AST']
df['USAGE_FEATURE'] = df['PTS'] + 0.5 * df['REB'] + 0.5 * df['AST']

# Encode categorical variables
df = pd.get_dummies(df, columns=['TEAM_SIDE', 'STATE'], drop_first=True)

# --- Select features and target ---
FEATURES = ['PTS', 'REB', 'AST', 'FG3M', 'MIN', 'REB_AST', 'USAGE_FEATURE'] + \
           [c for c in df.columns if c.startswith('TEAM_SIDE_') or c.startswith('STATE_')]

TARGET = 'PTS_OVER_20'

X = df[FEATURES]
y = df[TARGET]

# --- Split into train/test (for demonstration, you could train on past season data) ---
# Here we just do a quick split to get a working pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Train model ---
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Predict probabilities for today ---
df['OVER_PROB'] = model.predict_proba(X)[:, 1]

# --- Output top 10 best over picks ---
top10 = df.sort_values('OVER_PROB', ascending=False).head(10)
print(top10[['PLAYER', 'TEAM_FULL', 'PTS', 'OVER_PROB']])