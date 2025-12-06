import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. LOAD + PREPARE DATA
# ---------------------------------------------------------
def load_and_prepare_data(files):
    """
    Load multiple CSV files, merge them, clean NA, and build features + labels.
    """

    # Load all CSVs
    dfs = [pd.read_csv(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)

    # Fill missing values
    data.fillna(0, inplace=True)

    # Build win/loss label if not present
    if "home_team_win" not in data.columns:
        if "home_team_score" in data.columns and "away_team_score" in data.columns:
            data["home_team_win"] = (data["home_team_score"] > data["away_team_score"]).astype(int)
        else:
            raise ValueError("Dataset missing required scoring columns to compute home_team_win.")

    # Select numeric columns automatically
    numeric_cols = data.select_dtypes(include="number").columns.tolist()

    # home_team_win is the label, not a feature
    if "home_team_win" in numeric_cols:
        numeric_cols.remove("home_team_win")

    features = numeric_cols

    return data, features


# ---------------------------------------------------------
# 2. TRAIN MACHINE LEARNING MODEL
# ---------------------------------------------------------
def train_model(data, features):
    """
    Train a RandomForest win/loss predictor.
    """
    X = data[features]
    y = data["home_team_win"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    model.fit(X, y)
    return model


# ---------------------------------------------------------
# 3. RULE-BASED PREDICTOR
# ---------------------------------------------------------
def dynamic_rule_with_explanation(row):
    """
    A simple rule-based predictor with human-readable explanation.
    """

    explanation = []

    # Rule 1: compare historical scoring
    if "home_team_score" in row and "away_team_score" in row:
        if row["home_team_score"] > row["away_team_score"]:
            explanation.append("Home team historically scores more.")
            return 1, " | ".join(explanation)
        elif row["home_team_score"] < row["away_team_score"]:
            explanation.append("Away team historically scores more.")
            return 0, " | ".join(explanation)

    # Rule 2: fallback rule
    explanation.append("Insufficient scoring data. Defaulting to home team loss.")
    return 0, " | ".join(explanation)


# ---------------------------------------------------------
# 4. STRATEGY RECOMMENDATION ENGINE
# ---------------------------------------------------------
def generate_strategy_recommendations(row):
    """
    Produce basic coaching/strategy recommendations based on metric thresholds.
    """

    tips = []

    # Offense
    if "home_team_score" in row and row["home_team_score"] < 20:
        tips.append("Home team may need to improve offensive efficiency.")

    # Defense
    if "away_team_score" in row and row["away_team_score"] > 25:
        tips.append("Home defense must adjust to stop high-scoring opponents.")

    # Balance
    if "home_team_score" in row and "away_team_score" in row:
        diff = row["home_team_score"] - row["away_team_score"]
        if diff < -10:
            tips.append("Significant defensive weaknesses detected.")
        elif diff > 10:
            tips.append("Offense performing well; maintain current play-calling balance.")

    if not tips:
        tips.append("No major strategy adjustments recommended.")

    return " | ".join(tips)

