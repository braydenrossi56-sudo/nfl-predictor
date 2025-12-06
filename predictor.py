import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_and_prepare_data(files):
    dfs = [pd.read_csv(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)
    data.fillna(0, inplace=True)
    if 'home_team_win' not in data.columns and 'home_team_score' in data.columns:
        data['home_team_win'] = (data['home_team_score'] > data['away_team_score']).astype(int)
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    if 'home_team_win' in numeric_cols:
        numeric_cols.remove('home_team_win')
    features = numeric_cols
    return data, features

def train_model(data, features):
    X = data[features]
    y = data['home_team_win']
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    return rf

# Add your dynamic_rule_with_explanation() and generate_strategy_recommendations() functions here
