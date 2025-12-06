import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# 1. Set your folder path here
# ---------------------------
data_folder = r"Downloads\New Folder"  # <-- change this to your folder path

# ---------------------------
# 2. Unzip all .zip files
# ---------------------------
for file in os.listdir(data_folder):
    if file.endswith(".zip"):
        zip_path = os.path.join(data_folder, file)
        extract_path = os.path.join(data_folder, file.replace(".zip", ""))
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted: {file} -> {extract_path}")

# ---------------------------
# 3. Find all CSV files
# ---------------------------
csv_files = []
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print("\nFound CSV files:")
for f in csv_files:
    print(f)

# ---------------------------
# 4. Load and combine datasets (if compatible)
# ---------------------------
dfs = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

# For simplicity, use the first dataset as the main one
data = dfs[0].copy()
print("\nUsing dataset:", csv_files[0])
print("Shape:", data.shape)

# ---------------------------
# 5. Preprocessing / Feature Engineering
# ---------------------------

# Example factors (adjust to your dataset columns)
# You should update column names to match your CSV
data['win'] = (data['team_score'] > data['opponent_score']).astype(int)
data['turnover_margin'] = data['team_turnovers'] - data['opponent_turnovers']
data['home_game'] = data['location'].apply(lambda x: 1 if x=='home' else 0)

# Select features for analysis
features = ['turnover_margin', 'home_game', 'offensive_yards', 'defensive_yards']  # adjust as needed
target = 'win'

# Drop rows with missing values in selected columns
data = data.dropna(subset=features+[target])

# ---------------------------
# 6. Correlation Analysis
# ---------------------------
corr = data[features + [target]].corr()
print("\nCorrelation matrix:")
print(corr)

# Visualize correlation
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ---------------------------
# 7. Rule-Based Predictor
# Example: simple rules based on top correlations
# ---------------------------
# Suppose turnover_margin > 0 and home_game = 1 predicts win
def rule_based_predict(row):
    if row['turnover_margin'] > 0 and row['home_game'] == 1:
        return 1
    elif row['turnover_margin'] > 0:
        return 1
    else:
        return 0

data['rule_pred'] = data.apply(rule_based_predict, axis=1)
rule_acc = accuracy_score(data[target], data['rule_pred'])
print(f"\nRule-Based Predictor Accuracy: {rule_acc:.3f}")

# ---------------------------
# 8. Machine Learning Predictor (optional)
# ---------------------------
X = data[features]
y = data[target]

# Split into training and recent holdout set (20% recent games)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)
y_pred = ml_model.predict(X_test)

ml_acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy on Recent Games: {ml_acc:.3f}")

# Feature importance
importance = pd.Series(ml_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance:")
print(importance)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - ML Model")
plt.show()

print("\nPipeline complete!")
