import streamlit as st
from predictor import load_and_prepare_data, train_model, dynamic_rule_with_explanation, generate_strategy_recommendations

# Load data
data_files = ['NFLPlaybyPlay2015.csv', '2017-2025_scores.csv']
data, features = load_and_prepare_data(data_files)
model = train_model(data, features)

# Streamlit interface
st.title("AI NFL Game Predictor")

st.write("Upload a CSV file with new games to predict:")

uploaded_file = st.file_uploader("Choose CSV", type="csv")

if uploaded_file:
    new_games = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(new_games.head())
    
    # Make predictions
    predictions = new_games.apply(dynamic_rule_with_explanation, axis=1)
    new_games['Prediction'] = predictions.apply(lambda x: 'Win' if x[0]==1 else 'Loss')
    new_games['Explanation'] = predictions.apply(lambda x: x[1])
    
    # Show results
    st.write("Predictions and explanations:")
    st.dataframe(new_games[['Prediction','Explanation']])
    
    # Show strategy recommendations
    strategy_recs = generate_strategy_recommendations(model.feature_importances_, data)
    st.write("Strategy Recommendations:")
    for rec in strategy_recs:
        st.write("-", rec)
