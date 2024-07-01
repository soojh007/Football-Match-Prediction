
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
# import joblib # to save the model will integrate it later 

@st.cache_data
def load_data():
    matches = pd.read_csv('Bundesliga/data.csv')
    matches.dropna(inplace=True)
    return matches

def preprocess_data(matches):
    y_home = matches['FTHG']
    y_away = matches['FTAG']

    matches['goal_difference'] = matches['FTHG'] - matches['FTAG']
    matches['home_team_form'] = matches.groupby('HomeTeam')['goal_difference'].rolling(5).mean().reset_index(level=0, drop=True)
    matches['away_team_form'] = matches.groupby('AwayTeam')['goal_difference'].rolling(5).mean().reset_index(level=0, drop=True)

    features = ['HomeTeam', 'AwayTeam', 'home_team_form', 'away_team_form']
    X = matches[features]
    X.fillna(0, inplace=True)
    
    return X, y_home, y_away

def train_models(X, y_home, y_away):
    X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
    _, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

    numeric_features = ['home_team_form', 'away_team_form']
    categorical_features = ['HomeTeam', 'AwayTeam']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    home_goal_model = RandomForestRegressor(n_estimators=100, random_state=42)
    home_goal_model.fit(X_train, y_home_train)

    away_goal_model = RandomForestRegressor(n_estimators=100, random_state=42)
    away_goal_model.fit(X_train, y_away_train)

    return home_goal_model, away_goal_model, preprocessor

def custom_round(value):
    if value - int(value) >= 0.8:
        return int(value) + 1
    else:
        return int(value)

def calculate_average_form(team, historical_matches, home_or_away):
    if home_or_away == 'home':
        avg_form = historical_matches[historical_matches['HomeTeam'] == team]['goal_difference'].rolling(5).mean().iloc[-1]
    else:
        avg_form = historical_matches[historical_matches['AwayTeam'] == team]['goal_difference'].rolling(5).mean().iloc[-1]
    return avg_form

def predict_match(HomeTeam, AwayTeam, historical_matches, preprocessor, home_goal_model, away_goal_model):
    if historical_matches[historical_matches['HomeTeam'] == HomeTeam].shape[0] < 5:
        home_team_form = historical_matches[historical_matches['HomeTeam'] == HomeTeam]['goal_difference'].mean()
    else:
        home_team_form = calculate_average_form(HomeTeam, historical_matches, 'home')

    if historical_matches[historical_matches['AwayTeam'] == AwayTeam].shape[0] < 5:
        away_team_form = historical_matches[historical_matches['AwayTeam'] == AwayTeam]['goal_difference'].mean()
    else:
        away_team_form = calculate_average_form(AwayTeam, historical_matches, 'away')

    if pd.isna(home_team_form):
        home_team_form = 0
    if pd.isna(away_team_form):
        away_team_form = 0

    new_match = pd.DataFrame({
        'HomeTeam': [HomeTeam],
        'AwayTeam': [AwayTeam],
        'home_team_form': [home_team_form],
        'away_team_form': [away_team_form]
    })

    new_match_preprocessed = preprocessor.transform(new_match)

    home_goals = custom_round(home_goal_model.predict(new_match_preprocessed)[0])
    away_goals = custom_round(away_goal_model.predict(new_match_preprocessed)[0])

    if home_goals > away_goals:
        return 'Home Win', home_goals, away_goals
    elif home_goals < away_goals:
        return 'Away Win', home_goals, away_goals
    else:
        return 'Draw', home_goals, away_goals

def bundesliga():

    st.image('Bundesliga/logo.png', width=200)

    st.title('Bundesliga Football Match Outcome Predictor')

    Note_message = """The model has been trained on 25 years of historical results (1999-2024). It makes predictions based on past encounters between the teams and their current form. Please note that these predictions are not guaranteed to be accurate and should be used as a guide rather than a definitive forecast. Factors not accounted for by the model can influence match outcomes."""
    
    st.write("")
    with st.expander("Note", expanded=False):
        st.markdown(Note_message)

    team_names = [
        'Duisburg', 'Wolfsburg', 'Bayern Munich', 'Ein Frankfurt', 'Kaiserslautern', 'Schalke 04', 
        'Stuttgart', 'Hertha', 'Ulm', 'Bielefeld', 'Unterhaching', 'Dortmund', 'Freiburg', 'Hamburg',
        'Munich 1860', 'Werder Bremen', 'Hansa Rostock', 'Leverkusen', 'Cottbus', 'Bochum', 'FC Koln', 
        'Mgladbach', 'St Pauli', 'Nurnberg', 'Hannover', 'Mainz', 'Aachen', 'Karlsruhe', 'Hoffenheim', 
        'Augsburg', 'Greuther Furth', 'Fortuna Dusseldorf', 'Braunschweig', 'Paderborn', 
        'Darmstadt', 'Ingolstadt', 'RB Leipzig', 'Union Berlin', 'Heidenheim',
    ]

    highlighted_team_names = ', '.join([f"<span style='color: blue;'>{name}</span>" for name in team_names])
    st.write("Please follow the naming convention for the team names. The conventions are: ")
    st.markdown(f"{highlighted_team_names}", unsafe_allow_html=True)

    HomeTeam = st.text_input('Enter Home Team:')
    AwayTeam = st.text_input('Enter Away Team:')

    if st.button('Predict'):
        if HomeTeam and AwayTeam:
            with st.spinner('Processing...'):
                if 'home_goal_model' not in st.session_state or 'away_goal_model' not in st.session_state:
                    matches = load_data()
                    X, y_home, y_away = preprocess_data(matches)
                    home_goal_model, away_goal_model, preprocessor = train_models(X, y_home, y_away)
                    st.session_state['home_goal_model'] = home_goal_model
                    st.session_state['away_goal_model'] = away_goal_model
                    st.session_state['preprocessor'] = preprocessor
                    st.session_state['matches'] = matches
                else:
                    home_goal_model = st.session_state['home_goal_model']
                    away_goal_model = st.session_state['away_goal_model']
                    preprocessor = st.session_state['preprocessor']
                    matches = st.session_state['matches']

                result, home_goals, away_goals = predict_match(HomeTeam, AwayTeam, matches, preprocessor, home_goal_model, away_goal_model)
            st.success('Done!')
            st.write(f'Predicted goals: {HomeTeam} {home_goals} - {away_goals} {AwayTeam}')

            if result == 'Home Win':
                winning_team = HomeTeam
                st.write(f'The match result prediction: {winning_team} wins the match!')

            elif result == 'Away Win':
                winning_team = AwayTeam
                st.write(f'The match result prediction: {winning_team} wins the match!')
            else:
                winning_team = 'Draw'
                st.write(f'The match result prediction: The match ends in a draw!')
        else:
            st.write('Please enter both team names.')

if __name__ == "__main__":
    bundesliga()
