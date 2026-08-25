import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="PL Predictor AI", page_icon="🧠")
st.title("🧠 Premier League Predictor (AI & Form)")
st.markdown("Ten model iteruje przez 4 lata historii mecz po meczu. Uczy się na błędach z przeszłości, a **największą wagę przykłada do obecnej formy** drużyn.")

@st.cache_data(ttl=86400)
def load_historical_data():
    seasons = ["2324", "2425", "2526", "2627"]
    dfs = []
    for s in seasons:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
            df_temp = pd.read_csv(url, on_bad_lines='skip')
            df_temp = df_temp[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='%d/%m/%Y')
            dfs.append(df_temp)
        except Exception:
            continue
            
    df_all = pd.concat(dfs, ignore_index=True).sort_values('Date').reset_index(drop=True)
    return df_all

df = load_historical_data()
current_teams = sorted(list(set(df.tail(40)['HomeTeam'])))

# --- TRENOWANIE MODELU (ITERACYJNE UCZENIE) ---
# Współczynnik uczenia (Learning Rate). Im wyższy, tym model bardziej reaguje na ostatnie mecze (formę)
ALPHA = 0.15 
HOME_ADVANTAGE = 1.2 # Stała przewaga własnego boiska
AVG_GOALS = 1.4      # Bazowa średnia goli

# Słownik przechowujący dynamiczne ratingi
ratings = {}

# Funkcja inicjalizująca nowe drużyny
def init_team(team):
    if team not in ratings:
        ratings[team] = {'attack': 1.0, 'defense': 1.0, 'form_trend': 0.0}

# Pętla przez historię - dzień po dniu, mecz po meczu
for index, row in df.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    home_goals = row['FTHG']
    away_goals = row['FTAG']
    
    init_team(home)
    init_team(away)
    
    # 1. Przewidywanie modelu na ten moment historii
    pred_home_xg = ratings[home]['attack'] * ratings[away]['defense'] * HOME_ADVANTAGE * AVG_GOALS
    pred_away_xg = ratings[away]['attack'] * ratings[home]['defense'] * (1 / HOME_ADVANTAGE) * AVG_GOALS
    
    # 2. Obliczanie błędu (Rzeczywistość vs Oczekiwania)
    home_error = home_goals - pred_home_xg
    away_error = away_goals - pred_away_xg
    
    # 3. Aktualizacja ratingów i "formy" na podstawie błędów
    # Gospodarze: Strzelili więcej niż zakładano? Atak rośnie, obrona gości słabnie.
    ratings[home]['attack'] += ALPHA * home_error * 0.1
    ratings[away]['defense'] += ALPHA * home_error * 0.1 # Tracą obronę, jeśli stracili niespodziewanie gole
    
    # Goście:
    ratings[away]['attack'] += ALPHA * away_error * 0.1
    ratings[home]['defense'] += ALPHA * away_error * 0.1
    
    # Zapis trendu formy (skumulowany błąd z ostatnich spotkań jako momentum)
    ratings[home]['form_trend'] = (ratings[home]['form_trend'] * 0.8) + (home_error * 0.2)
    ratings[away]['form_trend'] = (ratings[away]['form_trend'] * 0.8) + (away_error * 0.2)
    
    # Zabezpieczenie przed ratingiem spadającym poniżej 0.1
    ratings[home]['attack'] = max(0.1, ratings[home]['attack'])
    ratings[home]['defense'] = max(0.1, ratings[home]['defense'])
    ratings[away]['attack'] = max(0.1, ratings[away]['attack'])
    ratings[away]['defense'] = max(0.1, ratings[away]['defense'])

# --- INTERFEJS UŻYTKOWNIKA ---
st.sidebar.header("Wybierz dzisiejszy mecz")
home_team = st.sidebar.selectbox("Gospodarz (Home)", current_teams)
away_team = st.sidebar.selectbox("Gość (Away)", current_teams, index=1)

if home_team != away_team:
    
    # Wyciąganie najnowszych, zaktualizowanych ratingów (stan na dzisiaj)
    home_att = ratings[home_team]['attack']
    home_def = ratings[home_team]['defense']
    home_trend = ratings[home_team]['form_trend']
    
    away_att = ratings[away_team]['attack']
    away_def = ratings[away_team]['defense']
    away_trend = ratings[away_team]['form_trend']
    
    # Obliczanie ostatecznego xG
    final_home_xg = home_att * away_def * HOME_ADVANTAGE * AVG_GOALS
    final_away_xg = away_att * home_def * (1 / HOME_ADVANTAGE) * AVG_GOALS
    
    # Generowanie macierzy prawdopodobieństw (Poisson)
    max_g = 6
    matrix = np.zeros((max_g, max_g))
    for i in range(max_g):
        for j in range(max_g):
            matrix[i, j] = poisson.pmf(i, final_home_xg) * poisson.pmf(j, final_away_xg)
            
    win_h = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    win_a = np.triu(matrix, 1).sum()

    st.write("---")
    
    # Pokazywanie Momentum/Formy
    st.subheader("🔥 Obecne Momentum (Zgodność z oczekiwaniami w ostatnich meczach)")
    st.markdown("Wskaźnik powyżej 0 oznacza, że drużyna regularnie **przebija oczekiwania** (jest w gazie). Spadki poniżej 0 to dołek formy.")
    
    c_f1, c_f2 = st.columns(2)
    c_f1.metric(f"Wskaźnik Formy: {home_team}", round(home_trend, 2), delta=round(home_trend, 2))
    c_f2.metric(f"Wskaźnik Formy: {away_team}", round(away_trend, 2), delta=round(away_trend, 2))

    st.write("---")

    # Wyświetlanie xG
    st.subheader("🎯 Oczekiwane Gole na dzisiejszy mecz (xG)")
    c1, c2 = st.columns(2)
    c1.metric(f"{home_team}", round(final_home_xg, 2))
    c2.metric(f"{away_team}", round(final_away_xg, 2))
    
    st.write("---")
    
    # Prawdopodobieństwa
    st.subheader("📊 Prawdopodobieństwo Wyniku")
    col1, col2, col3 = st.columns(3)
    col1.info(f"Wygra {home_team}\n\n### {win_h*100:.1f}%")
    col2.warning(f"Remis\n\n### {draw*100:.1f}%")
    col3.success(f"Wygra {away_team}\n\n### {win_a*100:.1f}%")

else:
    st.sidebar.error("Wybierz dwie różne drużyny!")
