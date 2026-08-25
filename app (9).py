import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="PL Predictor - Cała Kolejka", page_icon="📅", layout="wide")
st.title("📅 Premier League Predictor (Kolejka LIVE)")
st.markdown("Model uczy się na bazie 4 lat historii, a następnie symuluje **wszystkie nadchodzące mecze** w terminarzu.")

# --- TŁUMACZ NAZW (API vs CSV) ---
TEAM_MAPPING = {
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa", "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford", "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton", "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich", "Leicester City FC": "Leicester", "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City", "Manchester United FC": "Man United", "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottm Forest", "Southampton FC": "Southampton", "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham", "Wolverhampton Wanderers FC": "Wolves"
}

# --- POLE NA KLUCZ API (Dla Terminarza) ---
st.sidebar.header("🔑 Ustawienia API")
api_key = st.sidebar.text_input("Klucz API (Football-Data):", type="password")

if not api_key:
    st.warning("👈 Wklej klucz API po lewej stronie, aby pobrać terminarz!")
    st.stop()

# --- 1. TRENOWANIE MODELU NA CSV (Historia) ---
@st.cache_data(ttl=86400)
def load_and_train_model():
    seasons = ["2324", "2425", "2526", "2627"]
    dfs = []
    for s in seasons:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
            df_temp = pd.read_csv(url, on_bad_lines='skip')
            df_temp = df_temp[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
            dfs.append(df_temp)
        except:
            continue
            
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Trenowanie Elo / Formy
    ALPHA = 0.15 
    HOME_ADVANTAGE = 1.2 
    AVG_GOALS = 1.4      
    ratings = {}

    def init_team(team):
        if team not in ratings:
            ratings[team] = {'attack': 1.0, 'defense': 1.0, 'form_trend': 0.0}

    for index, row in df_all.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        home_g, away_g = row['FTHG'], row['FTAG']
        
        init_team(home)
        init_team(away)
        
        pred_h = ratings[home]['attack'] * ratings[away]['defense'] * HOME_ADVANTAGE * AVG_GOALS
        pred_a = ratings[away]['attack'] * ratings[home]['defense'] * (1 / HOME_ADVANTAGE) * AVG_GOALS
        
        err_h, err_a = home_g - pred_h, away_g - pred_a
        
        ratings[home]['attack'] = max(0.1, ratings[home]['attack'] + ALPHA * err_h * 0.1)
        ratings[away]['defense'] = max(0.1, ratings[away]['defense'] + ALPHA * err_h * 0.1)
        ratings[away]['attack'] = max(0.1, ratings[away]['attack'] + ALPHA * err_a * 0.1)
        ratings[home]['defense'] = max(0.1, ratings[home]['defense'] + ALPHA * err_a * 0.1)
        
        ratings[home]['form_trend'] = (ratings[home]['form_trend'] * 0.8) + (err_h * 0.2)
        ratings[away]['form_trend'] = (ratings[away]['form_trend'] * 0.8) + (err_a * 0.2)

    return ratings, HOME_ADVANTAGE, AVG_GOALS

ratings, HOME_ADVANTAGE, AVG_GOALS = load_and_train_model()

# --- 2. POBIERANIE TERMINARZA Z API ---
@st.cache_data(ttl=3600)
def get_upcoming_matches(token):
    headers = {'X-Auth-Token': token}
    # Pobieramy mecze zaplanowane (SCHEDULED) z Premier League (PL)
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        st.error(f"Błąd pobierania terminarza (Kod {response.status_code}).")
        return []
        
    data = response.json()
    matches = []
    # Pobieramy tylko najbliższe 10 meczów (jedna pełna kolejka)
    for match in data.get('matches', [])[:10]:
        home_api = match['homeTeam']['name']
        away_api = match['awayTeam']['name']
        
        # Tłumaczenie nazw
        home_csv = TEAM_MAPPING.get(home_api, home_api)
        away_csv = TEAM_MAPPING.get(away_api, away_api)
        
        matches.append({
            'Data': match['utcDate'][:10],
            'Gospodarz': home_csv,
            'Gość': away_csv
        })
    return matches

upcoming_matches = get_upcoming_matches(api_key)

# --- 3. SYMULACJA CAŁEJ KOLEJKI ---
if upcoming_matches:
    st.subheader("🔮 Symulacja najbliższej kolejki")
    
    results_list = []
    
    for m in upcoming_matches:
        home, away = m['Gospodarz'], m['Gość']
        
        # Zabezpieczenie, jeśli drużyny nie ma w bazie (np. nowy beniaminek ze złą nazwą)
        if home not in ratings or away not in ratings:
            continue
            
        home_xg = ratings[home]['attack'] * ratings[away]['defense'] * HOME_ADVANTAGE * AVG_GOALS
        away_xg = ratings[away]['attack'] * ratings[home]['defense'] * (1 / HOME_ADVANTAGE) * AVG_GOALS
        
        matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                
        win_h = np.tril(matrix, -1).sum() * 100
        draw = np.trace(matrix) * 100
        win_a = np.triu(matrix, 1).sum() * 100
        
        results_list.append({
            "Data": m['Data'],
            "Mecz": f"{home} vs {away}",
            "Wygra Gospodarz (1)": f"{win_h:.1f}%",
            "Remis (X)": f"{draw:.1f}%",
            "Wygra Gość (2)": f"{win_a:.1f}%",
            "xG Gospodarz": round(home_xg, 2),
            "xG Gość": round(away_xg, 2)
        })

    # Wyświetlanie wyników w ładnej tabeli
    df_results = pd.DataFrame(results_list)
    st.dataframe(df_results, use_container_width=True)
    
else:
    st.info("Brak zaplanowanych meczów do wyświetlenia (lub problem z API).")
