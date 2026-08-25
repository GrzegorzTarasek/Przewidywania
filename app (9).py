import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="PL Predictor - Następna Kolejka", page_icon="⚽", layout="wide")
st.title("⚽ Premier League Predictor (Najbliższa Kolejka)")
st.markdown("Aplikacja automatycznie wykrywa nadchodzącą kolejkę i symuluje wyniki za pomocą modelu AI (historia + forma).")

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

st.sidebar.header("🔑 Autoryzacja API")
raw_api_key = st.sidebar.text_input("Klucz API (Football-Data):", type="password")
api_key = raw_api_key.strip() if raw_api_key else ""

if not api_key:
    st.warning("👈 Wklej swój klucz API w panelu po lewej stronie, aby kontynuować.")
    st.stop()

# --- 1. TRENOWANIE MODELU NA HISTORII CSV ---
@st.cache_data(ttl=86400)
def train_model():
    seasons = ["2324", "2425", "2526", "2627"]
    dfs = []
    for s in seasons:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
            df_temp = pd.read_csv(url, on_bad_lines='skip')[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
            dfs.append(df_temp)
        except:
            continue
            
    df_all = pd.concat(dfs, ignore_index=True)
    ALPHA, HOME_ADV, AVG_GOALS = 0.15, 1.2, 1.4      
    ratings = {}

    def init_t(team):
        if team not in ratings: ratings[team] = {'attack': 1.0, 'defense': 1.0, 'trend': 0.0}

    for _, row in df_all.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        init_t(h); init_t(a)
        
        pred_h = ratings[h]['attack'] * ratings[a]['defense'] * HOME_ADV * AVG_GOALS
        pred_a = ratings[a]['attack'] * ratings[h]['defense'] * (1 / HOME_ADV) * AVG_GOALS
        
        err_h, err_a = row['FTHG'] - pred_h, row['FTAG'] - pred_a
        
        ratings[h]['attack'] = max(0.1, ratings[h]['attack'] + ALPHA * err_h * 0.1)
        ratings[a]['defense'] = max(0.1, ratings[a]['defense'] + ALPHA * err_h * 0.1)
        ratings[a]['attack'] = max(0.1, ratings[a]['attack'] + ALPHA * err_a * 0.1)
        ratings[h]['defense'] = max(0.1, ratings[h]['defense'] + ALPHA * err_a * 0.1)
        
        ratings[h]['trend'] = (ratings[h]['trend'] * 0.8) + (err_h * 0.2)
        ratings[a]['trend'] = (ratings[a]['trend'] * 0.8) + (err_a * 0.2)

    return ratings, HOME_ADV, AVG_GOALS

ratings, HOME_ADV, AVG_GOALS = train_model()

# --- 2. AUTOMATYCZNE POBIERANIE NAJBLIŻSZEJ KOLEJKI Z API ---
@st.cache_data(ttl=3600)
def get_next_matchday_matches(token):
    headers = {'X-Auth-Token': token}
    # Jawnie wskazujemy sezon 2026, aby uniknąć błędów 400 z parsowaniem domyślnych zakresów przez API
    url = "https://api.football-data.org/v4/competitions/PL/matches?season=2026"
    
    try:
        response = requests.get(url, headers=headers)
    except Exception as e:
        return None, f"Błąd połączenia: {e}"
    
    if response.status_code != 200:
        return None, f"Błąd API: {response.status_code} (Sprawdź, czy klucz jest poprawny)"
        
    data = response.json()
    all_matches = data.get('matches', [])
    
    scheduled = [m for m in all_matches if m['status'] == 'SCHEDULED']
    
    if not scheduled:
        return [], "Brak zaplanowanych meczów w bazie API dla tego sezonu."
        
    next_md = min(m['matchday'] for m in scheduled)
    next_matches = [m for m in scheduled if m['matchday'] == next_md]
    
    formatted_matches = []
    for m in next_matches:
        formatted_matches.append({
            'Kolejka': next_md,
            'Data': m['utcDate'][:10],
            'Gospodarz': TEAM_MAPPING.get(m['homeTeam']['name'], m['homeTeam']['name']),
            'Gość': TEAM_MAPPING.get(m['awayTeam']['name'], m['awayTeam']['name'])
        })
        
    return formatted_matches, next_md

matches_data, md_info = get_next_matchday_matches(api_key)

# --- 3. WYŚWIETLANIE WYNIKÓW ---
if isinstance(matches_data, list) and matches_data:
    st.subheader(f"📅 Prognoza na Kolejkę #{md_info}")
    st.divider()
    
    results = []
    for m in matches_data:
        h, a = m['Gospodarz'], m['Gość']
        
        if h not in ratings or a not in ratings:
            continue
            
        h_xg = ratings[h]['attack'] * ratings[a]['defense'] * HOME_ADV * AVG_GOALS
        a_xg = ratings[a]['attack'] * ratings[h]['defense'] * (1 / HOME_ADV) * AVG_GOALS
        
        matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6): 
                matrix[i, j] = poisson.pmf(i, h_xg) * poisson.pmf(j, a_xg)
                
        win_h = np.tril(matrix, -1).sum() * 100
        draw = np.trace(matrix) * 100
        win_a = np.triu(matrix, 1).sum() * 100
        
        results.append({
            "Data": m['Data'],
            "Mecz": f"{h} vs {a}",
            "Wygra Gospodarz (1)": f"{win_h:.1f}%",
            "Remis (X)": f"{draw:.1f}%",
            "Wygra Gość (2)": f"{win_a:.1f}%",
            "xG Gospodarz": round(h_xg, 2),
            "xG Gość": round(a_xg, 2)
        })
        
    if results:
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True, hide_index=True)
    else:
        st.warning("Nie udało się dopasować drużyn do bazy danych modeli.")
else:
    st.error(f"Nie udało się pobrać danych: {md_info}")
