import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="PL Predictor Ultimate", page_icon="⚽", layout="wide")
st.title("⚽ Premier League Predictor (Ultimate)")

# --- TŁUMACZ NAZW ---
TEAM_MAPPING = {
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa", "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford", "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton", "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich", "Leicester City FC": "Leicester", "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City", "Manchester United FC": "Man United", "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottm Forest", "Southampton FC": "Southampton", "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham", "Wolverhampton Wanderers FC": "Wolves"
}

st.sidebar.header("🔑 Autoryzacja")
api_key = st.sidebar.text_input("Klucz API (Football-Data):", type="password")

if not api_key:
    st.warning("👈 Wklej klucz API, aby załadować interfejs.")
    st.stop()

# --- TRENOWANIE MODELU ---
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

# --- POBIERANIE TERMINARZA ---
@st.cache_data(ttl=3600)
def get_all_matches(token):
    headers = {'X-Auth-Token': token}
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    response = requests.get(url, headers=headers)
    if response.status_code != 200: return []
    
    matches = []
    for m in response.json().get('matches', []):
        if m['status'] == 'SCHEDULED':
            matches.append({
                'Kolejka': m['matchday'],
                'Data': m['utcDate'][:10],
                'Gospodarz': TEAM_MAPPING.get(m['homeTeam']['name'], m['homeTeam']['name']),
                'Gość': TEAM_MAPPING.get(m['awayTeam']['name'], m['awayTeam']['name'])
            })
    return pd.DataFrame(matches)

df_matches = get_all_matches(api_key)

# --- UI: NAWIGACJA ---
st.sidebar.divider()
st.sidebar.header("⚙️ Tryb Pracy")
mode = st.sidebar.radio("Wybierz widok:", ["📅 Przegląd Kolejki", "🔍 Analiza Szczegółowa"])

# --- WIDOK 1: KOLEJKA ---
if mode == "📅 Przegląd Kolejki":
    st.subheader("Wybierz kolejkę do symulacji")
    if not df_matches.empty:
        available_matchdays = sorted(df_matches['Kolejka'].unique())
        selected_md = st.selectbox("Kolejka:", available_matchdays)
        
        md_matches = df_matches[df_matches['Kolejka'] == selected_md]
        results = []
        
        for _, m in md_matches.iterrows():
            h, a = m['Gospodarz'], m['Gość']
            if h not in ratings or a not in ratings: continue
                
            h_xg = ratings[h]['attack'] * ratings[a]['defense'] * HOME_ADV * AVG_GOALS
            a_xg = ratings[a]['attack'] * ratings[h]['defense'] * (1 / HOME_ADV) * AVG_GOALS
            
            matrix = np.zeros((6, 6))
            for i in range(6):
                for j in range(6): matrix[i, j] = poisson.pmf(i, h_xg) * poisson.pmf(j, a_xg)
                    
            results.append({
                "Data": m['Data'], "Mecz": f"{h} - {a}",
                "1": f"{np.tril(matrix, -1).sum()*100:.1f}%",
                "X": f"{np.trace(matrix)*100:.1f}%",
                "2": f"{np.triu(matrix, 1).sum()*100:.1f}%",
                "xG Gospodarz": round(h_xg, 2), "xG Gość": round(a_xg, 2)
            })
            
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("Brak dostępnych meczów (lub problem z limitem API).")

# --- WIDOK 2: SZCZEGÓŁY ---
elif mode == "🔍 Analiza Szczegółowa":
    st.subheader("Głęboka analiza statystyczna")
    teams = sorted(list(ratings.keys()))
    c1, c2 = st.columns(2)
    h_team = c1.selectbox("Wybierz Gospodarza:", teams)
    a_team = c2.selectbox("Wybierz Gościa:", teams, index=1)
    
    if h_team != a_team:
        st.divider()
        h_xg = ratings[h_team]['attack'] * ratings[a_team]['defense'] * HOME_ADV * AVG_GOALS
        a_xg = ratings[a_team]['attack'] * ratings[h_team]['defense'] * (1 / HOME_ADV) * AVG_GOALS
        
        # Wyświetlanie głębokich statystyk
        st.markdown("### ⚙️ Wskaźniki Modelu AI")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Siła Ataku ({h_team})", round(ratings[h_team]['attack'], 2))
        col2.metric(f"Szczelność Obrony ({h_team})", round(ratings[h_team]['defense'], 2))
        col3.metric(f"Siła Ataku ({a_team})", round(ratings[a_team]['attack'], 2))
        col4.metric(f"Szczelność Obrony ({a_team})", round(ratings[a_team]['defense'], 2))
        
        st.markdown("### 🔥 Wskaźnik Momentum (Forma)")
        cm1, cm2 = st.columns(2)
        cm1.metric(h_team, round(ratings[h_team]['trend'], 3))
        cm2.metric(a_team, round(ratings[a_team]['trend'], 3))
        
        st.markdown("### 🎯 Dokładny Wynik (TOP 5)")
        matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6): matrix[i, j] = poisson.pmf(i, h_xg) * poisson.pmf(j, a_xg)
                
        scores = [(f"{i}:{j}", matrix[i,j]) for i in range(6) for j in range(6)]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_5 = pd.DataFrame(scores[:5], columns=['Wynik', 'Szansa'])
        top_5['Szansa'] = (top_5['Szansa'] * 100).round(2).astype(str) + '%'
        
        st.table(top_5.set_index('Wynik'))
    else:
        st.error("Wybierz dwie różne drużyny!")
