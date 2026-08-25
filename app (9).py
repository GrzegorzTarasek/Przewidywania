import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="PL Predictor - Live API", page_icon="📡")

st.title("📡 Premier League Predictor (Live API)")
st.markdown("""
Ta wersja aplikacji pobiera **najnowsze statystyki w czasie rzeczywistym** używając profesjonalnego interfejsu Football-Data.org.
""")

# --- POBIERANIE DANYCH Z API ---
@st.cache_data(ttl=3600)  # Aktualizuj dane co godzinę (3600 sekund), aby nie wyczerpać darmowego limitu!
def get_live_data():
    # Pobieranie klucza API z bezpiecznego miejsca (Streamlit Secrets)
    try:
        api_key = st.secrets["FOOTBALL_API_KEY"]
    except KeyError:
        st.error("Błąd: Nie znaleziono klucza API w ustawieniach Streamlit Secrets!")
        st.stop()

    headers = {'X-Auth-Token': api_key}
    
    # ID ligi: PL to kod dla Premier League. Zwraca wszystkie rozegrane mecze z obecnego sezonu
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        st.error(f"Błąd API: {response.status_code} - Sprawdź klucz API lub darmowe limity zapytań.")
        st.stop()
        
    data = response.json()
    
    # Ekstrakcja danych JSON do formy tabelarycznej
    matches = []
    for match in data.get('matches', []):
        matches.append({
            'HomeTeam': match['homeTeam']['name'],
            'AwayTeam': match['awayTeam']['name'],
            'FTHG': match['score']['fullTime']['home'], # Gole gospodarza
            'FTAG': match['score']['fullTime']['away']  # Gole gościa
        })
        
    df_raw = pd.DataFrame(matches)
    
    if df_raw.empty:
         st.warning("API nie zwróciło żadnych zakończonych meczów. Trwa przerwa między sezonami?")
         st.stop()
    
    # Przetwarzanie danych
    home_stats = df_raw.groupby('HomeTeam').agg(
        Mecze_Dom=('FTHG', 'count'),
        Gole_Strzelone_Dom=('FTHG', 'sum'),
        Gole_Stracone_Dom=('FTAG', 'sum')
    ).reset_index().rename(columns={'HomeTeam': 'Druzyna'})
    
    away_stats = df_raw.groupby('AwayTeam').agg(
        Mecze_Wyjazd=('FTAG', 'count'),
        Gole_Strzelone_Wyjazd=('FTAG', 'sum'),
        Gole_Stracone_Wyjazd=('FTHG', 'sum')
    ).reset_index().rename(columns={'AwayTeam': 'Druzyna'})
    
    df = pd.merge(home_stats, away_stats, on='Druzyna', how='outer').fillna(0)
    return df

# Pobranie danych przy uruchomieniu aplikacji
df = get_live_data()

# --- ŚREDNIE LIGOWE ---
avg_home_scored = df['Gole_Strzelone_Dom'].sum() / max(1, df['Mecze_Dom'].sum())
avg_away_scored = df['Gole_Strzelone_Wyjazd'].sum() / max(1, df['Mecze_Wyjazd'].sum())

# --- FUNKCJE MODELU ---
def get_team_stats(team):
    return df[df['Druzyna'] == team].iloc[0]

def calculate_match_xg(home_team, away_team):
    home_stats = get_team_stats(home_team)
    away_stats = get_team_stats(away_team)
    
    # Zabezpieczenie przed dzieleniem przez zero na początku sezonu
    home_matches = max(1, home_stats['Mecze_Dom'])
    away_matches = max(1, away_stats['Mecze_Wyjazd'])
    
    home_attack = (home_stats['Gole_Strzelone_Dom'] / home_matches) / avg_home_scored
    away_defense = (away_stats['Gole_Stracone_Wyjazd'] / away_matches) / avg_home_scored
    
    away_attack = (away_stats['Gole_Strzelone_Wyjazd'] / away_matches) / avg_away_scored
    home_defense = (home_stats['Gole_Stracone_Dom'] / home_matches) / avg_away_scored
    
    home_xg = home_attack * away_defense * avg_home_scored
    away_xg = away_attack * home_defense * avg_away_scored
    
    return home_xg, away_xg

def generate_poisson_matrix(home_xg, away_xg, max_goals=6):
    score_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
    return score_matrix

# --- INTERFEJS UŻYTKOWNIKA ---
st.sidebar.header("Wybierz mecz")
teams_list = df['Druzyna'].sort_values().tolist()
home_team = st.sidebar.selectbox("Gospodarz (Home)", teams_list)
away_team = st.sidebar.selectbox("Gość (Away)", teams_list, index=1)

if home_team == away_team:
    st.sidebar.error("Wybierz różne drużyny!")
else:
    home_xg, away_xg = calculate_match_xg(home_team, away_team)
    prob_matrix = generate_poisson_matrix(home_xg, away_xg)
    
    home_win_prob = np.tril(prob_matrix, -1).sum()
    draw_prob = np.trace(prob_matrix)
    away_win_prob = np.triu(prob_matrix, 1).sum()

    st.write("---")
    
    st.subheader(f"Statystyki LIVE ({int(df['Mecze_Dom'].sum() + df['Mecze_Wyjazd'].sum())} zakończonych meczów ligowych)")
    st.write(f"**{home_team}** dom: {int(get_team_stats(home_team)['Gole_Strzelone_Dom'])} zdob. / {int(get_team_stats(home_team)['Gole_Stracone_Dom'])} strac.")
    st.write(f"**{away_team}** wyjazd: {int(get_team_stats(away_team)['Gole_Strzelone_Wyjazd'])} zdob. / {int(get_team_stats(away_team)['Gole_Stracone_Wyjazd'])} strac.")

    st.subheader("🎯 Wyliczone Oczekiwane Gole (xG)")
    col1, col2 = st.columns(2)
    col1.metric(f"xG - {home_team}", round(home_xg, 2))
    col2.metric(f"xG - {away_team}", round(away_xg, 2))
    
    st.write("---")
    
    st.subheader("📊 Prawdopodobieństwo wyniku")
    c1, c2, c3 = st.columns(3)
    c1.info(f"Wygra {home_team}\n\n### {home_win_prob*100:.1f}%")
    c2.warning(f"Remis\n\n### {draw_prob*100:.1f}%")
    c3.success(f"Wygra {away_team}\n\n### {away_win_prob*100:.1f}%")
    
    st.write("---")
    st.subheader("⚽ TOP 5 Dokładnych Wyników")
    
    scores = [(f"{i}:{j}", prob_matrix[i,j]) for i in range(6) for j in range(6)]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    top_5 = pd.DataFrame(scores[:5], columns=['Wynik', 'Szansa'])
    top_5['Szansa'] = (top_5['Szansa'] * 100).round(2).astype(str) + '%'
    
    st.table(top_5.set_index('Wynik'))
