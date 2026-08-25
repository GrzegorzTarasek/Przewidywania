import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="PL Predictor Ultimate", page_icon="⚽", layout="wide")
st.title("⚽ Premier League Predictor (Ultimate - Bez Błędów API)")

# --- TRENOWANIE MODELU NA HISTORİI CSV ---
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
teams_list = sorted(list(ratings.keys()))

# --- UI: NAWIGACJA ---
st.sidebar.header("⚙️ Tryb Pracy")
mode = st.sidebar.radio("Wybierz widok:", ["🔍 Analiza Jednego Meczu", "🛠️ Generator Własnej Kolejki"])

# --- WIDOK 1: ANALIZA JEDNEGO MECZU ---
if mode == "🔍 Analiza Jednego Meczu":
    st.subheader("Głęboka analiza statystyczna konkretnego spotkania")
    c1, c2 = st.columns(2)
    h_team = c1.selectbox("Wybierz Gospodarza:", teams_list)
    a_team = c2.selectbox("Wybierz Gościa:", teams_list, index=1)
    
    if h_team != a_team:
        st.divider()
        h_xg = ratings[h_team]['attack'] * ratings[a_team]['defense'] * HOME_ADV * AVG_GOALS
        a_xg = ratings[a_team]['attack'] * ratings[h_team]['defense'] * (1 / HOME_ADV) * AVG_GOALS
        
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

# --- WIDOK 2: GENERATOR WŁASNEJ KOLEJKI ---
elif mode == "🛠️ Generator Własnej Kolejki":
    st.subheader("Stwórz własny zestaw meczów (np. nadchodząca kolejka)")
    st.markdown("Wybierz pary meczowe, które chcesz przeanalizować w formie zbiorczej tabeli.")
    
    # Pozwalamy użytkownikowi wybrać kilka meczów do symulacji
    selected_matches = []
    
    num_matches = st.number_input("Ile meczów chcesz dodać do zestawienia?", min_value=1, max_value=10, value=5)
    
    st.divider()
    
    for i in range(int(num_matches)):
        cols = st.columns(2)
        h = cols[0].selectbox(f"Gospodarz #{i+1}", teams_list, key=f"h_{i}")
        a = cols[1].selectbox(f"Gość #{i+1}", teams_list, key=f"a_{i}", index=(i+1)%len(teams_list))
        selected_matches.append((h, a))
        
    if st.button("🚀 Symuluj wybrane mecze"):
        results = []
        for h, a in selected_matches:
            if h == a: continue
            h_xg = ratings[h]['attack'] * ratings[a]['defense'] * HOME_ADV * AVG_GOALS
            a_xg = ratings[a]['attack'] * ratings[h]['defense'] * (1 / HOME_ADV) * AVG_GOALS
            
            matrix = np.zeros((6, 6))
            for i in range(6):
                for j in range(6): matrix[i, j] = poisson.pmf(i, h_xg) * poisson.pmf(j, a_xg)
                
            results.append({
                "Mecz": f"{h} vs {a}",
                "1": f"{np.tril(matrix, -1).sum()*100:.1f}%",
                "X": f"{np.trace(matrix)*100:.1f}%",
                "2": f"{np.triu(matrix, 1).sum()*100:.1f}%",
                "xG Gospodarz": round(h_xg, 2), "xG Gość": round(a_xg, 2)
            })
            
        st.divider()
        st.subheader("📊 Wyniki symulacji zestawienia")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
