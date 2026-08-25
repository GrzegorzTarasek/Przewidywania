import codecs
import json
import math
import re
from datetime import date, datetime
from difflib import SequenceMatcher, get_close_matches

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson


st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

LEAGUES = {
    "PL": {
        "label": "Premier League",
        "country": "Anglia",
        "csv_code": "E0",
        "understat": "EPL",
    },
    "PD": {
        "label": "LaLiga",
        "country": "Hiszpania",
        "csv_code": "SP1",
        "understat": "La_liga",
    },
    "SA": {
        "label": "Serie A",
        "country": "Włochy",
        "csv_code": "I1",
        "understat": "Serie_A",
    },
    "BL1": {
        "label": "Bundesliga",
        "country": "Niemcy",
        "csv_code": "D1",
        "understat": "Bundesliga",
    },
    "FL1": {
        "label": "Ligue 1",
        "country": "Francja",
        "csv_code": "F1",
        "understat": "Ligue_1",
    },
}

TEAM_ALIASES = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Leeds United FC": "Leeds",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottm Forest",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Real Madrid CF": "Real Madrid",
    "FC Barcelona": "Barcelona",
    "Club Atlético de Madrid": "Ath Madrid",
    "Atlético de Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Real Betis Balompié": "Betis",
    "Real Sociedad de Fútbol": "Sociedad",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "Getafe CF": "Getafe",
    "Rayo Vallecano de Madrid": "Vallecano",
    "RC Celta de Vigo": "Celta",
    "CA Osasuna": "Osasuna",
    "Deportivo Alavés": "Alaves",
    "RCD Mallorca": "Mallorca",
    "Girona FC": "Girona",
    "RCD Espanyol de Barcelona": "Espanol",
    "Elche CF": "Elche",
    "Levante UD": "Levante",
    "Real Oviedo": "Oviedo",
    "AC Milan": "Milan",
    "FC Internazionale Milano": "Inter",
    "Juventus FC": "Juventus",
    "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna",
    "Cagliari Calcio": "Cagliari",
    "Como 1907": "Como",
    "ACF Fiorentina": "Fiorentina",
    "Genoa CFC": "Genoa",
    "SS Lazio": "Lazio",
    "Parma Calcio 1913": "Parma",
    "AS Roma": "Roma",
    "SSC Napoli": "Napoli",
    "Torino FC": "Torino",
    "Udinese Calcio": "Udinese",
    "US Lecce": "Lecce",
    "US Sassuolo Calcio": "Sassuolo",
    "Hellas Verona FC": "Verona",
    "Pisa SC": "Pisa",
    "US Cremonese": "Cremonese",
    "FC Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "Borussia Mönchengladbach": "M'gladbach",
    "SC Freiburg": "Freiburg",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "1. FC Köln": "FC Koln",
    "1. FSV Mainz 05": "Mainz",
    "SV Werder Bremen": "Werder Bremen",
    "FC Augsburg": "Augsburg",
    "1. FC Union Berlin": "Union Berlin",
    "FC St. Pauli 1910": "St Pauli",
    "Hamburger SV": "Hamburg",
    "1. FC Heidenheim 1846": "Heidenheim",
    "Paris Saint-Germain FC": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco",
    "Lille OSC": "Lille",
    "Stade Rennais FC 1901": "Rennes",
    "OGC Nice": "Nice",
    "RC Lens": "Lens",
    "FC Nantes": "Nantes",
    "Toulouse FC": "Toulouse",
    "Montpellier HSC": "Montpellier",
    "Stade Brestois 29": "Brest",
    "RC Strasbourg Alsace": "Strasbourg",
    "Angers SCO": "Angers",
    "FC Metz": "Metz",
    "AJ Auxerre": "Auxerre",
    "Le Havre AC": "Le Havre",
    "FC Lorient": "Lorient",
}


def normalize_name(value):
    if value is None:
        return ""
    text = str(value).lower()
    replacements = {
        "&": "and",
        "é": "e",
        "è": "e",
        "ê": "e",
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "í": "i",
        "ó": "o",
        "ö": "o",
        "ø": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
        "ç": "c",
        "ß": "ss",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\b(fc|cf|afc|sc|ac|as|ssc|bc|ud|rcd|rc|calcio|club)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def team_display_name(team):
    if not isinstance(team, dict):
        return str(team)
    return team.get("shortName") or team.get("name") or team.get("tla") or "Nieznany zespół"


def team_model_name(team):
    if not isinstance(team, dict):
        return TEAM_ALIASES.get(str(team), str(team))
    return (
        TEAM_ALIASES.get(team.get("name"))
        or TEAM_ALIASES.get(team.get("shortName"))
        or team.get("shortName")
        or team.get("name")
        or team.get("tla")
        or "Nieznany zespół"
    )


def season_start_year(today=None):
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def football_data_season_codes(back=5, forward=0):
    current = season_start_year()
    codes = []
    for start in range(current - back + 1, current + forward + 1):
        codes.append(f"{str(start)[-2:]}{str(start + 1)[-2:]}")
    return codes


def safe_float(value, default=0.0):
    try:
        if value in (None, "") or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def request_json(url, token=None, params=None, extra_headers=None):
    headers = {"User-Agent": "football-predictor-streamlit/1.0"}
    if token:
        headers["X-Auth-Token"] = token
    if extra_headers:
        headers.update(extra_headers)
    response = requests.get(url, headers=headers, params=params or {}, timeout=25)
    if response.status_code == 429:
        raise RuntimeError("Limit zapytań API został przekroczony. Odczekaj chwilę i spróbuj ponownie.")
    if response.status_code == 403:
        raise RuntimeError("API zwróciło 403. Ten zasób może wymagać innego planu lub poprawnego klucza.")
    if response.status_code >= 400:
        raise RuntimeError(f"API zwróciło błąd {response.status_code}: {response.text[:220]}")
    return response.json()


@st.cache_data(ttl=900, show_spinner=False)
def fd_get(path, token, params=None, unfold=False):
    headers = {}
    if unfold:
        headers = {
            "X-Unfold-Goals": "true",
            "X-Unfold-Bookings": "true",
        }
    return request_json(f"https://api.football-data.org/v4/{path.lstrip('/')}", token, params, headers)


@st.cache_data(ttl=1800, show_spinner=False)
def get_league_bundle(league_code, token):
    standings = fd_get(f"competitions/{league_code}/standings", token)
    matches = fd_get(f"competitions/{league_code}/matches", token, unfold=True)
    scorers = fd_get(f"competitions/{league_code}/scorers", token, params={"limit": 100})
    return standings, matches, scorers


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_detail(team_id, token):
    return fd_get(f"teams/{team_id}", token)


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_matches(team_id, token, league_code):
    return fd_get(
        f"teams/{team_id}/matches",
        token,
        params={"status": "FINISHED", "competitions": league_code, "limit": 50},
        unfold=True,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def load_understat(understat_code, season):
    url = f"https://understat.com/league/{understat_code}/{season}"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    if response.status_code >= 400:
        return {}, []

    def extract_var(name, default):
        pattern = rf"var\s+{name}\s*=\s*JSON\.parse\('([^']*)'\)"
        match = re.search(pattern, response.text)
        if not match:
            return default
        decoded = codecs.decode(match.group(1), "unicode_escape")
        return json.loads(decoded)

    teams = extract_var("teamsData", {})
    players = extract_var("playersData", [])
    return teams, players


@st.cache_data(ttl=86400, show_spinner=False)
def load_history_csv(csv_code):
    frames = []
    for season in football_data_season_codes(back=5, forward=0):
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{csv_code}.csv"
        try:
            frame = pd.read_csv(url, on_bad_lines="skip")
        except Exception:
            continue
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if required.issubset(frame.columns):
            frame = frame[list(required)].dropna()
            frame["SeasonCode"] = season
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "SeasonCode"])
    data = pd.concat(frames, ignore_index=True)
    data["ParsedDate"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    return data.dropna(subset=["ParsedDate"]).sort_values("ParsedDate").reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def train_prediction_model(csv_code):
    history = load_history_csv(csv_code)
    if history.empty:
        return {}, 1.18, 1.35, pd.DataFrame()

    ratings = {}
    home_adv = 1.18
    avg_goals = max(1.05, float(history[["FTHG", "FTAG"]].mean().mean()))
    current_code = football_data_season_codes(back=1)[0]

    def init_team(name):
        if name not in ratings:
            ratings[name] = {
                "attack": 1.0,
                "defense": 1.0,
                "trend": 0.0,
                "matches": 0,
                "gf": 0,
                "ga": 0,
            }

    for _, row in history.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        init_team(home)
        init_team(away)

        season_weight = 1.0 if row["SeasonCode"] == current_code else 0.55
        alpha = 0.09 * season_weight

        home_xg = ratings[home]["attack"] * ratings[away]["defense"] * home_adv * avg_goals
        away_xg = ratings[away]["attack"] * ratings[home]["defense"] * (1 / home_adv) * avg_goals
        home_goals = safe_float(row["FTHG"])
        away_goals = safe_float(row["FTAG"])
        home_error = home_goals - home_xg
        away_error = away_goals - away_xg

        ratings[home]["attack"] = max(0.25, ratings[home]["attack"] + alpha * home_error * 0.08)
        ratings[away]["defense"] = max(0.35, ratings[away]["defense"] + alpha * home_error * 0.055)
        ratings[away]["attack"] = max(0.25, ratings[away]["attack"] + alpha * away_error * 0.08)
        ratings[home]["defense"] = max(0.35, ratings[home]["defense"] + alpha * away_error * 0.055)

        ratings[home]["trend"] = ratings[home]["trend"] * 0.82 + home_error * 0.18
        ratings[away]["trend"] = ratings[away]["trend"] * 0.82 + away_error * 0.18

        ratings[home]["matches"] += 1
        ratings[away]["matches"] += 1
        ratings[home]["gf"] += int(home_goals)
        ratings[home]["ga"] += int(away_goals)
        ratings[away]["gf"] += int(away_goals)
        ratings[away]["ga"] += int(home_goals)

    return ratings, home_adv, avg_goals, history


def find_rating_name(team_name, ratings):
    if not ratings:
        return None
    if team_name in ratings:
        return team_name
    normalized_target = normalize_name(team_name)
    normalized = {normalize_name(name): name for name in ratings.keys()}
    if normalized_target in normalized:
        return normalized[normalized_target]
    best = get_close_matches(normalized_target, list(normalized.keys()), n=1, cutoff=0.55)
    return normalized[best[0]] if best else None


def probability_matrix(home_xg, away_xg, max_goals=7):
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    matrix = np.outer(home_probs, away_probs)
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    most_likely = max(
        ((h, a, matrix[h][a]) for h in range(max_goals + 1) for a in range(max_goals + 1)),
        key=lambda item: item[2],
    )
    return home_win, draw, away_win, most_likely


def unavailable_penalty(names):
    count = len([name for name in names if name.strip()])
    return max(0.82, 1.0 - min(count, 8) * 0.025)


def predict_match(home_name, away_name, ratings, home_adv, avg_goals, unavailable_home=None, unavailable_away=None):
    unavailable_home = unavailable_home or []
    unavailable_away = unavailable_away or []
    home_rating_name = find_rating_name(home_name, ratings)
    away_rating_name = find_rating_name(away_name, ratings)
    if not home_rating_name or not away_rating_name:
        return None

    home = ratings[home_rating_name]
    away = ratings[away_rating_name]
    home_xg = home["attack"] * away["defense"] * home_adv * avg_goals
    away_xg = away["attack"] * home["defense"] * (1 / home_adv) * avg_goals
    home_xg = max(0.15, home_xg + home["trend"] * 0.08)
    away_xg = max(0.15, away_xg + away["trend"] * 0.08)
    home_xg *= unavailable_penalty(unavailable_home)
    away_xg *= unavailable_penalty(unavailable_away)
    home_win, draw, away_win, likely = probability_matrix(home_xg, away_xg)
    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "score": f"{likely[0]}:{likely[1]}",
        "confidence": max(home_win, draw, away_win),
        "home_rating_name": home_rating_name,
        "away_rating_name": away_rating_name,
    }


def split_matches_by_round(matches):
    all_matches = matches.get("matches", [])
    finished = [match for match in all_matches if match.get("status") == "FINISHED" and match.get("matchday")]
    future = [
        match
        for match in all_matches
        if match.get("status") in {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "LIVE"} and match.get("matchday")
    ]
    last_md = max((match["matchday"] for match in finished), default=None)
    next_md = min((match["matchday"] for match in future), default=None)
    last_matches = [match for match in finished if match.get("matchday") == last_md] if last_md else []
    next_matches = [match for match in future if match.get("matchday") == next_md] if next_md else []
    return last_md, last_matches, next_md, next_matches


def standings_dataframe(standings):
    rows = []
    for group in standings.get("standings", []):
        if group.get("type") != "TOTAL":
            continue
        for row in group.get("table", []):
            team = row.get("team", {})
            rows.append(
                {
                    "Poz": row.get("position"),
                    "TeamID": team.get("id"),
                    "Drużyna": team_display_name(team),
                    "M": row.get("playedGames"),
                    "W": row.get("won"),
                    "R": row.get("draw"),
                    "P": row.get("lost"),
                    "Bramki": f"{row.get('goalsFor', 0)}:{row.get('goalsAgainst', 0)}",
                    "+/-": row.get("goalDifference"),
                    "Pkt": row.get("points"),
                    "Forma": " ".join(row.get("form", "").split(",")) if row.get("form") else "",
                    "Crest": team.get("crest"),
                    "FullName": team.get("name"),
                }
            )
    return pd.DataFrame(rows)


def scorers_dataframe(scorers):
    rows = []
    for item in scorers.get("scorers", []):
        player = item.get("player", {})
        team = item.get("team", {})
        rows.append(
            {
                "Zawodnik": player.get("name"),
                "Pozycja": player.get("section") or player.get("position", ""),
                "Drużyna": team_display_name(team),
                "TeamID": team.get("id"),
                "Gole": item.get("goals", 0),
                "Asysty": item.get("assists", 0),
                "Karne": item.get("penalties", 0),
                "PlayerID": player.get("id"),
            }
        )
    return pd.DataFrame(rows)


def understat_players_dataframe(players):
    rows = []
    for item in players:
        rows.append(
            {
                "Zawodnik": item.get("player_name"),
                "Drużyna": item.get("team_title"),
                "M": int(safe_float(item.get("games"))),
                "Min": int(safe_float(item.get("time"))),
                "Gole": int(safe_float(item.get("goals"))),
                "xG": safe_float(item.get("xG")),
                "Asysty": int(safe_float(item.get("assists"))),
                "xA": safe_float(item.get("xA")),
                "Strzały": int(safe_float(item.get("shots"))),
                "Żółte": int(safe_float(item.get("yellow_cards"))),
                "Czerwone": int(safe_float(item.get("red_cards"))),
                "Pozycja": item.get("position", ""),
            }
        )
    return pd.DataFrame(rows)


def select_team(team_id, team_name):
    st.session_state.selected_team_id = int(team_id)
    st.session_state.selected_team_name = team_name


def render_match(match, ratings, home_adv, avg_goals, unavailable):
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})
    home_name = team_display_name(home)
    away_name = team_display_name(away)
    home_model = team_model_name(home)
    away_model = team_model_name(away)
    score = match.get("score", {}).get("fullTime", {})
    dt = match.get("utcDate", "")[:16].replace("T", " ")

    cols = st.columns([1.5, 0.32, 1.5, 1.1])
    with cols[0]:
        if st.button(home_name, key=f"home-{match.get('id')}", use_container_width=True):
            select_team(home.get("id"), home_name)
            st.rerun()
    with cols[1]:
        if match.get("status") == "FINISHED":
            st.markdown(f"**{score.get('home', '-')}-{score.get('away', '-')}**")
        else:
            st.markdown("vs")
    with cols[2]:
        if st.button(away_name, key=f"away-{match.get('id')}", use_container_width=True):
            select_team(away.get("id"), away_name)
            st.rerun()
    with cols[3]:
        if match.get("status") == "FINISHED":
            st.caption(dt)
        else:
            prediction = predict_match(
                home_model,
                away_model,
                ratings,
                home_adv,
                avg_goals,
                unavailable.get(str(home.get("id")), []),
                unavailable.get(str(away.get("id")), []),
            )
            if prediction:
                st.caption(
                    f"{prediction['score']} | "
                    f"1 {prediction['home_win']:.0%} / X {prediction['draw']:.0%} / 2 {prediction['away_win']:.0%}"
                )
            else:
                st.caption(dt)


def render_standings(standings_df):
    header = st.columns([0.4, 2.0, 0.45, 0.45, 0.45, 0.45, 0.8, 0.55, 0.55])
    labels = ["#", "Drużyna", "M", "W", "R", "P", "B", "+/-", "Pkt"]
    for col, label in zip(header, labels):
        col.markdown(f"**{label}**")

    for _, row in standings_df.iterrows():
        cols = st.columns([0.4, 2.0, 0.45, 0.45, 0.45, 0.45, 0.8, 0.55, 0.55])
        cols[0].write(row["Poz"])
        with cols[1]:
            if st.button(row["Drużyna"], key=f"table-team-{row['TeamID']}", use_container_width=True):
                select_team(row["TeamID"], row["Drużyna"])
                st.rerun()
        cols[2].write(row["M"])
        cols[3].write(row["W"])
        cols[4].write(row["R"])
        cols[5].write(row["P"])
        cols[6].write(row["Bramki"])
        cols[7].write(row["+/-"])
        cols[8].write(row["Pkt"])


def team_understat_filter(df, team_name):
    if df.empty:
        return df
    names = sorted(df["Drużyna"].dropna().unique())
    target = normalize_name(team_name)
    scored = [
        (name, SequenceMatcher(None, target, normalize_name(name)).ratio())
        for name in names
    ]
    best_name, best_score = max(scored, key=lambda item: item[1], default=(None, 0))
    if best_name and best_score >= 0.45:
        return df[df["Drużyna"] == best_name].copy()
    return df[df["Drużyna"].map(lambda value: target in normalize_name(value) or normalize_name(value) in target)].copy()


def render_team_page(team_id, team_name, token, league_code, standings_df, scorers_df, understat_df, ratings):
    detail = get_team_detail(team_id, token)
    st.divider()
    top = st.columns([0.8, 3, 1.2])
    crest = detail.get("crest")
    if crest:
        top[0].image(crest, width=96)
    top[1].subheader(detail.get("name", team_name))
    top[1].caption(
        f"{detail.get('venue', 'brak stadionu')} | trener: "
        f"{detail.get('coach', {}).get('name', 'brak danych')}"
    )
    with top[2]:
        if st.button("Zamknij widok drużyny", use_container_width=True):
            st.session_state.selected_team_id = None
            st.session_state.selected_team_name = None
            st.rerun()

    team_row = standings_df[standings_df["TeamID"] == team_id]
    if not team_row.empty:
        row = team_row.iloc[0]
        metrics = st.columns(5)
        metrics[0].metric("Pozycja", int(row["Poz"]))
        metrics[1].metric("Punkty", int(row["Pkt"]))
        metrics[2].metric("Bilans", row["Bramki"])
        metrics[3].metric("Różnica", int(row["+/-"]))
        metrics[4].metric("Forma", row["Forma"] or "-")

    squad = pd.DataFrame(detail.get("squad", []))
    if not squad.empty:
        squad = squad.rename(
            columns={
                "name": "Zawodnik",
                "position": "Pozycja",
                "dateOfBirth": "Data ur.",
                "nationality": "Narodowość",
            }
        )
        visible_cols = [col for col in ["Zawodnik", "Pozycja", "Data ur.", "Narodowość"] if col in squad.columns]
        squad = squad[visible_cols]

    team_scorers = scorers_df[scorers_df["TeamID"] == team_id].copy() if not scorers_df.empty else pd.DataFrame()
    team_understat = team_understat_filter(understat_df, team_name)

    tabs = st.tabs(["Skład", "Statystyki zawodników", "Mecze i forma", "Model"])
    with tabs[0]:
        if squad.empty:
            st.info("Football-Data nie zwróciło składu dla tego zespołu w Twoim planie API.")
        else:
            st.dataframe(squad, use_container_width=True, hide_index=True)

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.markdown("**Gole i asysty z Football-Data**")
            if team_scorers.empty:
                st.info("Brak danych strzelców/asystentów dla tej drużyny w odpowiedzi API.")
            else:
                st.dataframe(
                    team_scorers[["Zawodnik", "Pozycja", "Gole", "Asysty", "Karne"]].sort_values(
                        ["Gole", "Asysty"], ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        with right:
            st.markdown("**xG, xA i kartki z Understat**")
            if team_understat.empty:
                st.info("Nie udało się dopasować tej drużyny do danych Understat.")
            else:
                st.dataframe(
                    team_understat.sort_values(["Min", "xG"], ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[2]:
        try:
            team_matches = get_team_matches(team_id, token, league_code).get("matches", [])
        except Exception as exc:
            team_matches = []
            st.warning(str(exc))
        if not team_matches:
            st.info("Brak ostatnich meczów dla tej drużyny.")
        else:
            form_rows = []
            for match in team_matches[-12:]:
                home = match.get("homeTeam", {})
                away = match.get("awayTeam", {})
                score = match.get("score", {}).get("fullTime", {})
                is_home = home.get("id") == team_id
                gf = score.get("home", 0) if is_home else score.get("away", 0)
                ga = score.get("away", 0) if is_home else score.get("home", 0)
                result = "W" if gf > ga else "R" if gf == ga else "P"
                form_rows.append(
                    {
                        "Data": match.get("utcDate", "")[:10],
                        "Rywal": team_display_name(away if is_home else home),
                        "Dom/Wyjazd": "Dom" if is_home else "Wyjazd",
                        "Wynik": f"{gf}:{ga}",
                        "Rezultat": result,
                    }
                )
            st.dataframe(pd.DataFrame(form_rows), use_container_width=True, hide_index=True)

    with tabs[3]:
        model_name = find_rating_name(team_model_name(detail), ratings) or find_rating_name(team_name, ratings)
        if not model_name:
            st.info("Model nie znalazł odpowiednika tej drużyny w historycznych plikach CSV.")
        else:
            rating = ratings[model_name]
            cols = st.columns(5)
            cols[0].metric("Nazwa w modelu", model_name)
            cols[1].metric("Siła ataku", f"{rating['attack']:.3f}")
            cols[2].metric("Siła obrony", f"{rating['defense']:.3f}")
            cols[3].metric("Trend", f"{rating['trend']:+.3f}")
            cols[4].metric("Mecze treningowe", int(rating["matches"]))
            st.caption(
                "Model uczy się z kilku poprzednich sezonów, ale mocniej waży bieżący sezon. "
                "Niedostępni zawodnicy wpisani w panelu bocznym obniżają oczekiwane gole drużyny."
            )


def init_state():
    st.session_state.setdefault("selected_team_id", None)
    st.session_state.setdefault("selected_team_name", None)
    st.session_state.setdefault("unavailable", {})


def main():
    init_state()
    st.title("Football Predictor")
    st.caption("Top 5 lig, tabela, terminarz, składy, statystyki i model predykcyjny aktualizowany bieżącym sezonem.")

    with st.sidebar:
        st.header("Ustawienia")
        token = st.text_input("Klucz API Football-Data", type="password")
        league_label_to_code = {f"{meta['label']} ({meta['country']})": code for code, meta in LEAGUES.items()}
        selected_label = st.selectbox("Liga", list(league_label_to_code.keys()))
        league_code = league_label_to_code[selected_label]
        current_season = season_start_year()
        st.caption(f"Sezon bazowy: {current_season}/{str(current_season + 1)[-2:]}")
        st.divider()
        st.markdown("**Kontuzje / zawieszenia**")
        st.caption("Wpisuj po jednym zawodniku w linii. Model traktuje ich jako niedostępnych.")

    if not token:
        st.warning("Wklej klucz API Football-Data w panelu po lewej stronie, żeby pobrać tabele i mecze.")
        st.stop()

    league_meta = LEAGUES[league_code]
    try:
        with st.spinner("Pobieram dane ligi..."):
            standings, matches, scorers = get_league_bundle(league_code, token.strip())
            understat_teams, understat_players = load_understat(league_meta["understat"], season_start_year())
            ratings, home_adv, avg_goals, history = train_prediction_model(league_meta["csv_code"])
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    standings_df = standings_dataframe(standings)
    scorers_df = scorers_dataframe(scorers)
    understat_df = understat_players_dataframe(understat_players)
    last_md, last_matches, next_md, next_matches = split_matches_by_round(matches)

    with st.sidebar:
        if not standings_df.empty:
            team_options = {row["Drużyna"]: str(row["TeamID"]) for _, row in standings_df.iterrows()}
            picked_team = st.selectbox("Drużyna do oznaczenia braków", list(team_options.keys()))
            unavailable_text = st.text_area(
                "Niedostępni",
                value="\n".join(st.session_state.unavailable.get(team_options[picked_team], [])),
                height=110,
                key=f"unavailable-{team_options[picked_team]}",
            )
            st.session_state.unavailable[team_options[picked_team]] = [
                line.strip() for line in unavailable_text.splitlines() if line.strip()
            ]
        st.divider()
        st.markdown("**Źródła danych**")
        st.caption("Football-Data: mecze, tabela, składy i strzelcy. Understat: xG, xA i kartki zawodników. football-data.co.uk: historia do modelu.")
        st.caption("Transfermarkt traktuj jako ręczne źródło do weryfikacji kontuzji, bo nie ma oficjalnego darmowego API.")

    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.subheader(f"{league_meta['label']} - kolejki")
        st.markdown(f"**Ostatnia kolejka: {last_md or 'brak'}**")
        if last_matches:
            for match in last_matches:
                render_match(match, ratings, home_adv, avg_goals, st.session_state.unavailable)
        else:
            st.info("Brak zakończonej kolejki w danych API.")

        st.divider()
        st.markdown(f"**Następna kolejka: {next_md or 'brak'}**")
        if next_matches:
            for match in next_matches:
                render_match(match, ratings, home_adv, avg_goals, st.session_state.unavailable)
        else:
            st.info("Brak nadchodzącej kolejki w danych API.")

    with right:
        st.subheader("Tabela")
        if standings_df.empty:
            st.info("Brak tabeli w danych API.")
        else:
            render_standings(standings_df)

    if st.session_state.selected_team_id:
        render_team_page(
            st.session_state.selected_team_id,
            st.session_state.selected_team_name,
            token.strip(),
            league_code,
            standings_df,
            scorers_df,
            understat_df,
            ratings,
        )

    with st.expander("Statystyki ligi i model"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Najlepsi strzelcy/asystenci**")
            if scorers_df.empty:
                st.info("Brak danych strzelców.")
            else:
                st.dataframe(
                    scorers_df[["Zawodnik", "Drużyna", "Gole", "Asysty", "Karne"]].sort_values(
                        ["Gole", "Asysty"], ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        with col2:
            st.markdown("**Najmocniejsze drużyny wg modelu**")
            model_rows = []
            for name, rating in ratings.items():
                model_rows.append(
                    {
                        "Drużyna": name,
                        "Atak": rating["attack"],
                        "Obrona": rating["defense"],
                        "Trend": rating["trend"],
                        "Wynik modelu": rating["attack"] / max(rating["defense"], 0.1) + rating["trend"] * 0.08,
                    }
                )
            if model_rows:
                st.dataframe(
                    pd.DataFrame(model_rows).sort_values("Wynik modelu", ascending=False).head(20),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Brak historii do trenowania modelu.")


if __name__ == "__main__":
    main()
