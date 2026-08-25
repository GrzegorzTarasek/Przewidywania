import codecs
import json
import re
from datetime import date
from difflib import SequenceMatcher, get_close_matches
from io import StringIO

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
        "clubelo_country": "ENG",
    },
    "PD": {
        "label": "LaLiga",
        "country": "Hiszpania",
        "csv_code": "SP1",
        "understat": "La_Liga",
        "clubelo_country": "ESP",
    },
    "SA": {
        "label": "Serie A",
        "country": "Włochy",
        "csv_code": "I1",
        "understat": "Serie_A",
        "clubelo_country": "ITA",
    },
    "BL1": {
        "label": "Bundesliga",
        "country": "Niemcy",
        "csv_code": "D1",
        "understat": "Bundesliga",
        "clubelo_country": "GER",
    },
    "FL1": {
        "label": "Ligue 1",
        "country": "Francja",
        "csv_code": "F1",
        "understat": "Ligue_1",
        "clubelo_country": "FRA",
    },
}

FPL_STATUS = {
    "a": "Dostępny",
    "d": "Wątpliwy",
    "i": "Kontuzja",
    "s": "Zawieszony",
    "u": "Niedostępny",
}

FPL_POSITION = {
    1: "Bramkarz",
    2: "Obrońca",
    3: "Pomocnik",
    4: "Napastnik",
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


def request_csv(url, params=None):
    response = requests.get(
        url,
        params=params or {},
        headers={"User-Agent": "football-predictor-streamlit/1.0"},
        timeout=25,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Źródło CSV zwróciło błąd {response.status_code}")
    return pd.read_csv(StringIO(response.text))


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


@st.cache_data(ttl=1800, show_spinner=False)
def get_match_detail(match_id, token):
    return fd_get(f"matches/{match_id}", token, unfold=True)


@st.cache_data(ttl=1800, show_spinner=False)
def load_fpl_bootstrap():
    try:
        return request_json("https://fantasy.premierleague.com/api/bootstrap-static/")
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def load_fpl_fixtures():
    try:
        return request_json("https://fantasy.premierleague.com/api/fixtures/")
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def load_clubelo_snapshot(country_code):
    today = date.today()
    for days_back in range(0, 45):
        day = today - pd.Timedelta(days=days_back)
        stamp = day.strftime("%Y-%m-%d")
        for base_url in ("https://api.clubelo.com", "http://api.clubelo.com"):
            try:
                frame = request_csv(f"{base_url}/{stamp}")
            except Exception:
                continue
            if frame.empty:
                continue
            country_column = next((col for col in ["Country", "country"] if col in frame.columns), None)
            if country_column:
                filtered = frame[frame[country_column] == country_code].copy()
                if not filtered.empty:
                    frame = filtered
            if "Level" in frame.columns:
                top_level = frame[pd.to_numeric(frame["Level"], errors="coerce").fillna(99) <= 1].copy()
                if not top_level.empty:
                    frame = top_level
            frame["SnapshotDate"] = stamp
            return frame.reset_index(drop=True)
    return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def load_understat(understat_code, season):
    league_codes = [understat_code]
    if understat_code == "La_Liga":
        league_codes.append("La_liga")
    elif understat_code == "La_liga":
        league_codes.append("La_Liga")

    seasons = list(dict.fromkeys([season, season - 1, season - 2, season - 3]))

    for league_code in league_codes:
        for candidate_season in seasons:
            url = f"https://understat.com/league/{league_code}/{candidate_season}"
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            except Exception:
                continue
            if response.status_code >= 400:
                continue
            page_text = response.text

            def extract_var(name, default):
                pattern = rf"var\s+{name}\s*=\s*JSON\.parse\('([^']*)'\)"
                match = re.search(pattern, page_text)
                if not match:
                    return default
                decoded = codecs.decode(match.group(1), "unicode_escape")
                return json.loads(decoded)

            teams = extract_var("teamsData", {})
            players = extract_var("playersData", [])
            if teams or players:
                return teams, players, candidate_season
    return {}, [], None


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


def predict_match(
    home_name,
    away_name,
    ratings,
    home_adv,
    avg_goals,
    unavailable_home=None,
    unavailable_away=None,
    clubelo_df=None,
):
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

    home_elo = clubelo_rating_for_team(home_name, clubelo_df)
    away_elo = clubelo_rating_for_team(away_name, clubelo_df)
    elo_diff = None
    if home_elo and away_elo:
        elo_diff = (home_elo["elo"] - away_elo["elo"]) + 65
        home_factor = float(np.clip(np.exp(elo_diff / 1200), 0.84, 1.18))
        away_factor = float(np.clip(np.exp(-elo_diff / 1200), 0.84, 1.18))
        home_xg *= home_factor
        away_xg *= away_factor

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
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
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


def fpl_players_dataframe(bootstrap):
    elements = bootstrap.get("elements", []) if isinstance(bootstrap, dict) else []
    teams = bootstrap.get("teams", []) if isinstance(bootstrap, dict) else []
    team_map = {team.get("id"): team for team in teams}
    rows = []
    for item in elements:
        team = team_map.get(item.get("team"), {})
        chance_this = item.get("chance_of_playing_this_round")
        chance_next = item.get("chance_of_playing_next_round")
        rows.append(
            {
                "Zawodnik": f"{item.get('first_name', '')} {item.get('second_name', '')}".strip()
                or item.get("web_name"),
                "Skrót": item.get("web_name"),
                "Drużyna": team.get("name") or team.get("short_name"),
                "TeamShort": team.get("short_name"),
                "Pozycja": FPL_POSITION.get(item.get("element_type"), ""),
                "Min": int(safe_float(item.get("minutes"))),
                "Gole": int(safe_float(item.get("goals_scored"))),
                "Asysty": int(safe_float(item.get("assists"))),
                "xG": safe_float(item.get("expected_goals")),
                "xA": safe_float(item.get("expected_assists")),
                "Forma": safe_float(item.get("form")),
                "PPM": safe_float(item.get("points_per_game")),
                "Punkty": int(safe_float(item.get("total_points"))),
                "Status": FPL_STATUS.get(item.get("status"), item.get("status", "")),
                "Kod statusu": item.get("status"),
                "Szansa teraz": chance_this if chance_this is not None else "",
                "Szansa następna": chance_next if chance_next is not None else "",
                "News": item.get("news") or "",
            }
        )
    return pd.DataFrame(rows)


def match_by_normalized_name(target_name, candidates):
    if not target_name or not candidates:
        return None, 0.0
    target = normalize_name(target_name)
    scored = [
        (candidate, SequenceMatcher(None, target, normalize_name(candidate)).ratio())
        for candidate in candidates
        if candidate
    ]
    return max(scored, key=lambda item: item[1], default=(None, 0.0))


def fpl_team_filter(fpl_df, team_name):
    if fpl_df.empty or "Drużyna" not in fpl_df.columns:
        return pd.DataFrame()
    candidates = sorted(fpl_df["Drużyna"].dropna().unique())
    best_name, best_score = match_by_normalized_name(team_name, candidates)
    if best_name and best_score >= 0.55:
        return fpl_df[fpl_df["Drużyna"] == best_name].copy()
    return pd.DataFrame()


def team_unavailable_from_fpl(fpl_team_df):
    if fpl_team_df.empty:
        return []
    missing = []
    for _, player in fpl_team_df.iterrows():
        chance_this = player.get("Szansa teraz", "")
        chance_next = player.get("Szansa następna", "")
        chance_values = [
            safe_float(value, 100.0)
            for value in [chance_this, chance_next]
            if value not in ("", None)
        ]
        has_low_chance = any(value < 75 for value in chance_values)
        has_bad_status = player.get("Kod statusu") in {"d", "i", "s", "u"}
        if has_bad_status or has_low_chance:
            label = player.get("Skrót") or player.get("Zawodnik")
            news = player.get("News")
            missing.append(f"{label} ({news})" if news else str(label))
    return missing


def build_auto_unavailable(standings_df, fpl_df):
    result = {}
    if standings_df.empty or fpl_df.empty:
        return result
    for _, row in standings_df.iterrows():
        team_id = str(row["TeamID"])
        fpl_team = fpl_team_filter(fpl_df, row["Drużyna"])
        unavailable = team_unavailable_from_fpl(fpl_team)
        if unavailable:
            result[team_id] = unavailable
    return result


def merge_unavailable(manual, automatic, include_automatic=True):
    merged = {}
    keys = set(manual.keys()) | (set(automatic.keys()) if include_automatic else set())
    for key in keys:
        values = list(manual.get(key, []))
        if include_automatic:
            values.extend(automatic.get(key, []))
        cleaned = []
        seen = set()
        for value in values:
            normalized = normalize_name(value)
            if normalized and normalized not in seen:
                cleaned.append(value)
                seen.add(normalized)
        merged[key] = cleaned
    return merged


def clubelo_name_column(clubelo_df):
    for column in ["Club", "club", "apiName", "displayName", "Name"]:
        if column in clubelo_df.columns:
            return column
    return None


def clubelo_rating_for_team(team_name, clubelo_df):
    if clubelo_df is None or clubelo_df.empty or "Elo" not in clubelo_df.columns:
        return None
    name_column = clubelo_name_column(clubelo_df)
    if not name_column:
        return None
    candidates = sorted(clubelo_df[name_column].dropna().astype(str).unique())
    best_name, best_score = match_by_normalized_name(team_name, candidates)
    if not best_name or best_score < 0.5:
        return None
    row = clubelo_df[clubelo_df[name_column].astype(str) == best_name].head(1)
    if row.empty:
        return None
    return {
        "name": best_name,
        "elo": safe_float(row.iloc[0].get("Elo")),
        "rank": int(safe_float(row.iloc[0].get("Rank"), 0)),
        "date": row.iloc[0].get("SnapshotDate", ""),
    }


def source_status(label, available, detail, required=False):
    return {
        "Źródło": label,
        "Status": "OK" if available else ("Brak danych" if required else "Pominięte"),
        "Opis": detail,
    }


def render_theme():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            background: rgba(250, 250, 250, 0.72);
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
        }
        div[data-testid="stMetricValue"] { font-size: 1.35rem; }
        div.stButton > button {
            border-radius: 7px;
            min-height: 2.35rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def select_team(team_id, team_name):
    st.session_state.selected_team_id = int(team_id)
    st.session_state.selected_team_name = team_name


def select_match(match_id, match_name):
    st.session_state.selected_match_id = int(match_id)
    st.session_state.selected_match_name = match_name


def close_match_view():
    st.session_state.selected_match_id = None
    st.session_state.selected_match_name = None


def close_team_view():
    st.session_state.selected_team_id = None
    st.session_state.selected_team_name = None


def display_minute(event):
    minute = event.get("minute")
    injury_time = event.get("injuryTime")
    if minute is None:
        return "-"
    if injury_time:
        return f"{minute}+{injury_time}'"
    return f"{minute}'"


def event_person_name(value):
    if isinstance(value, dict):
        return value.get("name") or value.get("firstName") or value.get("lastName") or "-"
    if value:
        return str(value)
    return "-"


def event_team_name(value):
    if isinstance(value, dict):
        return team_display_name(value)
    if value:
        return str(value)
    return "-"


def score_text(match):
    score = match.get("score", {}).get("fullTime", {})
    home_score = score.get("home")
    away_score = score.get("away")
    if home_score is None or away_score is None:
        return "vs"
    return f"{home_score}:{away_score}"


def match_title(match):
    home = team_display_name(match.get("homeTeam", {}))
    away = team_display_name(match.get("awayTeam", {}))
    return f"{home} - {away}"


def goals_dataframe(match):
    rows = []
    for goal in match.get("goals", []) or []:
        rows.append(
            {
                "Min": display_minute(goal),
                "Drużyna": event_team_name(goal.get("team")),
                "Strzelec": event_person_name(goal.get("scorer") or goal.get("player")),
                "Asysta": event_person_name(goal.get("assist")),
                "Typ": goal.get("type", ""),
                "Wynik": goal.get("score", ""),
            }
        )
    return pd.DataFrame(rows)


def bookings_dataframe(match):
    rows = []
    for booking in match.get("bookings", []) or []:
        rows.append(
            {
                "Min": display_minute(booking),
                "Drużyna": event_team_name(booking.get("team")),
                "Zawodnik": event_person_name(booking.get("player")),
                "Kartka": booking.get("card", booking.get("type", "")),
            }
        )
    return pd.DataFrame(rows)


def substitutions_dataframe(match):
    rows = []
    for substitution in match.get("substitutions", []) or []:
        rows.append(
            {
                "Min": display_minute(substitution),
                "Drużyna": event_team_name(substitution.get("team")),
                "Schodzi": event_person_name(substitution.get("playerOut") or substitution.get("out")),
                "Wchodzi": event_person_name(substitution.get("playerIn") or substitution.get("in")),
            }
        )
    return pd.DataFrame(rows)


def flatten_statistics(stats):
    if not stats:
        return {}
    if isinstance(stats, dict):
        return stats
    if isinstance(stats, list):
        flat = {}
        for item in stats:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("type") or item.get("key")
            value = item.get("value") or item.get("displayValue")
            if name:
                flat[name] = value
        return flat
    return {}


def match_statistics_dataframe(match):
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})
    raw_home = flatten_statistics(home.get("statistics"))
    raw_away = flatten_statistics(away.get("statistics"))

    if not raw_home and not raw_away:
        raw = match.get("statistics", {})
        if isinstance(raw, dict):
            raw_home = flatten_statistics(raw.get("home") or raw.get("homeTeam"))
            raw_away = flatten_statistics(raw.get("away") or raw.get("awayTeam"))

    rows = []
    for key in sorted(set(raw_home.keys()) | set(raw_away.keys())):
        rows.append(
            {
                "Statystyka": key,
                team_display_name(home): raw_home.get(key, "-"),
                team_display_name(away): raw_away.get(key, "-"),
            }
        )
    return pd.DataFrame(rows)


def match_lineups_dataframe(match, side):
    team = match.get(side, {})
    rows = []
    for key, label in [("lineup", "Wyjściowy"), ("bench", "Ławka")]:
        for player in team.get(key, []) or []:
            rows.append(
                {
                    "Status": label,
                    "Zawodnik": event_person_name(player),
                    "Pozycja": player.get("position", "") if isinstance(player, dict) else "",
                    "Numer": player.get("shirtNumber", "") if isinstance(player, dict) else "",
                }
            )
    return pd.DataFrame(rows)


def render_event_timeline(match):
    goals = []
    for goal in match.get("goals", []) or []:
        goals.append(
            {
                "minute": goal.get("minute") or 0,
                "injury": goal.get("injuryTime") or 0,
                "Opis": f"{display_minute(goal)} Gol: {event_person_name(goal.get('scorer') or goal.get('player'))} ({event_team_name(goal.get('team'))})",
            }
        )
    cards = []
    for booking in match.get("bookings", []) or []:
        cards.append(
            {
                "minute": booking.get("minute") or 0,
                "injury": booking.get("injuryTime") or 0,
                "Opis": f"{display_minute(booking)} Kartka: {event_person_name(booking.get('player'))} ({event_team_name(booking.get('team'))})",
            }
        )
    events = sorted(goals + cards, key=lambda item: (item["minute"], item["injury"]))
    if not events:
        st.info("API nie zwróciło osi wydarzeń dla tego meczu.")
        return
    for item in events:
        st.write(item["Opis"])


def render_match_page(match_id, token, ratings, home_adv, avg_goals, unavailable, clubelo_df=None):
    try:
        match = get_match_detail(match_id, token)
    except Exception as exc:
        st.warning(f"Nie udało się pobrać szczegółów meczu: {exc}")
        return

    st.divider()
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})
    home_name = team_display_name(home)
    away_name = team_display_name(away)
    date_text = match.get("utcDate", "")[:16].replace("T", " ")

    top = st.columns([1.3, 0.7, 1.3, 1.0])
    with top[0]:
        if st.button(home_name, key=f"detail-home-{match_id}", use_container_width=True):
            if home.get("id"):
                select_team(home.get("id"), home_name)
                st.rerun()
    top[1].metric("Wynik", score_text(match))
    with top[2]:
        if st.button(away_name, key=f"detail-away-{match_id}", use_container_width=True):
            if away.get("id"):
                select_team(away.get("id"), away_name)
                st.rerun()
    with top[3]:
        if st.button("Zamknij mecz", use_container_width=True):
            close_match_view()
            st.rerun()
        st.caption(date_text or match.get("status", ""))

    if match.get("status") != "FINISHED":
        prediction = predict_match(
            team_model_name(home),
            team_model_name(away),
            ratings,
            home_adv,
            avg_goals,
            unavailable.get(str(home.get("id")), []),
            unavailable.get(str(away.get("id")), []),
            clubelo_df,
        )
        if prediction:
            cols = st.columns(5)
            cols[0].metric("Typowany wynik", prediction["score"])
            cols[1].metric("xG gospodarzy", f"{prediction['home_xg']:.2f}")
            cols[2].metric("xG gości", f"{prediction['away_xg']:.2f}")
            cols[3].metric("Szansa 1-X-2", f"{prediction['home_win']:.0%} / {prediction['draw']:.0%} / {prediction['away_win']:.0%}")
            if prediction.get("elo_diff") is None:
                cols[4].metric("Pewność", f"{prediction['confidence']:.0%}")
            else:
                cols[4].metric("Elo diff", f"{prediction['elo_diff']:+.0f}")

    tabs = st.tabs(["Przebieg", "Gole i kartki", "Statystyki", "Składy"])
    with tabs[0]:
        render_event_timeline(match)

    with tabs[1]:
        goals_df = goals_dataframe(match)
        bookings_df = bookings_dataframe(match)
        substitutions_df = substitutions_dataframe(match)
        left, right = st.columns(2)
        with left:
            st.markdown("**Bramki**")
            if goals_df.empty:
                st.info("Brak danych o bramkach w odpowiedzi API.")
            else:
                st.dataframe(goals_df, use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Kartki**")
            if bookings_df.empty:
                st.info("Brak danych o kartkach w odpowiedzi API.")
            else:
                st.dataframe(bookings_df, use_container_width=True, hide_index=True)
        st.markdown("**Zmiany**")
        if substitutions_df.empty:
            st.info("Brak danych o zmianach w odpowiedzi API.")
        else:
            st.dataframe(substitutions_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        stats_df = match_statistics_dataframe(match)
        if stats_df.empty:
            st.info("Football-Data często nie udostępnia pełnych statystyk meczowych w darmowym planie. Jeśli endpoint je zwróci, pojawią się tutaj automatycznie.")
        else:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with tabs[3]:
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{home_name}**")
            lineup = match_lineups_dataframe(match, "homeTeam")
            if lineup.empty:
                st.info("Brak składu meczowego w odpowiedzi API.")
            else:
                st.dataframe(lineup, use_container_width=True, hide_index=True)
        with right:
            st.markdown(f"**{away_name}**")
            lineup = match_lineups_dataframe(match, "awayTeam")
            if lineup.empty:
                st.info("Brak składu meczowego w odpowiedzi API.")
            else:
                st.dataframe(lineup, use_container_width=True, hide_index=True)


def render_match(match, ratings, home_adv, avg_goals, unavailable, clubelo_df=None):
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})
    home_name = team_display_name(home)
    away_name = team_display_name(away)
    home_model = team_model_name(home)
    away_model = team_model_name(away)
    score = match.get("score", {}).get("fullTime", {})
    dt = match.get("utcDate", "")[:16].replace("T", " ")

    cols = st.columns([1.4, 0.32, 1.4, 0.75, 1.1])
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
        if st.button("Szczegóły", key=f"match-detail-{match.get('id')}", use_container_width=True):
            select_match(match.get("id"), match_title(match))
            st.rerun()
    with cols[4]:
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
                clubelo_df,
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


def render_team_page(
    team_id,
    team_name,
    token,
    league_code,
    standings_df,
    scorers_df,
    understat_df,
    ratings,
    fpl_df=None,
    clubelo_df=None,
):
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
            close_team_view()
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
    team_fpl = fpl_team_filter(fpl_df, team_name) if fpl_df is not None else pd.DataFrame()

    tabs = st.tabs(["Skład", "Statystyki zawodników", "Dostępność", "Mecze i forma", "Model"])
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
        if league_code != "PL":
            st.info("Automatyczna dostępność z FPL działa teraz dla Premier League. Dla pozostałych lig użyj ręcznej listy braków w panelu bocznym.")
        elif team_fpl.empty:
            st.info("Nie udało się dopasować tej drużyny do danych FPL.")
        else:
            missing = team_unavailable_from_fpl(team_fpl)
            st.markdown("**Kontuzje, zawieszenia i wątpliwi zawodnicy z FPL**")
            if missing:
                st.dataframe(
                    team_fpl[
                        team_fpl["Skrót"].map(lambda name: any(normalize_name(str(name)) in normalize_name(item) for item in missing))
                    ][["Zawodnik", "Pozycja", "Status", "Szansa teraz", "Szansa następna", "News", "Forma", "Punkty"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("FPL nie oznacza aktualnie istotnych braków dla tej drużyny.")
            st.markdown("**Kadra FPL i forma**")
            st.dataframe(
                team_fpl.sort_values(["Min", "Punkty"], ascending=False)[
                    ["Zawodnik", "Pozycja", "Min", "Gole", "Asysty", "xG", "xA", "Forma", "PPM", "Punkty", "Status"]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
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

    with tabs[4]:
        model_name = find_rating_name(team_model_name(detail), ratings) or find_rating_name(team_name, ratings)
        clubelo = clubelo_rating_for_team(team_model_name(detail), clubelo_df) or clubelo_rating_for_team(team_name, clubelo_df)
        if not model_name:
            st.info("Model nie znalazł odpowiednika tej drużyny w historycznych plikach CSV.")
        else:
            rating = ratings[model_name]
            cols = st.columns(6)
            cols[0].metric("Nazwa w modelu", model_name)
            cols[1].metric("Siła ataku", f"{rating['attack']:.3f}")
            cols[2].metric("Siła obrony", f"{rating['defense']:.3f}")
            cols[3].metric("Trend", f"{rating['trend']:+.3f}")
            cols[4].metric("ClubElo", f"{clubelo['elo']:.0f}" if clubelo else "-")
            cols[5].metric("Mecze treningowe", int(rating["matches"]))
            st.caption(
                "Model uczy się z kilku poprzednich sezonów, ale mocniej waży bieżący sezon. "
                "ClubElo stabilizuje ocenę siły drużyny, a niedostępni zawodnicy obniżają oczekiwane gole."
            )


def init_state():
    st.session_state.setdefault("selected_team_id", None)
    st.session_state.setdefault("selected_team_name", None)
    st.session_state.setdefault("selected_match_id", None)
    st.session_state.setdefault("selected_match_name", None)
    st.session_state.setdefault("unavailable", {})


def main():
    init_state()
    render_theme()
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
        show_league_stats = st.checkbox("Pokaż statystyki ligi i model", value=False)
        use_fpl_availability = st.checkbox("Uwzględniaj automatyczne braki FPL", value=True)
        use_clubelo = st.checkbox("Uwzględniaj ClubElo w predykcji", value=True)
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
            understat_teams, understat_players, understat_season = load_understat(
                league_meta["understat"],
                season_start_year(),
            )
            ratings, home_adv, avg_goals, history = train_prediction_model(league_meta["csv_code"])
            fpl_bootstrap = load_fpl_bootstrap() if league_code == "PL" else {}
            clubelo_df = load_clubelo_snapshot(league_meta["clubelo_country"]) if use_clubelo else pd.DataFrame()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    standings_df = standings_dataframe(standings)
    scorers_df = scorers_dataframe(scorers)
    understat_df = understat_players_dataframe(understat_players)
    fpl_df = fpl_players_dataframe(fpl_bootstrap) if league_code == "PL" else pd.DataFrame()
    auto_unavailable = build_auto_unavailable(standings_df, fpl_df) if league_code == "PL" else {}
    active_clubelo = clubelo_df if use_clubelo else pd.DataFrame()
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
            if use_fpl_availability and auto_unavailable.get(team_options[picked_team]):
                with st.expander("Automatyczne braki FPL"):
                    for item in auto_unavailable[team_options[picked_team]]:
                        st.write(f"- {item}")
        st.divider()
        st.markdown("**Źródła danych**")
        st.caption("Football-Data: mecze, tabela, składy i strzelcy. Understat: xG, xA i kartki zawodników. football-data.co.uk: historia do modelu.")
        st.caption("FPL: automatyczne braki i forma zawodników Premier League. ClubElo: niezależny rating siły drużyn.")

    all_unavailable = merge_unavailable(st.session_state.unavailable, auto_unavailable, use_fpl_availability)

    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.subheader(f"{league_meta['label']} - kolejki")
        st.markdown(f"**Ostatnia kolejka: {last_md or 'brak'}**")
        if last_matches:
            for match in last_matches:
                render_match(match, ratings, home_adv, avg_goals, all_unavailable, active_clubelo)
        else:
            st.info("Brak zakończonej kolejki w danych API.")

        st.divider()
        st.markdown(f"**Następna kolejka: {next_md or 'brak'}**")
        if next_matches:
            for match in next_matches:
                render_match(match, ratings, home_adv, avg_goals, all_unavailable, active_clubelo)
        else:
            st.info("Brak nadchodzącej kolejki w danych API.")

    with right:
        st.subheader("Tabela")
        if standings_df.empty:
            st.info("Brak tabeli w danych API.")
        else:
            render_standings(standings_df)

    if st.session_state.selected_match_id:
        render_match_page(
            st.session_state.selected_match_id,
            token.strip(),
            ratings,
            home_adv,
            avg_goals,
            all_unavailable,
            active_clubelo,
        )

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
            fpl_df,
            active_clubelo,
        )

    if show_league_stats:
        st.divider()
        st.subheader("Statystyki ligi i model")
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
                clubelo = clubelo_rating_for_team(name, active_clubelo)
                model_rows.append(
                    {
                        "Drużyna": name,
                        "Atak": rating["attack"],
                        "Obrona": rating["defense"],
                        "Trend": rating["trend"],
                        "ClubElo": clubelo["elo"] if clubelo else np.nan,
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
        st.markdown("**Status źródeł danych**")
        status_rows = [
            source_status(
                "Football-Data API",
                bool(matches.get("matches")) and not standings_df.empty,
                "Tabela, terminarz, wyniki, składy, strzelcy",
                required=True,
            ),
            source_status("football-data.co.uk", not history.empty, "Historia wyników i statystyki do treningu modelu"),
            source_status(
                "Understat",
                bool(understat_df is not None and not understat_df.empty),
                f"xG, xA, strzały i kartki zawodników"
                + (f" | użyty sezon: {understat_season}" if understat_season else " | opcjonalne źródło pominięte"),
            ),
            source_status("FPL API", not fpl_df.empty if league_code == "PL" else False, "Dostępność i forma zawodników Premier League"),
            source_status(
                "ClubElo",
                not active_clubelo.empty,
                "Niezależny rating siły drużyn"
                + (
                    f" | snapshot: {active_clubelo['SnapshotDate'].iloc[0]}"
                    if not active_clubelo.empty and "SnapshotDate" in active_clubelo.columns
                    else " | opcjonalne źródło pominięte"
                ),
            ),
        ]
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
