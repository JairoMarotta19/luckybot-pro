
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from scipy.stats import poisson

st.set_page_config(page_title="LuckyBot - Evaluación Real", layout="wide")
st.title("🔍 LuckyBot PRO - Comparativa Real: Predicciones vs Resultados")

# Simulación de modelo Poisson
@st.cache_data
def cargar_datos_poisson():
    df = pd.read_csv("SP1.csv")
    df = df.rename(columns={'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'FTHG': 'home_goals', 'FTAG': 'away_goals'})
    return df

df = cargar_datos_poisson()
equipos = sorted(pd.concat([df["home_team"], df["away_team"]]).unique())
avg_home_goals = df["home_goals"].mean()
avg_away_goals = df["away_goals"].mean()

team_stats = {}
for team in equipos:
    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]
    attack_home = home["home_goals"].mean() if not home.empty else 0.1
    attack_away = away["away_goals"].mean() if not away.empty else 0.1
    defense_home = home["away_goals"].mean() if not home.empty else 1
    defense_away = away["home_goals"].mean() if not away.empty else 1
    team_stats[team] = {
        "attack_home": attack_home / avg_home_goals,
        "attack_away": attack_away / avg_away_goals,
        "defense_home": defense_home / avg_away_goals,
        "defense_away": defense_away / avg_home_goals
    }

def predecir_poisson(local, visitante, max_goals=5):
    h_attack = team_stats.get(local, {}).get("attack_home", 1)
    a_defense = team_stats.get(visitante, {}).get("defense_away", 1)
    h_expected = h_attack * a_defense * avg_home_goals
    a_attack = team_stats.get(visitante, {}).get("attack_away", 1)
    h_defense = team_stats.get(local, {}).get("defense_home", 1)
    a_expected = a_attack * h_defense * avg_away_goals

    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i, j] = poisson.pmf(i, h_expected) * poisson.pmf(j, a_expected)

    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()

    return {
        "prob_1": home_win,
        "prob_X": draw,
        "prob_2": away_win
    }

# Cargar resultados reales desde EduardoLosilla
@st.cache_data
def cargar_resultados_losilla():
    url = "https://www.eduardolosilla.es/quiniela/resultados/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    tabla = soup.find("table", class_="tablaJornada")
    filas = tabla.find_all("tr")[1:]

    resultados = []
    for fila in filas:
        columnas = fila.find_all("td")
        if len(columnas) < 5:
            continue
        equipos = columnas[1].text.strip().split(" - ")
        if len(equipos) != 2:
            continue
        local = equipos[0].strip()
        visitante = equipos[1].strip()
        resultado_raw = columnas[3].text.strip()
        goles = re.findall(r"\d+", resultado_raw)
        if len(goles) >= 2:
            goles_local = int(goles[0])
            goles_visitante = int(goles[1])
        else:
            goles_local = goles_visitante = None
        resultados.append({
            "equipo_local": local,
            "equipo_visitante": visitante,
            "goles_local": goles_local,
            "goles_visitante": goles_visitante
        })
    return pd.DataFrame(resultados)

df_resultados = cargar_resultados_losilla()

# Mostrar y comparar con predicción real
st.subheader("🧠 Comparando predicciones Poisson vs resultados reales")

def resultado_real(row):
    if row["goles_local"] > row["goles_visitante"]:
        return "1"
    elif row["goles_local"] < row["goles_visitante"]:
        return "2"
    else:
        return "X"

predicciones = []
for _, row in df_resultados.iterrows():
    if row["equipo_local"] in team_stats and row["equipo_visitante"] in team_stats:
        probs = predecir_poisson(row["equipo_local"], row["equipo_visitante"])
        pred_max = max(probs, key=probs.get)
        predicciones.append({
            "equipo_local": row["equipo_local"],
            "equipo_visitante": row["equipo_visitante"],
            "prediccion_poisson": pred_max,
            "real": resultado_real(row),
            "acierto": pred_max == resultado_real(row)
        })

df_eval = pd.DataFrame(predicciones)
st.dataframe(df_eval)

if not df_eval.empty:
    st.metric("🎯 Precisión Poisson", f"{df_eval['acierto'].mean()*100:.2f}%")
else:
    st.warning("No hay partidos compatibles entre los datos reales y el modelo.")

