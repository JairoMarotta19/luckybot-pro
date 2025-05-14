
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
from scipy.stats import poisson

st.set_page_config(page_title="LuckyBot PRO", layout="centered")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("SP1.csv")
    df = df.rename(columns={'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'FTHG': 'home_goals', 'FTAG': 'away_goals'})
    return df

df = cargar_datos()
equipos = sorted(pd.concat([df['home_team'], df['away_team']]).unique())
avg_home_goals = df['home_goals'].mean()
avg_away_goals = df['away_goals'].mean()

team_stats = {}
for team in equipos:
    home = df[df['home_team'] == team]
    away = df[df['away_team'] == team]
    attack_home = home['home_goals'].mean() if not home.empty else 0.1
    attack_away = away['away_goals'].mean() if not away.empty else 0.1
    defense_home = home['away_goals'].mean() if not home.empty else 1
    defense_away = away['home_goals'].mean() if not away.empty else 1
    team_stats[team] = {
        'attack_home': attack_home / avg_home_goals,
        'attack_away': attack_away / avg_away_goals,
        'defense_home': defense_home / avg_away_goals,
        'defense_away': defense_away / avg_home_goals
    }

def predecir_poisson(local, visitante, max_goals=5):
    h_attack = team_stats[local]['attack_home']
    a_defense = team_stats[visitante]['defense_away']
    h_expected = h_attack * a_defense * avg_home_goals
    a_attack = team_stats[visitante]['attack_away']
    h_defense = team_stats[local]['defense_home']
    a_expected = a_attack * h_defense * avg_away_goals

    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i, j] = poisson.pmf(i, h_expected) * poisson.pmf(j, a_expected)

    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()

    return {
        "g_local": round(h_expected, 2),
        "g_visitante": round(a_expected, 2),
        "prob_1": round(home_win, 4),
        "prob_X": round(draw, 4),
        "prob_2": round(away_win, 4)
    }

tab1, tab2, tab3 = st.tabs(["📊 Predicción", "💸 Apuestas con Valor", "📄 Exportar PDF"])

with tab1:
    st.title("📊 Predicción 1X2")
    local = st.selectbox("Equipo local", equipos, key="local1")
    visitante = st.selectbox("Equipo visitante", equipos, index=1, key="visitante1")

    if st.button("🔮 Predecir", key="btn1"):
        if local == visitante:
            st.warning("Selecciona dos equipos diferentes.")
        else:
            resultado = predecir_poisson(local, visitante)
            st.metric("🏠 Victoria local", f"{resultado['prob_1']*100:.2f}%")
            st.metric("🤝 Empate", f"{resultado['prob_X']*100:.2f}%")
            st.metric("🚶 Victoria visitante", f"{resultado['prob_2']*100:.2f}%")

with tab2:
    st.title("💸 Apuestas con Valor")
    local = st.selectbox("Equipo local", equipos, key="local2")
    visitante = st.selectbox("Equipo visitante", equipos, index=1, key="visitante2")

    if local != visitante:
        pred = predecir_poisson(local, visitante)

        cuota_1 = st.number_input("Cuota 1", value=1.90)
        cuota_X = st.number_input("Cuota X", value=3.30)
        cuota_2 = st.number_input("Cuota 2", value=4.20)

        value_1 = round(pred['prob_1'] * cuota_1 - 1, 3)
        value_X = round(pred['prob_X'] * cuota_X - 1, 3)
        value_2 = round(pred['prob_2'] * cuota_2 - 1, 3)

        def color(value):
            return "🟢 Valor fuerte" if value > 0.10 else "🟡 Valor débil" if value > 0 else "🔴 Sin valor"

        st.write(f"1 → {value_1:.2f} → {color(value_1)}")
        st.write(f"X → {value_X:.2f} → {color(value_X)}")
        st.write(f"2 → {value_2:.2f} → {color(value_2)}")
    else:
        st.warning("Selecciona dos equipos diferentes.")

with tab3:
    st.title("📄 Exportar Resumen PDF")
    partidos = [
        ("Real Madrid", "Valencia", "1", "65%", "Dobles: 1X"),
        ("Barcelona", "Atletico", "X", "58%", "Dobles: 1X"),
        ("Sevilla", "Betis", "2", "61%", "Dobles: X2"),
    ]
    precision_por_jornada = [(1, 66), (2, 50)]

    if st.button("📥 Generar PDF Resumen de la Jornada"):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "LuckyBot PRO - Resumen de Jornada", ln=True, align="C")
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Fecha: {date.today().strftime('%d/%m/%Y')}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Predicciones principales:", ln=True)
        pdf.set_font("Arial", '', 11)
        for local, visitante, pred, conf, dobles in partidos:
            pdf.cell(0, 8, f"{local} vs {visitante} -> {pred} ({conf}) [{dobles}]", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Precisión de jornadas anteriores:", ln=True)
        pdf.set_font("Arial", '', 11)
        for jornada, pct in precision_por_jornada:
            pdf.cell(0, 8, f"Jornada {jornada}: {pct}%", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(0, 6, "Este documento ha sido generado automaticamente por LuckyBot PRO.\nPredice. Aprende. Gana.")

        pdf_output_path = "resumen_jornada_luckybot.pdf"
        pdf.output(pdf_output_path)

        with open(pdf_output_path, "rb") as f:
            st.download_button("📤 Descargar PDF", f, file_name=pdf_output_path, mime="application/pdf")
