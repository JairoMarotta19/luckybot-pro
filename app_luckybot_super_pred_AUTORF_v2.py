
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import joblib

st.set_page_config(page_title="LuckyBot ⚽", layout="centered")

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

total_matches = len(df)
home_wins = (df["home_goals"] > df["away_goals"]).sum()
draws = (df["home_goals"] == df["away_goals"]).sum()
away_wins = (df["home_goals"] < df["away_goals"]).sum()
pct_home = round((home_wins / total_matches) * 100, 2)
pct_draw = round((draws / total_matches) * 100, 2)
pct_away = round((away_wins / total_matches) * 100, 2)

tab1, tab2 = st.tabs(["📊 Predicción", "💸 Value Betting"])

with tab1:
    st.title("📊 Predicción de Partidos")
    col1, col2 = st.columns(2)
    local = col1.selectbox("Equipo local", equipos, key="local1")
    visitante = col2.selectbox("Equipo visitante", equipos, index=1, key="visitante1")

    if st.button("🔮 Predecir"):
        if local == visitante:
            st.warning("Selecciona dos equipos diferentes.")
        else:
            resultado = predecir_poisson(local, visitante)
            st.success(f"{local} vs {visitante}")
            st.write(f"Goles esperados: **{resultado['g_local']} - {resultado['g_visitante']}**")
            st.metric("🏠 Victoria local", f"{resultado['prob_1']*100:.2f}%")
            st.metric("🤝 Empate", f"{resultado['prob_X']*100:.2f}%")
            st.metric("🚶 Victoria visitante", f"{resultado['prob_2']*100:.2f}%")
            st.markdown("### 🧠 Historial de la temporada")
            st.write(f"🏠 Victoria local: **{pct_home}%**")
            st.write(f"🤝 Empate: **{pct_draw}%**")
            st.write(f"🚶 Victoria visitante: **{pct_away}%**")

with tab2:
    st.title("💸 Análisis de Apuestas con Valor")
    col1, col2 = st.columns(2)
    local = col1.selectbox("Equipo local", equipos, key="local2")
    visitante = col2.selectbox("Equipo visitante", equipos, index=1, key="visitante2")

    if local != visitante:
        pred = predecir_poisson(local, visitante)
        st.write("🧠 Probabilidades estimadas por el modelo:")
        col1, col2, col3 = st.columns(3)
        col1.metric("1", f"{pred['prob_1']*100:.2f}%")
        col2.metric("X", f"{pred['prob_X']*100:.2f}%")
        col3.metric("2", f"{pred['prob_2']*100:.2f}%")
        st.subheader("💵 Introduce las cuotas reales de la casa de apuestas")
        c1, c2, c3 = st.columns(3)
        cuota_1 = c1.number_input("Cuota 1", value=1.90)
        cuota_X = c2.number_input("Cuota X", value=3.30)
        cuota_2 = c3.number_input("Cuota 2", value=4.20)
        value_1 = round(pred['prob_1'] * cuota_1 - 1, 3)
        value_X = round(pred['prob_X'] * cuota_X - 1, 3)
        value_2 = round(pred['prob_2'] * cuota_2 - 1, 3)

        def color(value):
            return "🟢 Valor fuerte" if value > 0.10 else "🟡 Valor débil" if value > 0 else "🔴 Sin valor"

        st.markdown("### 🧠 Historial de la temporada")
        st.write(f"🏠 Victoria local: **{pct_home}%**")
        st.write(f"🤝 Empate: **{pct_draw}%**")
        st.write(f"🚶 Victoria visitante: **{pct_away}%**")
        st.markdown("### 📈 Resultado del análisis:")
        st.write(f"1 → {value_1:.2f} → {color(value_1)}")
        st.write(f"X → {value_X:.2f} → {color(value_X)}")
        st.write(f"2 → {value_2:.2f} → {color(value_2)}")
    else:
        st.warning("Selecciona dos equipos diferentes.")

st.markdown("## 🤖 Super Predicción Combinada (Poisson + Random Forest)")
local_sp = st.selectbox("Equipo local", equipos, key="local_sp")
visitante_sp = st.selectbox("Equipo visitante", equipos, index=1, key="visitante_sp")

if local_sp != visitante_sp:
    pred_poisson = predecir_poisson(local_sp, visitante_sp)
    probas = {'1': pred_poisson['prob_1'], 'X': pred_poisson['prob_X'], '2': pred_poisson['prob_2']}
    pred_poisson_result = max(probas, key=probas.get)
    equipo_map = {team: idx for idx, team in enumerate(equipos)}

    try:
        modelo_rf = joblib.load("modelo_random_forest.pkl")
        home_id = equipo_map[local_sp]
        away_id = equipo_map[visitante_sp]
        features = pd.DataFrame([[home_id, away_id, 0, 2]], columns=["home_id", "away_id", "goal_diff", "total_goals"])
        if not np.issubdtype(features.values.dtype, np.number):
            st.error("❌ Error: las características para Random Forest no son numéricas.")
        else:
            pred_rf = modelo_rf.predict(features)[0]
            st.write("🔮 Poisson predice:", pred_poisson_result)
            st.write("🌲 Random Forest predice:", pred_rf)
            if pred_poisson_result == pred_rf:
                st.success(f"✅ Super Predicción: {pred_rf} (alta confianza)")
            else:
                combi = f"{pred_poisson_result} / {pred_rf}"
                st.error(f"❌ Super Predicción: Mixta ({combi}) – baja confianza")
            if pred_poisson['prob_1'] > 0.6:
                razonamiento = f"{local_sp} ataca bien y {visitante_sp} defiende mal fuera de casa."
            elif pred_poisson['prob_2'] > 0.6:
                razonamiento = f"{visitante_sp} marca bien fuera y {local_sp} concede goles en casa."
            else:
                razonamiento = "Partido igualado, sin favorito claro según el modelo."
            st.info(f"🧠 Razonamiento: {razonamiento}")
    
    except Exception as e:
        st.warning("⚠️ No se pudo cargar el modelo Random Forest. Entrenando uno nuevo...")
        df_rf = pd.read_csv("SP1.csv")
        equipos_rf = pd.concat([df_rf["home_team"], df_rf["away_team"]]).unique()
        equipo_map = {equipo: i for i, equipo in enumerate(equipos_rf)}
        df_rf["home_id"] = df_rf["home_team"].map(equipo_map)
        df_rf["away_id"] = df_rf["away_team"].map(equipo_map)
        df_rf["resultado"] = df_rf.apply(lambda row: "1" if row["home_goals"] > row["away_goals"] else "2" if row["home_goals"] < row["away_goals"] else "X", axis=1)
        X = df_rf[["home_id", "away_id"]]
        y = df_rf["resultado"]
        from sklearn.ensemble import RandomForestClassifier
        modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo_rf.fit(X, y)
        import joblib
        joblib.dump(modelo_rf, "modelo_random_forest.pkl")
        joblib.dump(equipo_map, "equipo_map.pkl")
        pred_rf = modelo_rf.predict(pd.DataFrame([[home_id, away_id, 0, 2]], columns=["home_id", "away_id", "goal_diff", "total_goals"]))[0]
        st.write("🔮 Poisson predice:", pred_poisson_result)
        st.write("🌲 Random Forest predice:", pred_rf)
        if pred_poisson_result == pred_rf:
            st.success(f"✅ Super Predicción: {pred_rf} (alta confianza)")
        else:
            combi = f"{pred_poisson_result} / {pred_rf}"
            st.error(f"❌ Super Predicción: Mixta ({combi}) – baja confianza")
        if pred_poisson['prob_1'] > 0.6:
            razonamiento = f"{local_sp} ataca bien y {visitante_sp} defiende mal fuera de casa."
        elif pred_poisson['prob_2'] > 0.6:
            razonamiento = f"{visitante_sp} marca bien fuera y {local_sp} concede goles en casa."
        else:
            razonamiento = "Partido igualado, sin favorito claro según el modelo."
        st.info(f"🧠 Razonamiento: {razonamiento}")
else:
    st.warning("Selecciona dos equipos diferentes.")
