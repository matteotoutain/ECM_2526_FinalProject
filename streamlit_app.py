"""
streamlit_app.py
Application Streamlit locale pour interroger le modèle TGVmax.
"""

import datetime as dt
import pathlib

import streamlit as st
import pandas as pd

from tgvmax_backend import (
    TgvMaxModel,
    build_model,
    forecast_opening_curve,
    get_most_likely_opening_date,
)


# =====================
# Configuration Streamlit
# =====================

st.set_page_config(
    page_title="Prévision ouverture TGVmax",
    page_icon="🚄",
    layout="centered",
)

st.title("🚄 Prévision d'ouverture TGVmax")

st.write(
    "Cette interface utilise tes snapshots journaliers `tgvmax_YYYY-MM-DD.csv` "
    "pour estimer **l'évolution de la probabilité d'ouverture** d'une réservation TGVmax "
    "jusqu'à la date de départ."
)


# =====================
# Chargement du modèle
# =====================

@st.cache_resource
def load_model(data_dir: str) -> TgvMaxModel:
    return build_model(data_dir=data_dir)


st.sidebar.header("Configuration des données")

default_data_dir = pathlib.Path(__file__).parent / "snapshots"
data_dir_str = st.sidebar.text_input(
    "Dossier contenant les fichiers tgvmax_*.csv",
    value=str(default_data_dir),
)

load_btn = st.sidebar.button("🔁 (Re)charger le modèle")

model: TgvMaxModel | None = None
if load_btn or "model_loaded" not in st.session_state:
    try:
        model = load_model(data_dir_str)
        st.session_state["model_loaded"] = True
        st.success("Modèle chargé avec succès ✅")
        st.sidebar.write(f"{len(model.trains):,} lignes après filtrage.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
else:
    # Si déjà en cache
    try:
        model = load_model(data_dir_str)
    except Exception as e:
        st.error(f"Erreur lors du chargement (cache) des données : {e}")


# =====================
# Formulaire utilisateur
# =====================

st.subheader("Paramètres du voyage")

col1, col2 = st.columns(2)
with col1:
    origin = st.text_input("Gare de départ (pour l'instant indicatif)", value="Paris Montparnasse")
with col2:
    destination = st.text_input("Gare d'arrivée (pour l'instant indicatif)", value="Bordeaux Saint-Jean")

today = dt.date.today()
min_date = today + dt.timedelta(days=1)
max_date = today + dt.timedelta(days=365)

trip_date = st.date_input(
    "Date de départ",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    format="DD/MM/YYYY",
)

st.caption(
    "Pour l'instant, le modèle utilise une **proba globale** en fonction de `delta_days` "
    "(jours avant départ), tout trajet confondu. On pourra affiner plus tard par OD / axe."
)

if st.button("🔮 Lancer la prédiction"):

    if model is None:
        st.error("Le modèle n'est pas chargé. Vérifie le dossier de données et recharge.")
    else:
        try:
            with st.spinner("Calcul de la probabilité d'ouverture TGVmax..."):
                forecast_df = forecast_opening_curve(
                    model=model,
                    departure_date=trip_date,
                    today=today,
                )

            # Résultats principaux
            st.subheader("Résultats")

            date_ml, prob_ml = get_most_likely_opening_date(forecast_df)
            st.markdown(
                f"**Date la plus probable d'ouverture :** "
                f"📅 **{date_ml.strftime('%d/%m/%Y')}** "
                f"(probabilité ≈ **{prob_ml * 100:.1f} %** ce jour-là)."
            )

            # Probabilité que ce soit déjà ouvert aujourd'hui
            if today in forecast_df["date"].values:
                prob_today = float(
                    forecast_df.loc[forecast_df["date"] == today, "prob_open_cum"].iloc[0]
                )
            else:
                prob_today = 0.0

            st.markdown(
                f"- **Probabilité que TGVmax soit déjà ouvert aujourd'hui** : "
                f"≈ **{prob_today * 100:.1f} %**"
            )

            prob_before_dep = float(forecast_df["prob_open_cum"].iloc[-1])
            st.markdown(
                f"- **Probabilité que l'ouverture ait lieu avant le jour du départ** : "
                f"≈ **{prob_before_dep * 100:.1f} %**"
            )

            # Graphiques
            st.subheader("Évolution de la probabilité dans le temps")

            chart_df = forecast_df.set_index("date")[["prob_open_cum"]]
            chart_df["Probabilité cumulée d'ouverture (%)"] = chart_df["prob_open_cum"] * 100
            st.line_chart(chart_df["Probabilité cumulée d'ouverture (%)"])

            st.caption(
                "La courbe ci-dessus montre la probabilité que la réservation TGVmax "
                "soit *déjà ouverte* à chaque date."
            )

            with st.expander("Voir la probabilité d'ouverture par jour"):
                bar_df = forecast_df.set_index("date")[["prob_open"]]
                bar_df["Proba ouverture ce jour-là (%)"] = bar_df["prob_open"] * 100
                st.bar_chart(bar_df["Proba ouverture ce jour-là (%)"])

            with st.expander("Détails chiffrés"):
                df_show = forecast_df.copy()
                df_show["date"] = pd.to_datetime(df_show["date"]).dt.strftime("%d/%m/%Y")
                df_show["prob_open"] = (df_show["prob_open"] * 100).round(2)
                df_show["prob_open_cum"] = (df_show["prob_open_cum"] * 100).round(2)
                df_show.rename(
                    columns={
                        "date": "Date",
                        "prob_open": "Proba ouverture ce jour-là (%)",
                        "prob_open_cum": "Proba ouverture avant ou à cette date (%)",
                    },
                    inplace=True,
                )
                st.dataframe(df_show, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
