"""
streamlit_app.py

Version "prod" qui lit uniquement les stats pré-calculées dans ./precomputed.
"""

import datetime as dt

import streamlit as st
import pandas as pd

from tgvmax_stats_backend import (
    TgvMaxStats,
    load_stats,
    forecast_opening_curve,
    get_most_likely_opening_date,
)

MODEL_VERSION = 1  # incrémente si tu changes la structure des stats


st.set_page_config(
    page_title="MaxCast – Prévision ouverture TGVmax",
    page_icon="🚄",
    layout="centered",
)

st.title("🚄 MaxCast – Prévision d'ouverture TGVmax")

st.write(
    "Cette interface utilise des statistiques **pré-calculées** à partir de snapshots journaliers "
    "`tgvmax_YYYY-MM-DD.csv` pour estimer l'évolution de la probabilité d'ouverture TGVmax "
    "par trajet (Origine / Destination)."
)


@st.cache_resource
def get_stats(version: int) -> TgvMaxStats:
    return load_stats()


try:
    stats = get_stats(MODEL_VERSION)
    st.success("Stats pré-calculées chargées ✅")
    st.write(f"- Gares disponibles : **{len(stats.stations)}**")
except Exception as e:
    stats = None
    st.error(
        "Impossible de charger les stats pré-calculées.\n\n"
        f"Détail : `{e}`\n\n"
        "Vérifie que le dossier `precomputed/` contient bien `proba_global.csv`, "
        "`proba_od.parquet` et `stations.json`."
    )

st.subheader("Paramètres du voyage")

if stats is None:
    st.info("Corrige d'abord le chargement des stats pour pouvoir lancer une prédiction.")
else:
    stations = stats.stations

    col_search = st.columns(2)

    with col_search[0]:
        search_origin = st.text_input("🔎 Rechercher une gare de départ")
        if search_origin:
            origins_filtered = [s for s in stations if search_origin.lower() in s.lower()]
            if not origins_filtered:
                origins_filtered = stations
                st.caption("Aucune gare trouvée, affichage complet.")
        else:
            origins_filtered = stations
        origin = st.selectbox("Gare de départ", options=origins_filtered)

    with col_search[1]:
        search_dest = st.text_input("🔎 Rechercher une gare d'arrivée")
        if search_dest:
            dest_filtered = [s for s in stations if search_dest.lower() in s.lower()]
            if not dest_filtered:
                dest_filtered = stations
                st.caption("Aucune gare trouvée, affichage complet.")
        else:
            dest_filtered = stations
        destination = st.selectbox("Gare d'arrivée", options=dest_filtered)

    if origin == destination:
        st.warning("Origine = destination, ce n'est probablement pas un vrai trajet 😄")

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
        "Le modèle utilise la distribution historique de disponibilité TGVmax "
        "en fonction de `delta_days` pour ce trajet lorsque c'est possible, "
        "et retombe sur une proba plus globale sinon."
    )

    if st.button("🔮 Lancer la prédiction"):
        try:
            with st.spinner("Calcul de la probabilité d'ouverture TGVmax..."):
                forecast_df = forecast_opening_curve(
                    stats=stats,
                    departure_date=trip_date,
                    today=today,
                    origin=origin,
                    destination=destination,
                )

            st.subheader("Résultats")

            date_ml, prob_ml = get_most_likely_opening_date(forecast_df)
            st.markdown(
                f"**Trajet :** {origin} ➜ {destination}<br>"
                f"**Date la plus probable d'ouverture :** "
                f"📅 **{date_ml.strftime('%d/%m/%Y')}** "
                f"(probabilité ≈ **{prob_ml * 100:.1f} %** ce jour-là).",
                unsafe_allow_html=True,
            )

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

            st.subheader("Évolution de la probabilité dans le temps")

            chart_df = forecast_df.set_index("date")[["prob_open_cum"]]
            chart_df["Probabilité cumulée d'ouverture (%)"] = chart_df["prob_open_cum"] * 100
            st.line_chart(chart_df["Probabilité cumulée d'ouverture (%)"])

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
