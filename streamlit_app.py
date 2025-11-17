"""
streamlit_app.py
Application Streamlit locale pour interroger le modèle TGVmax.

Version sans sidebar :
- Les fichiers sont cherchés dans le dossier ./snapshots
- Sélection Origine / Destination via listes déroulantes
- Recherche texte pour filtrer les gares
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
# Version du modèle (pour casser le cache si le backend change)
# =====================

MODEL_VERSION = 2  # incrémente si tu modifies fortement tgvmax_backend.py


# =====================
# Configuration Streamlit
# =====================

st.set_page_config(
    page_title="MaxCast – Prévision ouverture TGVmax",
    page_icon="🚄",
    layout="centered",
)

st.title("🚄 MaxCast – Prévision d'ouverture TGVmax")

st.write(
    "Cette interface utilise tes snapshots journaliers `tgvmax_YYYY-MM-DD.csv` "
    "pour estimer **l'évolution de la probabilité d'ouverture** d'une réservation TGVmax "
    "par trajet (Origine / Destination) jusqu'à la date de départ."
)


# =====================
# Chargement du modèle (depuis ./snapshots)
# =====================

@st.cache_resource
def load_model(data_dir: str, version: int) -> TgvMaxModel:
    """
    version est juste là pour forcer Streamlit à reconstruire le modèle
    quand on change la structure (ajout de .stations, etc.).
    """
    model = build_model(data_dir=data_dir)
    return model


default_data_dir = pathlib.Path(__file__).parent / "snapshots"
data_dir_str = str(default_data_dir)

st.markdown(f"📂 Dossier de données utilisé : **`{data_dir_str}`**")

model: TgvMaxModel | None = None
try:
    model = load_model(data_dir_str, MODEL_VERSION)

    # Sécurité : certains vieux modèles pourraient ne pas avoir .stations
    if not hasattr(model, "stations"):
        st.warning(
            "Le modèle chargé ne contient pas l'attribut `stations`.\n\n"
            "👉 Vérifie que `tgvmax_backend.py` est bien à jour "
            "avec `proba_by_od` et `stations` dans le dataclass `TgvMaxModel`."
        )
        stations_count = "?"
    else:
        stations_count = len(model.stations)

    st.success("Modèle chargé avec succès ✅")
    st.write(f"- Lignes après filtrage : **{len(model.trains):,}**")
    st.write(f"- Gares détectées : **{stations_count}**")

except Exception as e:
    st.error(
        "Erreur lors du chargement des données : "
        f"```{e}```\n\n"
        "👉 Vérifie que le dossier `snapshots/` existe et contient des fichiers "
        "`tgvmax_YYYY-MM-DD.csv` lisibles."
    )


# =====================
# Formulaire utilisateur
# =====================

st.subheader("Paramètres du voyage")

if model is None:
    st.info("Corrige d'abord le chargement des données pour pouvoir lancer une prédiction.")
else:
    if not hasattr(model, "stations"):
        st.error(
            "Le modèle chargé ne contient pas l'attribut `stations`.\n\n"
            "👉 Assure-toi d'avoir bien remplacé `tgvmax_backend.py` par la version avec :\n"
            "`proba_by_od` et `stations` dans le dataclass TgvMaxModel."
        )
    else:
        stations = model.stations

        col_search = st.columns(2)

        # ----- Sélection Origine -----
        with col_search[0]:
            search_origin = st.text_input("🔎 Rechercher une gare de départ")
            if search_origin:
                origins_filtered = [s for s in stations if search_origin.lower() in s.lower()]
                if not origins_filtered:
                    origins_filtered = stations
                    st.caption("Aucune gare trouvée pour cette recherche, affichage complet.")
            else:
                origins_filtered = stations

            origin = st.selectbox(
                "Gare de départ",
                options=origins_filtered,
                index=0 if origins_filtered else None,
            )

        # ----- Sélection Destination -----
        with col_search[1]:
            search_dest = st.text_input("🔎 Rechercher une gare d'arrivée")
            if search_dest:
                dest_filtered = [s for s in stations if search_dest.lower() in s.lower()]
                if not dest_filtered:
                    dest_filtered = stations
                    st.caption("Aucune gare trouvée pour cette recherche, affichage complet.")
            else:
                dest_filtered = stations

            destination = st.selectbox(
                "Gare d'arrivée",
                options=dest_filtered,
                index=0 if dest_filtered else None,
            )

        if origin == destination:
            st.warning(
                "Origine et destination sont identiques. "
                "Le modèle fonctionnera, mais ce n'est sans doute pas un vrai trajet 😄"
            )

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
            "en fonction de `delta_days` pour **ce trajet** lorsque c'est possible, "
            "et retombe sur une proba globale sinon."
        )

        if st.button("🔮 Lancer la prédiction"):

            try:
                with st.spinner("Calcul de la probabilité d'ouverture TGVmax..."):
                    forecast_df = forecast_opening_curve(
                        model=model,
                        departure_date=trip_date,
                        today=today,
                        origin=origin,
                        destination=destination,
                    )

                # Résultats principaux
                st.subheader("Résultats")

                date_ml, prob_ml = get_most_likely_opening_date(forecast_df)
                st.markdown(
                    f"**Trajet :** {origin} ➜ {destination}<br>"
                    f"**Date la plus probable d'ouverture :** "
                    f"📅 **{date_ml.strftime('%d/%m/%Y')}** "
                    f"(probabilité ≈ **{prob_ml * 100:.1f} %** ce jour-là).",
                    unsafe_allow_html=True,
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
                    "soit *déjà ouverte* à chaque date pour ce trajet."
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
