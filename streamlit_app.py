"""
streamlit_app.py

Version "prod" avec interface utilisateur peaufinée :
- Utilise uniquement les stats pré-calculées dans ./precomputed.
- Menus déroulants simples pour Origine / Destination.
- Aucun trajet sélectionné par défaut.
- Logos noir/blanc qui s'adaptent automatiquement au thème clair/sombre.
- Footer "Made with ❤️ in Centrale Méditerranée".
"""

import base64
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


# =====================
# Configuration générale
# =====================

# IMPORTANT : set_page_config doit être le premier appel Streamlit
st.set_page_config(
    page_title="MaxCast – Prévision TGVmax",
    page_icon="icon.png",  # icon.png doit être à côté de streamlit_app.py
    layout="centered",
)


# =====================
# Helpers & logo thème clair/sombre
# =====================

def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# Ces deux fichiers doivent être à côté de streamlit_app.py
white_logo_b64 = load_image_b64("whitelogo.png")  # pour dark mode
black_logo_b64 = load_image_b64("blacklogo.png")  # pour light mode

# Logo + CSS theme-aware avec prefers-color-scheme
st.markdown(
    f"""
    <style>
    .logo-container {{
        text-align: center;
        margin-top: 10px;
        margin-bottom: 15px;
    }}

    /* Par défaut → clair */
    .logo-dark {{ display: none; }}
    .logo-light {{ display: inline-block; }}

    /* Si mode sombre détecté par le navigateur */
    @media (prefers-color-scheme: dark) {{
        .logo-dark {{ display: inline-block; }}
        .logo-light {{ display: none; }}
    }}

    .main-title {{
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        text-align: center;
    }}
    .subtitle {{
        font-size: 0.98rem;
        color: #BBBBBB;
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    .section-card {{
        border-radius: 0.75rem;
        padding: 1.2rem 1.3rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }}
    .metric-title {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #999999;
        margin-bottom: 0.2rem;
    }}
    .metric-value {{
        font-size: 1.25rem;
        font-weight: 600;
    }}
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        padding: 8px 0 10px 0;
        text-align: center;
        font-size: 0.85rem;
        color: #888888;
        background: transparent;
    }}
    </style>

    <div class="logo-container">
        <img class="logo-light" src="data:image/png;base64,{black_logo_b64}" width="170">
        <img class="logo-dark" src="data:image/png;base64,{white_logo_b64}" width="170">
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">MaxCast – Prévision TGVmax</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    "Estime la date la plus probable d’ouverture TGVmax pour ton trajet, "
    "à partir de données historiques issues des snapshots de réservation."
    "</div>",
    unsafe_allow_html=True,
)


# =====================
# Chargement des stats
# =====================

@st.cache_resource
def get_stats(version: int) -> TgvMaxStats:
    return load_stats()


try:
    stats = get_stats(MODEL_VERSION)
except Exception as e:
    stats = None
    st.error(
        "Impossible de charger les statistiques pré-calculées.\n\n"
        f"Détail : `{e}`\n\n"
        "Vérifie que le dossier `precomputed/` contient bien `proba_global.csv`, "
        "`proba_od.parquet` et `stations.json`."
    )

st.markdown("")

# =====================
# Paramètres utilisateur
# =====================

st.markdown("### Paramètres du trajet")

if stats is None:
    st.info("Corrige d’abord le chargement des stats pour pouvoir lancer une prédiction.")
else:
    stations = stats.stations
    sentinel = "— Sélectionnez une gare —"

    col1, col2 = st.columns(2)

    with col1:
        origin_choice = st.selectbox(
            "Gare de départ",
            options=[sentinel] + stations,
            index=0,
        )

    with col2:
        destination_choice = st.selectbox(
            "Gare d’arrivée",
            options=[sentinel] + stations,
            index=0,
        )

    origin = None if origin_choice == sentinel else origin_choice
    destination = None if destination_choice == sentinel else destination_choice

    if origin is not None and destination is not None and origin == destination:
        st.warning(
            "La gare de départ et la gare d’arrivée sont identiques. "
            "Le calcul fonctionnera, mais ce n’est probablement pas un trajet réel 😄"
        )

    today = dt.date.today()
    min_date = today + dt.timedelta(days=1)
    max_date = today + dt.timedelta(days=365)

    trip_date = st.date_input(
        "Date de départ souhaitée",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
        help="Sélectionne la date de circulation souhaitée pour ton TGV.",
    )

    st.caption(
        "MaxCast utilise la distribution historique de disponibilité TGVmax "
        "en fonction du nombre de jours avant le départ. "
        "Lorsque c’est possible, la prévision est spécifique à ton trajet "
        "(origine/destination) ; sinon, le modèle retombe sur une statistique plus globale."
    )

    # =====================
    # Bouton de prédiction
    # =====================

    disabled = origin is None or destination is None

    if st.button("🔮 Lancer la prévision", disabled=disabled):
        if disabled:
            st.error("Merci de sélectionner une **gare de départ** et une **gare d’arrivée**.")
        else:
            try:
                with st.spinner("Calcul de la probabilité d’ouverture TGVmax…"):
                    forecast_df = forecast_opening_curve(
                        stats=stats,
                        departure_date=trip_date,
                        today=today,
                        origin=origin,
                        destination=destination,
                    )

                # =====================
                # Résultats principaux
                # =====================
                st.markdown("### Résultats pour ton trajet")

                date_ml, prob_ml = get_most_likely_opening_date(forecast_df)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)

                st.markdown(
                    f"**Trajet analysé :** {origin} → {destination} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Date de départ :** {trip_date.strftime('%d/%m/%Y')}"
                )

                colA, colB, colC = st.columns(3)

                with colA:
                    st.markdown(
                        '<div class="metric-title">Date la plus probable</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="metric-value">{date_ml.strftime("%d/%m/%Y")}</div>',
                        unsafe_allow_html=True,
                    )

                if today in forecast_df["date"].values:
                    prob_today = float(
                        forecast_df.loc[forecast_df["date"] == today, "prob_open_cum"].iloc[0]
                    )
                else:
                    prob_today = 0.0

                with colB:
                    st.markdown('<div class="metric-title">Déjà ouvert aujourd’hui</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-value">{prob_today * 100:.1f} %</div>',
                        unsafe_allow_html=True,
                    )

                prob_before_dep = float(forecast_df["prob_open_cum"].iloc[-1])

                with colC:
                    st.markdown('<div class="metric-title">Ouvert avant le jour J</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-value">{prob_before_dep * 100:.1f} %</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

                # =====================
                # Graphiques
                # =====================

                st.markdown("### Évolution de la probabilité")

                chart_df = forecast_df.set_index("date")[["prob_open_cum"]]
                chart_df["Probabilité cumulée d'ouverture (%)"] = chart_df["prob_open_cum"] * 100
                st.line_chart(chart_df["Probabilité cumulée d'ouverture (%)"])

                st.caption(
                    "La courbe ci-dessus représente la probabilité que la réservation TGVmax "
                    "soit *déjà ouverte* à chaque date entre aujourd’hui et la veille du départ."
                )

                with st.expander("Détail jour par jour"):
                    bar_df = forecast_df.set_index("date")[["prob_open"]]
                    bar_df["Proba ouverture ce jour-là (%)"] = bar_df["prob_open"] * 100
                    st.bar_chart(bar_df["Proba ouverture ce jour-là (%)"])

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


# =====================
# Footer
# =====================

st.markdown(
    """
    <div class="footer">
        Made with ❤️ in <strong>Centrale Méditerranée</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
