"""
streamlit_app.py
Version "prod" avec :
- Logos adaptés au thème clair/sombre
- Interface peaufinée
- Stats pré-calculées dans ./precomputed
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

MODEL_VERSION = 1


# ======================================================
# Configuration générale
# ======================================================

st.set_page_config(
    page_title="MaxCast – Prévision TGVmax",
    page_icon="🚄",
    layout="centered",
)

# ---- Détection du thème clair/sombre ----
theme = st.get_option("theme.base")
if theme == "dark":
    logo_path = "whitelogo.png"
else:
    logo_path = "blacklogo.png"

# ---- Logo centré ----
st.markdown(
    f"""
    <div style="text-align:center; margin-bottom: 1.2rem;">
        <img src="data:image/png;base64,{st.image(logo_path, use_column_width=False).image_to_url(logo_path)}" style="width:180px;" />
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Style global ----
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1rem;
        text-align: center;
        color: #BBBBBB;
        margin-bottom: 1.5rem;
    }
    .section-card {
        border-radius: 0.75rem;
        padding: 1.2rem 1.3rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #999999;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.1rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Titre ----
st.markdown('<div class="main-title">MaxCast – Prévision TGVmax</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    "Estime la date la plus probable d’ouverture TGVmax pour ton trajet, "
    "grâce à une analyse quotidienne des historiques de réservation."
    "</div>",
    unsafe_allow_html=True,
)


# ======================================================
# Chargement des stats
# ======================================================

@st.cache_resource
def get_stats(version: int) -> TgvMaxStats:
    return load_stats()


try:
    stats = get_stats(MODEL_VERSION)
except Exception as e:
    stats = None
    st.error(
        "❌ Impossible de charger les statistiques pré-calculées.\n"
        f"`{e}`"
    )


st.markdown("")

# ======================================================
# Paramètres utilisateur
# ======================================================

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

    if origin == destination and origin is not None:
        st.warning("La gare de départ et la gare d’arrivée sont identiques 😄")

    today = dt.date.today()
    min_date = today + dt.timedelta(days=1)
    max_date = today + dt.timedelta(days=365)

    trip_date = st.date_input(
        "Date de départ souhaitée",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
    )

    st.caption(
        "MaxCast utilise la distribution historique de disponibilité TGVmax "
        "en fonction du nombre de jours avant le départ pour établir une prévision "
        "globale ou spécifique au trajet."
    )

    disabled = origin is None or destination is None

    # ======================================================
    # Bouton de prédiction
    # ======================================================

    if st.button("🔮 Lancer la prévision", disabled=disabled):
        if disabled:
            st.error("Merci de sélectionner une **gare de départ** et une **gare d’arrivée**.")
        else:
            try:
                with st.spinner("Calcul des probabilités…"):
                    forecast_df = forecast_opening_curve(
                        stats=stats,
                        departure_date=trip_date,
                        today=today,
                        origin=origin,
                        destination=destination,
                    )

                date_ml, prob_ml = get_most_likely_opening_date(forecast_df)

                # ======================================================
                # Résultats principaux
                # ======================================================
                st.markdown("### Résultats")

                st.markdown('<div class="section-card">', unsafe_allow_html=True)

                st.markdown(
                    f"**Trajet :** {origin} → {destination} &nbsp;&nbsp;|&nbsp;&nbsp;"
                    f"**Départ :** {trip_date.strftime('%d/%m/%Y')}"
                )

                colA, colB, colC = st.columns(3)

                with colA:
                    st.markdown('<div class="metric-title">Date la plus probable</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-value">{date_ml.strftime("%d/%m/%Y")}</div>',
                        unsafe_allow_html=True,
                    )

                if today in forecast_df["date"].values:
                    prob_today = float(
                        forecast_df.loc[forecast_df["date"] == today, "prob_open_cum"].iloc[0]
                    )
                else:
                    prob_today = 0

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

                # ======================================================
                # Graphiques
                # ======================================================

                st.markdown("### Évolution de la probabilité")

                chart_df = forecast_df.set_index("date")[["prob_open_cum"]]
                chart_df["Probabilité cumulée d'ouverture (%)"] = chart_df["prob_open_cum"] * 100
                st.line_chart(chart_df["Probabilité cumulée d'ouverture (%)"])

                st.caption(
                    "Cette courbe représente la probabilité que la réservation soit déjà ouverte "
                    "entre aujourd’hui et la veille du départ."
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
                st.error(f"Erreur lors du calcul : {e}")
