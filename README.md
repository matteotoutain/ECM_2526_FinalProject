# 📘 MaxCast – Prévision d’ouverture TGVmax

**MaxCast** est un outil de prévision permettant d’estimer **la probabilité d’ouverture TGVmax** pour un trajet donné, à partir :

- des **snapshots journaliers réels** de disponibilité TGVmax,
- de **statistiques pré-calculées** basées sur l’historique,
- et d’une **interface Streamlit** simple et élégante.

Le système fonctionne entièrement **offline** et peut être déployé gratuitement via Streamlit Cloud ou Vercel (via Streamlit Server).

---

## 🚄 Fonctionnalités principales

- 🎯 **Prédiction par trajet** (origine → destination)  
  Basée sur les probabilités historiques d’ouverture selon le nombre de jours avant départ.

- 📊 **Courbe prédictive interactive**  
  Affiche la probabilité cumulée d’ouverture et l’évolution jour par jour.

- 🔍 **Prise en compte du snapshot du jour**  
  Si le dernier fichier `tgvmax_YYYY-MM-DD.csv` indique que le trajet est déjà ouvert/fermé :  
  → le front l’affiche immédiatement avec un message clair.

- ⚡ **Mode ultra-léger**  
  Le backend statique (`tgvmax_stats_backend.py`) charge uniquement des CSV/Parquet pré-calculés.

- 🎨 **Interface moderne**  
  Logos adaptatifs clair/sombre, sections stylées, et UX optimisée.

---

## 📁 Structure du projet

```bash
.
├── precomputed/
│   ├── proba_global.csv
│   ├── proba_od.parquet
│   ├── stations.json
│   └── (optionnel) snapshot_today.csv
│
├── snapshots/
│   ├── tgvmax_2025-09-28.csv
│   ├── tgvmax_2025-09-29.csv
│   └── ...  (dernier snapshot utilisé automatiquement)
│
├── streamlit_app.py
├── tgvmax_stats_backend.py
├── tgvmax_backend.py              # backend complet, optionnel
├── requirements.txt
└── README.md
