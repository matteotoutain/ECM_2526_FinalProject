# MaxCast

MaxCast est une application de **data science appliquée au transport ferroviaire** qui permet d’estimer la **probabilité d’obtenir une place avec l’abonnement TGVmax** sur un train donné.

Le projet combine **collecte de données, traitement automatisé, analyse statistique et machine learning** afin d’aider les utilisateurs à mieux planifier leurs trajets.

L’objectif est simple : **transformer des données brutes en information exploitable pour la prise de décision.**

---

# Objectif du projet

Les abonnés **TGVmax** peuvent réserver gratuitement des trains, mais **la disponibilité des places varie fortement** selon :

- l’heure
- le jour de la semaine
- la gare de départ
- la gare d’arrivée
- la période de l’année

Aujourd’hui, il est impossible de savoir à l’avance si un train aura des places disponibles.

MaxCast répond à ce problème en :

- collectant des **snapshots de disponibilité**
- analysant les **patterns historiques**
- construisant un **modèle prédictif**
- fournissant une **probabilité d’obtenir une place**

L’utilisateur peut ainsi **choisir le train avec les meilleures chances de réservation.**

---

# Fonctionnalités

Le projet inclut plusieurs briques :

### Collecte de données
- récupération régulière de snapshots de disponibilité
- stockage et historisation des données

### Pipeline de traitement
- nettoyage et structuration des données
- enrichissement avec les gares et métadonnées

### Analyse exploratoire
- étude des comportements de réservation
- identification des facteurs influents

### Machine Learning
- modélisation de la probabilité de disponibilité
- prédiction sur de nouveaux trajets

### Interface web
- application permettant de consulter les probabilités
- visualisations des résultats

---

# Architecture du projet

Le projet est structuré autour de plusieurs composants :

data/
snapshots de disponibilité collectés

notebooks/
analyse exploratoire et expérimentation

src/
pipeline de traitement
modèles de machine learning

app/
application web de consultation des résultats

---

# Contributeurs

### Matteo Toutain  
Engineering Student @ Centrale Méditerranée  
Salesforce Consultant @ SpringFive  

GitHub :  
https://github.com/matteotoutain

LinkedIn :  
https://www.linkedin.com/in/matteotoutain/

---

### Hadrien Bardon

GitHub :  
https://github.com/hadrienbardon

---

# Finalité académique

Ce projet a été réalisé dans le cadre du **Master 2 DATA & Machine Learning** de l’École Centrale Méditerranée.

Il constitue un **cas complet de projet data**, couvrant :

- acquisition de données
- data engineering
- analyse statistique
- modélisation prédictive
- déploiement applicatif

---

# Perspectives

Plusieurs évolutions sont envisagées :

- amélioration du modèle prédictif
- enrichissement du dataset historique
- intégration de nouveaux facteurs (saisonnalité, événements)
- amélioration de l’interface utilisateur

---

# Licence

Projet académique — utilisation à des fins pédagogiques et de recherche.
