# -*- coding: utf-8 -*-
# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
import requests
from bs4 import BeautifulSoup

# Configuration de la page
st.set_page_config(page_title="Tableau de bord - Sénégal", layout="centered")

# CSS pour fond d'écran + lisibilité
st.markdown("""
<style>
    body {
        background-image: url("https://lot.dhl.com/wp-content/uploads/2020/06/Article-Key-Image-1072036361-800x420.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }

    .stApp {
        background-color: rgba(0, 0, 0, 0.75);
        padding: 2rem;
        border-radius: 10px;
        color: white !important;
    }

    h1, h2, h3, h4, h5, h6, p, div {
        color: white !important;
    }

    .stDataFrame div {
        color: black !important;
    }

    /* =========================================================
       🔒 CORRECTION UNIQUE — MENU DÉROULANT (SELECTBOX)
       - texte en noir
       - fond blanc
       - ascenseur visible en noir
       ========================================================= */

    /* Champ sélectionné */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
    }

    /* Texte sélectionné */
    div[data-baseweb="select"] span {
        color: black !important;
    }

    /* Menu déroulant */
    div[data-baseweb="menu"] {
        background-color: white !important;
    }

    /* Tous les éléments de la liste */
    div[data-baseweb="menu"] * {
        color: black !important;
        background-color: white !important;
    }

    /* Élément survolé */
    div[data-baseweb="option"]:hover {
        background-color: #e6e6e6 !important;
        color: black !important;
    }

    /* Ascenseur (scrollbar) */
    div[data-baseweb="menu"]::-webkit-scrollbar {
        width: 10px;
    }

    div[data-baseweb="menu"]::-webkit-scrollbar-track {
        background: #f0f0f0;
    }

    div[data-baseweb="menu"]::-webkit-scrollbar-thumb {
        background-color: #000000;
        border-radius: 6px;
    }

</style>
""", unsafe_allow_html=True)

st.title("Tableau de bord – Écosystème économique du Sénégal (1960 - 2024)")

with st.expander("Présentation de l'application", expanded=True):
    st.markdown("""
### Application de Data Science & Data Engineering pour l’économie sénégalaise
#### Développée par ***Mohamed Falilou Fall***

Cette application interactive propose une exploration approfondie de **l’écosystème économique du Sénégal** de **1960 à 2024**, à partir des données officielles de la **Banque Mondiale**.

Elle permet de :
- Visualiser l’évolution de centaines d’indicateurs
- Analyser tendances et ruptures
- Télécharger les données (.csv)
- Détecter automatiquement les anomalies
- Rechercher leurs causes

/* Mettre le label et le contenu du selectbox en noir */
div[data-testid="stSelectbox"] label,
div[data-baseweb="select"] * {
    color: black !important;
}


""")

@st.cache_data
def charger_donnees():
    url = r"API_SEN_DS2_en_csv_v2_11156.csv"
    df = pd.read_csv(url, skiprows=4)
    df = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="Year",
        value_name="Value"
    )
    df = df[df["Country Name"] == "Senegal"]
    df["Year"] = df["Year"].astype(str)
    return df

df_long = charger_donnees()
indicator_list = sorted(df_long['Indicator Name'].dropna().unique())

indicateur_unique = st.selectbox(
    "Choisir un indicateur économique à visualiser :",
    indicator_list
)

df_filtre = df_long[df_long['Indicator Name'] == indicateur_unique]

if not df_filtre.empty:
    fig = px.line(
        df_filtre,
        x="Year",
        y="Value",
        color="Indicator Name",
        title=f"Évolution de l’indicateur : {indicateur_unique}"
    )
    st.plotly_chart(fig)
    st.dataframe(df_filtre)

    csv = df_filtre.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Télécharger les données de l’indicateur",
        data=csv,
        file_name="indicateur_senegal.csv",
        mime='text/csv'
    )

    st.subheader("Détection automatique des anomalies")

    df_analyse = df_filtre.copy()
    df_analyse = df_analyse[df_analyse["Year"].str.isnumeric()]
    df_analyse["Year"] = df_analyse["Year"].astype(int)
    df_analyse = df_analyse.sort_values("Year")
    df_analyse = df_analyse.dropna(subset=["Value"])

    model = IsolationForest(contamination=0.1, random_state=42)
    df_analyse["Anomaly_Score"] = model.fit_predict(df_analyse[["Value"]])
    df_analyse["Anomalie"] = df_analyse["Anomaly_Score"].apply(
        lambda x: "⚠️" if x == -1 else ""
    )

    fig_anomalie = px.line(
        df_analyse,
        x="Year",
        y="Value",
        title=f"Anomalies détectées pour : {indicateur_unique}"
    )

    anomalies = df_analyse[df_analyse["Anomalie"] == "⚠️"]

    fig_anomalie.add_scatter(
        x=anomalies["Year"],
        y=anomalies["Value"],
        mode='markers+text',
        text=anomalies["Anomalie"],
        marker=dict(color='red', size=10),
        name='Anomalie détectée'
    )

    st.plotly_chart(fig_anomalie)
    st.dataframe(df_analyse[["Year", "Value", "Anomalie"]])

else:
    st.warning("Aucune donnée pour cet indicateur.")

st.markdown("""
---
**Conceptualisé et développé par Mohamed Falilou Fall**  
Juin 2025
""")
