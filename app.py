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
            background-image: url("https://www.lemoniteur.fr/mediatheque/8/3/3/002422338_896x598_c.jpeg");
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
    </style>
""", unsafe_allow_html=True)

st.title("Tableau de bord – Écosystème économique du Sénégal (1960 - 2024)")

with st.expander("Présentation de l'application", expanded=True):
    st.markdown("""
### Application de Data Science & Data Engineering pour l’économie sénégalaise
#### Développée par **Mohamed Falilou Fall**

Cette application interactive propose une exploration approfondie de **l’écosystème économique du Sénégal** de **1960 à 2024**, à partir des données officielles de la **Banque Mondiale**.

Elle met à disposition un tableau de bord dynamique, conçu pour :
-  **Visualiser** l’évolution de centaines d’indicateurs économiques
-  **Analyser** les tendances et ruptures
-  **Télécharger** les données filtrées (.csv)
-  **Détecter automatiquement les anomalies statistiques**
-  **Rechercher les causes des anomalies sur Google**

---

**️Une brique essentielle pour un pilotage économique éclairé par la donnée.**
""")

@st.cache_data
def charger_donnees():
    url = r"API_SEN_DS2_en_csv_v2_11156.csv"
    df = pd.read_csv(url, skiprows=4)
    df = df.melt(id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"], 
                 var_name="Year", value_name="Value")
    df = df[df["Country Name"] == "Senegal"]
    df["Year"] = df["Year"].astype(str)
    return df

df_long = charger_donnees()
indicator_list = sorted(df_long['Indicator Name'].dropna().unique())

indicateur_unique = st.selectbox("Choisir un indicateur économique à visualiser :", indicator_list)
df_filtre = df_long[df_long['Indicator Name'] == indicateur_unique]

if not df_filtre.empty:
    fig = px.line(df_filtre, x="Year", y="Value", color="Indicator Name", 
                  title=f"Évolution de l’indicateur : {indicateur_unique}")
    st.plotly_chart(fig)
    st.dataframe(df_filtre)

    csv = df_filtre.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger les données de l’indicateur", data=csv, file_name="indicateur_senegal.csv", mime='text/csv')

    st.subheader("🔎 Détection automatique des anomalies")
    st.markdown("""
> Une **anomalie** correspond à une variation brutale ou inhabituelle d’un indicateur.
> Elle peut résulter d’un choc économique, d’une réforme majeure, ou d’un changement structurel.
    """)

    df_analyse = df_filtre.copy()
    df_analyse = df_analyse[df_analyse["Year"].str.isnumeric()]
    df_analyse["Year"] = df_analyse["Year"].astype(int)
    df_analyse = df_analyse.sort_values("Year")
    df_analyse = df_analyse.dropna(subset=["Value"])

    model = IsolationForest(contamination=0.1, random_state=42)
    df_analyse["Anomaly_Score"] = model.fit_predict(df_analyse[["Value"]])
    df_analyse["Anomalie"] = df_analyse["Anomaly_Score"].apply(lambda x: "⚠️" if x == -1 else "")

    fig_anomalie = px.line(df_analyse, x="Year", y="Value", title=f"Anomalies détectées pour : {indicateur_unique}")
    anomalies = df_analyse[df_analyse["Anomalie"] == "⚠️"]

    fig_anomalie.add_scatter(x=anomalies["Year"], y=anomalies["Value"],
                             mode='markers+text',
                             text=anomalies["Anomalie"],
                             marker=dict(color='red', size=10),
                             name='Anomalie détectée')

    st.plotly_chart(fig_anomalie)
    st.dataframe(df_analyse[["Year", "Value", "Anomalie"]])

    st.subheader("🌐 Recherche des causes possibles des anomalies (via Google)")

    def rechercher_causes(indicateur, annee):
        requete = f"Causes {indicateur} Sénégal {annee}"
        url = f"https://www.google.com/search?q={requete.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            resultats = soup.select(".tF2Cxc")
            liens = []
            for r in resultats[:3]:
                titre = r.select_one("h3")
                lien = r.select_one("a")["href"]
                if titre and lien:
                    liens.append((titre.text.strip(), lien))
            return liens if liens else [("Aucun résultat pertinent trouvé", "#")]
        except Exception as e:
            return [("Erreur lors de la recherche", str(e))]

    for _, row in anomalies.iterrows():
        st.markdown(f"### 📅 Année : {int(row['Year'])} – Anomalie détectée")
        st.markdown(f"**Indicateur concerné :** `{indicateur_unique}`")

        with st.spinner("🔍 Recherche des causes..."):
            resultats = rechercher_causes(indicateur_unique, int(row["Year"]))
        
        for titre, lien in resultats:
            st.markdown(f"- 🔗 [{titre}]({lien})")
        st.markdown("---")

else:
    st.warning("Aucune donnée pour cet indicateur.")

st.markdown("""
---
**Conceptualisé et développé par Mohamed Falilou Fall**  
📧 [mff.falilou.fall@gmail.com](mailto:mff.falilou.fall@gmail.com)  
🗓️ Juin 2025
""")
