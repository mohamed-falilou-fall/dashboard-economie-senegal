# -*- coding: utf-8 -*-
# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# Configuration de la page
st.markdown("""
    <style>
        body {
            background-image: url("https://www.fao.org/4/AC919F/AC919F05.gif");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }

        .stApp {
            background-color: rgba(0, 0, 0, 0.75); /* fond sombre pour contraste */
            padding: 2rem;
            border-radius: 10px;
            color: white !important;
        }

        h1, h2, h3, h4, h5, h6, p, div {
            color: white !important;
        }

        .stMarkdown, .stTextInput, .stSelectbox, .stDataFrame {
            color: white !important;
        }

        .stDataFrame div {
            color: black !important;  /* tableau en texte noir sur fond clair */
        }
    </style>
""", unsafe_allow_html=True)
            
    
        

st.title("Tableau de bord – Écosystème économique du Sénégal (1960 - 2024)")

# Présentation de l'application
with st.expander(" Présentation de l'application", expanded=True):
    st.markdown("""
### Application de Data Science & Data Engineering pour l’économie sénégalaise
#### Développée par **Mohamed Falilou Fall**

Cette application interactive propose une exploration approfondie de **l’écosystème économique du Sénégal** de **1960 à 2024**, à partir des données officielles de la **Banque Mondiale**.

Elle s’inscrit à la croisée de la **science des données** et de l’**ingénierie des données**, en mettant à disposition un tableau de bord dynamique, conçu pour :

-  **Visualiser** l’évolution de centaines d’indicateurs économiques (PIB, inflation, commerce, agriculture, éducation, etc.)
-  **Analyser** les tendances structurelles et conjoncturelles sur plus de six décennies
-  **Exporter** les données prêtes à l’emploi pour vos rapports, travaux de recherche, politiques publiques ou projets de développement

#### Fonctionnalités clés :
- Sélection d’un indicateur parmi plus de 400 disponibles
- Graphique interactif retraçant l’évolution temporelle de l’indicateur
- Affichage des données tabulaires brutes
- Option de **téléchargement direct** du jeu de données filtré (.csv)

---

### Public cible :
- **Économistes** et analystes macroéconomiques
- **Chercheurs** et étudiants en sciences sociales
- **Décideurs publics**, experts en planification
- **Organisations internationales** et ONG
- **Journalistes économiques**, entrepreneurs, investisseurs

---

### Pourquoi c’est utile ?
-  Fournit une base factuelle pour la formulation de politiques publiques
-  Permet d’identifier des ruptures, tendances ou anomalies historiques
-  Facilite l'intégration des données macro dans des projets de développement, des rapports d’impact ou des analyses prospectives

---

**️ Une brique essentielle pour un pilotage économique éclairé par la donnée, en phase avec les enjeux de transformation structurelle du Sénégal.**
""")

# Chargement des données
@st.cache_data
def charger_donnees():
    url = "API_SEN_DS2_en_csv_v2_11156.csv"
    df = pd.read_csv(url, skiprows=4)
    df = df.melt(id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"], 
                 var_name="Year", value_name="Value")
    df = df[df["Country Name"] == "Senegal"]
    df["Year"] = df["Year"].astype(str)
    return df

df_long = charger_donnees()
indicator_list = sorted(df_long['Indicator Name'].dropna().unique())

# Interface utilisateur
indicateur_unique = st.selectbox("Choisir un indicateur économique à visualiser :", indicator_list)
df_filtre = df_long[df_long['Indicator Name'] == indicateur_unique]

# Affichage du graphique et des données
if not df_filtre.empty:
    fig = px.line(df_filtre, x="Year", y="Value", color="Indicator Name", 
                  title=f"Évolution de l’indicateur : {indicateur_unique}")
    st.plotly_chart(fig)
    st.dataframe(df_filtre)
    
    csv = df_filtre.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger les données de l’indicateur", 
                       data=csv, 
                       file_name="indicateur_senegal.csv", 
                       mime='text/csv')
else:
    st.warning("Aucune donnée pour cet indicateur.")

st.markdown("""
---
**Conceptualisé et développé par Mohamed Falilou Fall**  
Juin 2025  
📧 Email : [mff.falilou.fall@gmail.com](mailto:mff.falilou.fall@gmail.com)
""")
