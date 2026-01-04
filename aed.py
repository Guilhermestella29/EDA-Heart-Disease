import streamlit as st
import os
import zipfile
import pandas as pd

st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

st.title("❤️ Heart Disease – Análise Exploratória de Dados")
st.markdown("Análise interativa baseada no dataset de predição de doenças cardíacas.")

# =========================
# CONFIGURAÇÃO KAGGLE
# =========================
os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

# =========================
# CARREGAMENTO DOS DADOS
# =========================
@st.cache_data
def load_data():
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/heart.csv"):
        os.system("kaggle datasets download -d johnsmith/heart-disease -p data")
        with zipfile.ZipFile("data/heart-disease.zip", "r") as zip_ref:
            zip_ref.extractall("data")

    return pd.read_csv("data/heart.csv")

df = load_data()

st.subheader("📊 Visão geral dos dados")
st.dataframe(df.head())
st.write("Dimensão do dataset:", df.shape)
