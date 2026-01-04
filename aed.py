import streamlit as st
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações iniciais
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

st.title("❤️ Heart Disease – Análise Exploratória de Dados")
st.markdown("Análise interativa baseada no dataset de predição de doenças cardíacas.")

@st.cache_data

# =========================
# Carregamento dos dados
# =========================

# Configuração da Kaggle API
os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

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
st.write(df.head())
st.write("Dimensão do dataset:", df.shape)

# =========================
# Informações gerais
# =========================
if st.checkbox("Mostrar informações do dataset"):
