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

# =========================
# Carregamento dos dados
# =========================
@st.cache_data

os.environ["guilhermebstella"] = st.secrets["guilhermebstella"]
os.environ["710d19fdcfada37e89085fc6dfdb5c52"] = st.secrets["710d19fdcfada37e89085fc6dfdb5c52"]

os.system("kaggle datasets download -d johnsmith/heart-disease")
with zipfile.ZipFile("heart-disease.zip", "r") as zip_ref:
    zip_ref.extractall("data")

df = pd.read_csv("data/heart.csv")

st.subheader("📊 Visão geral dos dados")
st.write(df.head())
st.write("Dimensão do dataset:", df.shape)

# =========================
# Informações gerais
# =========================
if st.checkbox("Mostrar informações do dataset"):
