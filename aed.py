import streamlit as st
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
def load_data():
    return pd.read_csv("Heart_Disease_Prediction.csv")

df = load_data()

st.subheader("📊 Visão geral dos dados")
st.write(df.head())
st.write("Dimensão do dataset:", df.shape)

# =========================
# Informações gerais
# =========================
if st.checkbox("Mostrar informações do dataset"):
