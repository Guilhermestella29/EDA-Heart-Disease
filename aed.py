import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import zipfile
import os

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
    return pd.read_csv("heart.csv")

df = load_data()

st.subheader("📊 Visão geral dos dados")
st.dataframe(df.head())
st.write("Dimensão do dataset:", df.shape)
#----------------------------------------------------------------- Age vs Cholestereol  ---------------------
x = df['Cholesterol']
y = df['Age']

fig, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=x, y=y, s=6, color="black", alpha=0.8, ax=ax)
sns.histplot(x=x, y=y, bins=60, pthresh=0.05, cmap="rocket", cbar=True, ax=ax)
sns.kdeplot(x=x, y=y, levels=8, color="blue", linewidths=1.2, ax=ax)

ax.set_title("Age vs Cholesterol Distribution")

plt.show()




