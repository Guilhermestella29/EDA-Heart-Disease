import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Heart Disease Exploratory Data Analysis",
    layout="wide"
)

# =========================
# TITLE & INTRODUCTION
# =========================
st.title("❤️ Heart Disease – Exploratory Data Analysis (EDA)")

st.markdown(
    """
This application presents an **exploratory data analysis (EDA)** of a heart disease dataset.
The goal is to understand variable distributions, relationships between clinical indicators,
and potential patterns related to cardiovascular risk.
"""
)

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# =========================
# DATA OVERVIEW
# =========================
st.header("📊 Dataset Overview")

st.markdown(
    """
This section provides a high-level view of the dataset, including its structure,
dimensions, and the first observations.
"""
)

st.write("**Dataset shape:**", df.shape)
st.dataframe(df.head())

# =========================
# DESCRIPTIVE STATISTICS
# =========================
st.header("📈 Descriptive Statistics")

st.markdown(
    """
Descriptive statistics summarize the central tendency, dispersion, and range
of numerical variables in the dataset.
"""
)

st.dataframe(df.describe())

# =========================
# DISTRIBUTION OF AGE
# =========================
st.header("🎂 Age Distribution")

st.markdown(
    """
The age distribution helps identify the population profile included in the dataset
and possible age concentrations related to heart disease risk.
"""
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="steelblue", ax=ax)
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
st.pyplot(fig)
plt.close(fig)
#---------------------------------------------------------------
st.header("📊 Distribution of Clinical Variables")

st.markdown(
    """
The histograms below show the distribution of the main clinical variables.
This visualization helps identify skewness, concentration ranges, and
potential outliers in the dataset.
"""
)

columns = [
    'Age',
    'BP',
    'Cholesterol',
    'Max HR',
    'ST depression',
    'Slope of ST',
    'Chest pain type',
    'Thallium',
    'Heart Disease'
]

fig, axs = plt.subplots(3, 3, figsize=(15, 10))

for ax, col in zip(axs.flat, columns):
    sns.histplot(df[col], ax=ax, bins=30, kde=True)
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("Frequency")

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# =========================
# CHOLESTEROL vs AGE
# =========================
st.header("🩸 Age vs Cholesterol")

st.markdown(
    """
This visualization combines scatter points, density estimation, and contour lines
to analyze the relationship between **age** and **cholesterol levels**.
"""
)

x = df["Cholesterol"]
y = df["Age"]

fig, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=x, y=y, s=6, color="black", alpha=0.7, ax=ax)
sns.histplot(x=x, y=y, bins=60, pthresh=0.05, cmap="rocket", cbar=True, ax=ax)
sns.kdeplot(x=x, y=y, levels=8, color="blue", linewidths=1.2, ax=ax)

ax.set_title("Age vs Cholesterol Distribution")
ax.set_xlabel("Cholesterol")
ax.set_ylabel("Age")

st.pyplot(fig)
plt.close(fig)

# =========================
# BLOOD PRESSURE ANALYSIS
# =========================
st.header("💉 Blood Pressure Analysis")

st.markdown(
    """
Blood pressure is a key clinical indicator for cardiovascular diseases.
The distribution below shows how systolic blood pressure values are spread across patients.
"""
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["BP"], bins=30, kde=True, color="darkred", ax=ax)
ax.set_title("Blood Pressure Distribution")
ax.set_xlabel("Blood Pressure")
ax.set_ylabel("Frequency")
st.pyplot(fig)
plt.close(fig)

# =========================
# CORRELATION HEATMAP
# =========================
st.header("🔗 Correlation Analysis")

st.markdown(
    """
The correlation heatmap highlights linear relationships between numerical variables.
Strong positive or negative correlations may indicate relevant clinical associations.
"""
)

numeric_data = df.select_dtypes(include="number")
corr = numeric_data.corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax
)
ax.set_title("Correlation Heatmap")

st.pyplot(fig)
plt.close(fig)

# =========================
# CONCLUSION
# =========================
st.header("🧠 Key Insights")

st.markdown(
    """
- The dataset presents a wide age range, allowing meaningful demographic analysis.
- Cholesterol and blood pressure show relevant variability across patients.
- Correlation analysis helps identify potential predictors for heart disease risk.
- These insights can support further **feature selection**, **statistical analysis**, or **machine learning modeling**.
"""
)
