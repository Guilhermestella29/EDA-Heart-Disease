import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Heart Disease EDA",
    page_icon="❤️",
    layout="wide"
)

# =========================
# DATA LOADING
# =========================
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# =========================
# DATA CLEANING
# =========================
df["Heart Disease"] = pd.to_numeric(
    df["Heart Disease"],
    errors="coerce"
)

# =========================
# TITLE
# =========================
st.markdown(
    "<h1 style='text-align:center;'>❤️ Heart Disease – Exploratory Data Analysis</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Exploratory analysis of clinical indicators related to cardiovascular disease.</p>",
    unsafe_allow_html=True
)

st.divider()

# =========================
# DATA OVERVIEW
# =========================
st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)

col1.metric("Patients", df.shape[0])
col2.metric("Features", df.shape[1])

if df["Heart Disease"].notna().any():
    disease_rate = df["Heart Disease"].mean() * 100
    disease_text = f"{disease_rate:.1f}%"
else:
    disease_text = "N/A"

st.metric("Heart Disease Prevalence", disease_text)

with st.expander("📄 Dataset Preview"):
    st.dataframe(df.head(), use_container_width=True)

# =========================
# AGE & BP ANALYSIS
# =========================
st.header("🎂 Age & Blood Pressure")

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(df["Age"], bins=25, kde=True, ax=ax)
    ax.set_title("Age Distribution", loc="center")
    st.pyplot(fig)
    plt.close(fig)

with c2:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(
        x="Age",
        y="BP",
        estimator="mean",
        data=df,
        ax=ax
    )
    ax.set_title("Average Blood Pressure by Age", loc="center")
    st.pyplot(fig)
    plt.close(fig)

# =========================
# AGE vs CHOLESTEROL
# =========================
st.header("🩸 Age vs Cholesterol")

fig, ax = plt.subplots(figsize=(6, 6))

sns.scatterplot(
    x=df["Cholesterol"],
    y=df["Age"],
    alpha=0.6,
    s=12,
    ax=ax
)

sns.kdeplot(
    x=df["Cholesterol"],
    y=df["Age"],
    levels=5,
    linewidths=1,
    ax=ax
)

ax.set_title("Age vs Cholesterol Density", loc="center")
st.pyplot(fig)
plt.close(fig)

# =========================
# CORRELATION HEATMAP
# =========================
st.header("🔗 Correlation Analysis")

numeric_df = df.select_dtypes(include="number")

with st.expander("View Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Correlation Matrix", loc="center")
    st.pyplot(fig)
    plt.close(fig)

# =========================
# AUTOMATIC REPORT
# =========================
st.header("📄 Automatic Analytical Report")

mean_age = df["Age"].mean()
mean_bp = df["BP"].mean()
mean_chol = df["Cholesterol"].mean()

corr_target = (
    numeric_df.corr()["Heart Disease"]
    .sort_values(ascending=False)
    .dropna()
)

report = f"""
## Dataset Summary

- Total patients: {df.shape[0]}
- Heart disease prevalence: {disease_text}

## Clinical Averages
- Average age: {mean_age:.1f} years
- Average blood pressure: {mean_bp:.1f}
- Average cholesterol: {mean_chol:.1f}

## Strongest Correlations with Heart Disease
{corr_target.head(6).to_string()}
"""

st.markdown(report)

st.download_button(
    label="⬇️ Download Report (TXT)",
    data=report,
    file_name="heart_disease_eda_report.txt"
)

# =========================
# FOOTER
# =========================
st.markdown(
    """
---
This exploratory analysis provides a clear overview of cardiovascular risk factors
and serves as a foundation for statistical modeling and predictive analytics.
"""
)
