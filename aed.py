import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Heart Disease EDA",
    layout="wide"
)

# =========================
# TITLE & INTRO
# =========================
st.markdown(
    "<h1 style='text-align: center;'>Heart Disease – Exploratory Data Analysis</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
<p style='text-align: center; font-size:16px;'>
This dashboard presents a detailed <b>Exploratory Data Analysis (EDA)</b> of a heart disease dataset.
The objective is to understand the <b>statistical behavior</b>, <b>clinical variability</b>, and 
<b>relationships among cardiovascular risk factors</b>, providing a solid analytical foundation 
for predictive modeling.
</p>
""",
    unsafe_allow_html=True
)

st.divider()

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
st.header("📊 Statistical Overview of the Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Observations", df.shape[0])
    st.metric("Total Variables", df.shape[1])
    st.dataframe("Data Types", df.info())

with col2:
    st.markdown("**Dataset preview:**")
    st.dataframe(df.head(), use_container_width=True)

# =========================
# DESCRIPTIVE STATISTICS
# =========================
st.header("📈 Descriptive Statistical Analysis")

st.markdown(
    """
Descriptive statistics summarize central tendency, dispersion, and value ranges
for numerical variables, enabling early detection of variability and outliers
in clinical measurements.
"""
)

with st.expander("View Descriptive Statistics"):
    st.dataframe(df.describe(), use_container_width=True)

# =========================
# AGE & BP ANALYSIS
# =========================
st.header("🎂 Age & Blood Pressure Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(df["Age"], bins=30, kde=True, ax=ax)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(
        x="Age",
        y="BP",
        estimator="mean",
        data=df,
        ax=ax
    )
    ax.set_title("Average Blood Pressure by Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Blood Pressure")
    st.pyplot(fig)
    plt.close(fig)

st.info(
    "🔎 **Interpretation:** Age presents a broad distribution, and average blood pressure "
    "tends to vary with age, reinforcing its relevance as a cardiovascular risk factor."
)

# =========================
# AGE vs HEART DISEASE (BOXPLOT)
# =========================
st.header("❤️ Age Distribution by Heart Disease Outcome")

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(
    x='Heart Disease',
    y='Age',
    data=df,
    ax=ax
)

ax.set_xlabel("Heart Disease (0 = No, 1 = Yes)")
ax.set_ylabel("Age")
ax.set_title("Age Distribution by Heart Disease Status")

st.pyplot(fig)
plt.close(fig)

st.info(
    "🔎 **Interpretation:** Patients diagnosed with heart disease tend to show a higher "
    "median age, suggesting age as a strong contributing factor in cardiovascular risk."
)

# =========================
# CLINICAL VARIABLES
# =========================
st.header("📊 Distribution of Clinical Variables")

selected_columns = st.multiselect(
    "Select clinical variables:",
    options=[
        'Age', 'BP', 'Cholesterol', 'Max HR',
        'ST depression', 'Slope of ST',
        'Chest pain type', 'Thallium', 'Heart Disease'
    ],
    default=['Age', 'BP', 'Cholesterol', 'Max HR']
)

if selected_columns:
    n_cols = 3
    rows = (len(selected_columns) + n_cols - 1) // n_cols

    fig, axs = plt.subplots(rows, n_cols, figsize=(14, rows * 3))
    axs = axs.flatten()

    for ax, col in zip(axs, selected_columns):
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")

    for ax in axs[len(selected_columns):]:
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =========================
# AGE vs CHOLESTEROL
# =========================
st.header("🩸 Age vs Cholesterol Relationship")

fig, ax = plt.subplots(figsize=(6, 6))

sns.scatterplot(
    x=df["Cholesterol"],
    y=df["Age"],
    s=10,
    alpha=0.6,
    ax=ax
)

sns.kdeplot(
    x=df["Cholesterol"],
    y=df["Age"],
    levels=6,
    linewidths=1,
    ax=ax
)

ax.set_title("Age vs Cholesterol Density")
ax.set_xlabel("Cholesterol")
ax.set_ylabel("Age")

st.pyplot(fig)
plt.close(fig)

st.info(
    "🔎 **Interpretation:** Cholesterol values are widely distributed across age groups, "
    "with higher-density regions indicating common patient profiles rather than a strict linear trend."
)

# =========================
# BP vs CHOLESTEROL BINS (VIOLIN + HUE)
# =========================
st.header("🩺 Blood Pressure Across Cholesterol Levels and Heart Disease")

bp_age = df[['Cholesterol', 'BP', 'Heart Disease']].copy()
bp_age['Cholesterol_bins'] = pd.cut(
    bp_age['Cholesterol'],
    bins=5,
    labels=False
)

fig, ax = plt.subplots(figsize=(7, 4))
sns.violinplot(
    x='Cholesterol_bins',
    y='BP',
    hue='Heart Disease',
    data=bp_age,
    split=True,
    ax=ax
)

ax.set_xlabel("Cholesterol Bins (Low → High)")
ax.set_ylabel("Blood Pressure")
ax.set_title("Blood Pressure Distribution by Cholesterol Level and Heart Disease")

st.pyplot(fig)
plt.close(fig)

st.info(
    "🔎 **Interpretation:** Across higher cholesterol bins, patients with heart disease "
    "tend to exhibit greater blood pressure dispersion and higher central values, "
    "suggesting a combined effect of cholesterol and blood pressure on cardiovascular risk."
)

# =========================
# CORRELATION HEATMAP
# =========================
st.header("🔗 Correlation Analysis")

numeric_df = df.select_dtypes(include="number")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax
)

ax.set_title("Correlation Heatmap")
st.pyplot(fig)
plt.close(fig)

st.info(
    "🔎 **Interpretation:** Several clinical variables show moderate correlations, "
    "highlighting potential predictors while reinforcing the need to address multicollinearity."
)

# =========================
# CONCLUSIONS
# =========================
st.header("🧠 Key Insights and Conclusions")

st.markdown(
    """
- Age, blood pressure, and cholesterol demonstrate meaningful variability across patients.
- Visual comparisons reveal clear differences between patients with and without heart disease.
- Combined distribution analyses strengthen clinical interpretability.
- This EDA establishes a strong foundation for **feature engineering**, **classification models**, 
  and **cardiovascular risk prediction**.
"""
)

st.markdown(
    """
---
This project demonstrates how structured exploratory analysis supports
data-driven insights in healthcare analytics.
"""
)
