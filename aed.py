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
    "<h1 style='text-align: center;'> Heart Disease – Exploratory Data Analysis</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
<p style='text-align: center; font-size:16px;'>
This dashboard presents a comprehensive <b>Exploratory Data Analysis (EDA)</b> 
of a heart disease dataset, focusing on clinical distributions, relationships,
and potential cardiovascular risk indicators.
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
st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])

with col2:
    st.write("Preview of the dataset:")
    st.dataframe(df.head(), use_container_width=True)

# =========================
# DESCRIPTIVE STATISTICS
# =========================
with st.expander("📈 Descriptive Statistics"):
    st.dataframe(df.describe(), use_container_width=True)

# =========================
# AGE & BP ANALYSIS
# =========================
st.header("🎂 Age & Blood Pressure Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(df["Age"], bins=30, kde=True, ax=ax)
    ax.set_title("Age Distribution", loc="center")
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
    ax.set_title("Average Blood Pressure by Age", loc="center")
    ax.set_xlabel("Age")
    ax.set_ylabel("Blood Pressure")
    st.pyplot(fig)
    plt.close(fig)

# =========================
# CLINICAL VARIABLES
# =========================
st.header("📊 Distribution of Clinical Variables")

st.markdown(
    "Select clinical variables to visualize their distributions:"
)

selected_columns = st.multiselect(
    "Clinical variables",
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
        ax.set_title(col, loc="center")
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

fig, ax = plt.subplots(figsize=(3, 3))

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

ax.set_title("Age vs Cholesterol Density", loc="center")
ax.set_xlabel("Cholesterol")
ax.set_ylabel("Age")

st.pyplot(fig)
plt.close(fig)

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

ax.set_title("Correlation Heatmap", loc="center")
st.pyplot(fig)
plt.close(fig)

# =========================
# CONCLUSIONS
# =========================
st.header("🧠 Key Insights")

st.markdown(
    """
- The dataset covers a broad age range, supporting demographic analysis.
- Blood pressure and cholesterol present relevant variability across patients.
- Correlation patterns highlight potential predictors of heart disease.
- This EDA provides a solid foundation for **feature engineering** and **machine learning models**.
"""
)
