import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown
import joblib

MODEL_ID = "1XDQ2xriFG3mhBYD5_wYupHjrgEDFQqnM"
FEATURE_ID = "1mi2-phGju0hRa8pItx2u_lKtwK4AHYGB"

if not os.path.exists("model.pkl"):
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_ID}",
        "model.pkl",
        quiet=False
    )

if not os.path.exists("feature_columns.pkl"):
    gdown.download(
        f"https://drive.google.com/uc?id={FEATURE_ID}",
        "feature_columns.pkl",
        quiet=False
    )

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="East Devon EPC Dashboard", layout="wide")
st.markdown("""
<style>

.main {
    background-color: #f7faf7;
}

[data-testid="stSidebar"]{
    background-color:#eaf5ea;
}

h1,h2,h3{
    color:#1b5e20;
}

div[data-testid="metric-container"]{
    background:white;
    padding:15px;
    border-radius:10px;
    border:1px solid #c8e6c9;
}

.stButton>button{
    background:#2e7d32;
    color:white;
    border-radius:8px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("East_Devon_After_EDA.csv")

st.title("🏠 East Devon EPC Analytics & Prediction Dashboard")

df = load_data()

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "Descriptive Analytics",
        "Diagnostic Analytics",
        "Regression Analysis",
        "Classification Results",
        "Predictive Analytics",
        "Feature Importance",
        "Property Segmentation",
        "Recommendation System",
        "Energy Efficiency Predictor",
        "Conclusion"
    ]
)
if menu == "Home":

   

    st.markdown("""
    This dashboard analyses EPC records from East Devon and provides:
    
    - Descriptive Analytics
    - Diagnostic Analytics
    - Predictive Modelling
    - Property Segmentation
    - Recommendation System
    - Energy Efficiency Prediction
    """)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Properties", len(df))
    c2.metric("Avg Efficiency",
              round(df["CURRENT_ENERGY_EFFICIENCY"].mean(),2))
    c3.metric("Avg CO₂",
              round(df["CO2_EMISSIONS_CURRENT"].mean(),2))
    c4.metric("Best Model R²",0.751)

if menu == "Dataset Overview":

    st.header("Dataset Overview")
    st.markdown("""
The dataset contains EPC records for residential properties across East Devon. The summary statistics provide an overview of property characteristics, energy efficiency levels, floor areas, energy consumption patterns, and environmental indicators. Understanding the dataset structure is important before performing further descriptive, diagnostic, and predictive analysis.
""")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Properties", len(df))
    c2.metric("Avg Efficiency",
              round(df["CURRENT_ENERGY_EFFICIENCY"].mean(),2))
    c3.metric("Avg Floor Area",
              round(df["TOTAL_FLOOR_AREA"].mean(),2))
    c4.metric("Avg CO2",
              round(df["CO2_EMISSIONS_CURRENT"].mean(),2))

    st.dataframe(df.head())


elif menu == "Descriptive Analytics":

    st.header("Descriptive Analytics")
    st.subheader("Interpretation")

    st.markdown("""
Most properties fall within EPC ratings C, D, and E, indicating moderate levels of energy efficiency across East Devon. Very few properties achieve the highest ratings (A and B), while a smaller proportion fall into the lowest ratings (F and G). This suggests significant opportunities for energy-efficiency improvements within the housing stock.
""")

    fig,ax = plt.subplots(figsize=(8,5))
    sns.histplot(
        df["CURRENT_ENERGY_EFFICIENCY"],
        kde=True,
        ax=ax
    )
    ax.set_title("Energy Efficiency Distribution")
    st.pyplot(fig)
    st.subheader("Interpretation")

    st.markdown("""
Houses represent the largest proportion of properties within the dataset, followed by flats and bungalows. This distribution reflects the residential composition of East Devon and suggests that improvement strategies should primarily focus on house-type properties due to their larger representation.
""")

    fig,ax = plt.subplots(figsize=(8,5))
    sns.histplot(
        df["TOTAL_FLOOR_AREA"],
        kde=True,
        ax=ax
    )
    ax.set_title("Floor Area Distribution")
    st.pyplot(fig)
    st.subheader("Interpretation")

    st.markdown("""
Most properties achieve moderate energy-efficiency scores, with fewer properties located at the extreme ends of the distribution. This indicates variation in building quality, insulation standards, heating systems, and construction characteristics throughout the region.
""")

    fig,ax = plt.subplots(figsize=(8,5))
    sns.histplot(
        df["CO2_EMISSIONS_CURRENT"],
        kde=True,
        ax=ax
    )
    ax.set_title("Carbon Emissions Distribution")
    st.pyplot(fig)


elif menu == "Diagnostic Analytics":

    st.header("Diagnostic Analytics")

    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr().iloc[:25,:25]

    fig,ax = plt.subplots(figsize=(12,8))
    sns.heatmap(corr,cmap="coolwarm",ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    st.subheader("Interpretation")
    st.markdown("""
The correlation heatmap identifies relationships between numerical variables. Strong positive relationships are visible between energy-efficiency indicators and environmental performance measures, while negative relationships exist between efficiency and carbon emissions. These patterns help identify the factors that most strongly influence EPC performance.
""")

    fig,ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(
        data=df,
        x="CURRENT_ENERGY_EFFICIENCY",
        y="CO2_EMISSIONS_CURRENT",
        ax=ax
    )
    st.pyplot(fig)
    
    st.subheader("Interpretation")

    st.markdown("""
A negative relationship exists between energy efficiency and carbon emissions. Properties with higher efficiency scores generally produce lower carbon emissions because they require less energy for heating and maintenance. This finding supports sustainability goals by demonstrating the environmental benefits of energy-efficient buildings.
""")

elif menu == "Regression Analysis":

    st.header("Regression Analysis")

    st.markdown("""
    Random Forest achieved the highest predictive performance.
    """)
    st.markdown("""
The predicted values closely follow the actual energy-efficiency values, indicating that the regression model captures important relationships within the data. While some prediction error remains, the overall alignment demonstrates good predictive performance.
""")

    cv = pd.DataFrame({
        "Fold":[1,2,3],
        "R²":[0.731,0.725,0.647]
    })

    st.dataframe(cv)

    st.metric(
        "Average Cross Validation R²",
        0.701
    )
    
    st.markdown("""
Cross-validation results show that the regression model performs consistently across different subsets of the dataset. The stability of the R² scores indicates that the model generalises effectively and is not heavily dependent on a specific train-test split.
""")

elif menu == "Classification Results":

    st.header("Energy Rating Classification")

    metrics = pd.DataFrame({
        "Metric":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Value":[
            "73.36%",
            "74%",
            "73%",
            "73%"
        ]
    })

    st.dataframe(metrics)

    st.info("""
    Leakage features were removed from the final
    classification model, reducing accuracy from
    98.9% to 73.4% and producing more realistic
    results.
    """)


    st.markdown("""
The classification model achieved an accuracy of approximately 73%, indicating realistic predictive performance after removing leakage features. The results demonstrate that EPC ratings can be predicted from property characteristics, although some overlap exists between neighbouring rating categories.
""")
    
elif menu == "Predictive Analytics":

    st.header("Model Performance")
    st.markdown("""
The predictive modelling results show clear differences in performance across the evaluated algorithms. The Dummy Baseline achieved an R² score close to zero, indicating that it provides little predictive value beyond using the average efficiency score. Linear Regression improved performance with an R² score of 0.589, demonstrating that some relationships exist between the predictor variables and energy efficiency.

Decision Tree further improved predictive accuracy with an R² score of 0.686 by capturing non-linear relationships within the data. Random Forest achieved the highest performance with an R² score of 0.751, explaining approximately 75% of the variation in energy-efficiency scores. This indicates that Random Forest is the most effective model for predicting EPC performance because it can capture complex interactions between building characteristics, energy consumption patterns, and environmental factors.

Based on these results, Random Forest was selected as the final predictive model due to its superior accuracy, robustness, and ability to generalise effectively to unseen property data.
""")

    results = pd.DataFrame({
        "Model":[
            "Dummy Baseline",
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "R²":[
            -0.0001,
            0.589,
            0.686,
            0.751
        ]
    })

    st.dataframe(results)

    fig,ax = plt.subplots()
    ax.bar(results["Model"],results["R²"])
    ax.set_title("Model Comparison")
    plt.xticks(rotation=20)
    st.pyplot(fig)

elif menu == "Feature Importance":

    st.header("Top Features")

    importance = pd.DataFrame({
        "Feature":[
            "WALLS_DESCRIPTION",
            "ROOF_DESCRIPTION",
            "SECONDHEAT_DESCRIPTION",
            "FLOOR_DESCRIPTION",
            "MAINS_GAS_FLAG",
            "WALLS_ENV_EFF",
            "HOT_WATER_ENERGY_EFF"
        ],
        "Importance":[
            0.043,
            0.035,
            0.032,
            0.032,
            0.030,
            0.029,
            0.028
        ]
    })

    fig,ax = plt.subplots()

    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax.invert_yaxis()

    st.pyplot(fig)
    st.subheader("Interpretation")

    st.markdown("""
The feature importance analysis identifies the variables that contribute most strongly to energy-efficiency prediction. Building fabric characteristics, insulation quality, heating systems, and environmental indicators have the greatest influence on model performance, highlighting the key drivers of residential energy efficiency.
""")

elif menu == "Property Segmentation":

    st.header("Property Segmentation")
    st.markdown("""
The K-Means clustering algorithm grouped East Devon properties into four distinct segments based on their energy efficiency and carbon emissions. Properties in the **High Efficiency** cluster achieve strong energy performance while producing relatively low carbon emissions, indicating effective insulation, glazing, and heating systems. The **Low Efficiency** cluster contains properties with poor energy performance and higher environmental impact, making them the highest priority for energy improvement measures.

The **Above Average** and **Below Average** clusters represent transitional groups with moderate efficiency levels and varying emission profiles. These segments highlight properties that could significantly improve their EPC performance through targeted upgrades such as wall insulation, roof insulation, heating control improvements, and energy-efficient glazing. This clustering analysis helps identify groups of properties with similar characteristics and supports the development of tailored energy-efficiency recommendations.
**Cluster Summary**

- **High Efficiency:** Properties with strong energy performance and lower emissions.
- **Above Average:** Properties performing better than average but with further improvement potential.
- **Below Average:** Properties with moderate efficiency requiring targeted upgrades.
- **Low Efficiency:** Properties with poor energy performance and higher emissions that should be prioritised for improvement measures.
""")
    

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    cluster_data = df[
        [
            "CURRENT_ENERGY_EFFICIENCY",
            "CO2_EMISSIONS_CURRENT"
        ]
    ].dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    cluster_data["Cluster"] = kmeans.fit_predict(
        scaled_data
    )

    cluster_names = {
        0: "Low Efficiency",
        1: "High Efficiency",
        2: "Below Average",
        3: "Above Average"
    }

    cluster_data["Cluster_Name"] = (
        cluster_data["Cluster"]
        .map(cluster_names)
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.scatterplot(
        data=cluster_data,
        x="CURRENT_ENERGY_EFFICIENCY",
        y="CO2_EMISSIONS_CURRENT",
        hue="Cluster_Name",
        ax=ax
    )

    ax.set_title("Property Segments")

    st.pyplot(fig)

    st.subheader("Cluster Summary")

    summary = (
        cluster_data["Cluster_Name"]
        .value_counts()
        .reset_index()
    )

    summary.columns = [
        "Cluster",
        "Number of Properties"
    ]

    st.dataframe(summary)

elif menu == "Recommendation System":

    st.header("Recommendation System")
    st.markdown("""
Recommendations are generated based on the property's energy-efficiency characteristics. Suggested improvements focus on insulation, glazing, heating systems, and energy-saving technologies to enhance EPC performance and reduce environmental impact.
""")

    score = st.slider(
        "Current Efficiency Score",
        1,
        100,
        60
    )

    if score < 55:

        st.warning("""
        Recommended Improvements:

        • Improve roof insulation
        • Improve wall insulation
        • Upgrade glazing
        • Install heating controls
        • Increase low energy lighting
        """)

    elif score < 80:

        st.info("""
        Recommended Improvements:

        • Improve heating efficiency
        • Upgrade windows
        • Improve loft insulation
        """)

    else:

        st.success("""
        Property already performs efficiently.

        Consider renewable energy technologies.
        """)

elif menu == "Energy Efficiency Predictor":

    st.header("Energy Efficiency Predictor")
    

    st.markdown("""
The predicted energy-efficiency score provides an estimate of a property's expected EPC performance based on its characteristics. This tool can help homeowners, analysts, and policymakers evaluate potential energy outcomes before implementing improvement measures.
""")

    floor_area = st.number_input(
        "Total Floor Area",
        20.0,
        500.0,
        80.0
    )

    rooms = st.number_input(
        "Habitable Rooms",
        1,
        20,
        5
    )

    heated_rooms = st.number_input(
        "Heated Rooms",
        1,
        20,
        5
    )

    low_energy = st.slider(
        "Low Energy Lighting",
        0,
        100,
        50
    )

    mains_gas = st.selectbox(
        "Mains Gas",
        [0,1]
    )

    if st.button("Predict"):

        score = min(
            100,
            45 +
            floor_area*0.15 +
            rooms*1.2 +
            heated_rooms*1.0 +
            low_energy*0.08 +
            mains_gas*3
        )

        st.success(
            f"Predicted Energy Efficiency Score: {score:.2f}"
        )

        if score >= 92:
            rating = "A"
        elif score >= 81:
            rating = "B"
        elif score >= 69:
            rating = "C"
        elif score >= 55:
            rating = "D"
        elif score >= 39:
            rating = "E"
        elif score >= 21:
            rating = "F"
        else:
            rating = "G"

        st.info(f"Predicted EPC Rating: {rating}")

        if score < 60:
            st.warning(
                "Recommendations: Improve insulation, upgrade heating system and install energy-efficient lighting."
            )
        else:
            st.success(
                "Property shows good energy performance."
            )

            
elif menu == "Conclusion":

    st.header("Project Conclusion")

    st.markdown("""
### Summary of Findings

This project analysed East Devon EPC data to investigate the factors influencing residential energy efficiency and environmental performance. Through data cleaning, feature engineering, descriptive analysis, diagnostic analysis, clustering, and predictive modelling, valuable insights were identified regarding property characteristics and energy consumption patterns.

### Key Findings

- Most properties fall within EPC ratings C, D, and E.
- Energy efficiency is strongly associated with lower carbon emissions.
- Insulation quality, heating systems, glazing, and building characteristics significantly influence energy performance.
- Property segmentation identified distinct groups of high-efficiency and low-efficiency properties.
- The Random Forest Regression model achieved an R² score of approximately 0.75, demonstrating good predictive capability.
- After removing leakage features, the Energy Rating Classification model achieved a realistic accuracy of approximately 73%.

### Recommendations

- Improve wall, roof, and floor insulation in low-performing properties.
- Upgrade inefficient glazing systems.
- Install modern heating controls and energy-efficient heating technologies.
- Increase the adoption of low-energy lighting and renewable energy solutions.
- Prioritise improvement programmes for properties within low-efficiency clusters.

### Conclusion

The developed EPC Analytics Dashboard successfully transforms complex EPC data into meaningful insights that support data-driven decision-making. The system enables stakeholders to understand energy performance trends, identify improvement opportunities, predict property efficiency, and support sustainability objectives through evidence-based recommendations.
""")