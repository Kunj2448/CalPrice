import streamlit as st
import pandas as pd
import subprocess

from src.predictor import predict_single, predict_bulk, get_feature_importance
from src.history import save_history, load_history
from src.metrics import load_metrics

# --------------------
# Page Config
# --------------------
st.set_page_config(
    page_title="CalPrice AI – California Housing Predictor",
    page_icon="🏡",
    layout="wide"
)

# --------------------
# Header
# --------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🏡 CalPrice AI</h1>
    <p style='text-align:center;color:gray;'>
    Smart California Housing Price Prediction System
    </p>
    """,
    unsafe_allow_html=True
)

# --------------------
# Sidebar Inputs
# --------------------
st.sidebar.header("📥 Property Details")

longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.slider("House Age (Years)", 1, 100, 30)
total_rooms = st.sidebar.number_input("Total Rooms", value=1000, min_value=1)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=200, min_value=1)
population = st.sidebar.number_input("Population", value=500, min_value=1)
households = st.sidebar.number_input("Households", value=200, min_value=1)
median_income = st.sidebar.slider("Median Income", 0.0, 15.0, 5.0)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"]
)

predict_btn = st.sidebar.button("🚀 Predict Price")
retrain_btn = st.sidebar.button("🔄 Retrain Model")

# --------------------
# Retrain Button
# --------------------
if retrain_btn:
    with st.spinner("Training model... Please wait ⏳"):
        subprocess.run(["python", "final.py"])
    st.success("✅ Model retrained successfully!")
    st.experimental_rerun()

# --------------------
# Tabs
# --------------------
tab1, tab2, tab3 = st.tabs(
    ["🔮 Prediction", "📂 Bulk Prediction", "📊 Analytics"]
)

# --------------------
# Prediction Tab
# --------------------
with tab1:
    st.subheader("🔮 House Price Prediction")

    metrics = load_metrics()
    rmse = metrics.get("RMSE", 0)

    if predict_btn:
        input_data = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }

        price = predict_single(input_data)
        min_price = round(price - rmse, 2)
        max_price = round(price + rmse, 2)

        save_history({**input_data, "price": price})

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Estimated Price", f"₹ {price}")
        col2.metric("📉 Min Expected", f"₹ {min_price}")
        col3.metric("📈 Max Expected", f"₹ {max_price}")

        st.success("Prediction completed successfully ✅")

# --------------------
# Bulk Prediction Tab
# --------------------
with tab2:
    st.subheader("📂 Bulk CSV Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("⚡ Predict All"):
            result = predict_bulk(df)
            st.dataframe(result.head())

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )

# --------------------
# Analytics Tab
# --------------------
with tab3:
    st.subheader("📊 Model Analytics")

    metrics = load_metrics()
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 R2 Score", metrics.get("R2 Score", "NA"))
        col2.metric("📈 Accuracy (%)", metrics.get("Accuracy (%)", "NA"))
        col3.metric("📉 MAE", metrics.get("MAE", "NA"))
        col4.metric("📊 RMSE", metrics.get("RMSE", "NA"))

    st.divider()

    # Feature Importance
    st.subheader("🔥 Feature Importance")
    fi = get_feature_importance()

    if fi is not None:
        st.dataframe(fi.head(15))
        st.bar_chart(fi.set_index("Feature").head(10))
    else:
        st.warning("Feature importance not available.")

    st.divider()

    # Prediction History
    st.subheader("🕒 Prediction History")
    history = load_history()

    if history is not None and not history.empty:
        st.dataframe(history.tail(30))
        st.line_chart(history["price"])
    else:
        st.info("No prediction history yet.")
