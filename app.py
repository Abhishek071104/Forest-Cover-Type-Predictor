import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# Load model and scaler
model = joblib.load("forest_cover_model.pkl")
scaler = joblib.load("scaler.pkl")

# Forest type mapping
cover_mapping = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# Streamlit Config
st.set_page_config(page_title="Forest Cover Type Predictor", page_icon="🌲", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("### 👤 Made by [M ABHISHEK](https://github.com/Abhishek071104)")
    st.markdown("🔗[GitHub](https://github.com/Abhishek071104)")
    st.markdown("🔗[LinkedIn](https://www.linkedin.com/in/-mabhishek/)")
    st.markdown("📧 [manipatruniabhishek07@gmail.com](mailto:manipatruniabhishek07@gmail.com)")
    st.markdown("---")
    st.markdown("### 🌿 About This App")
    st.write("This Streamlit app uses a trained **Random Forest model** to predict the forest cover type for a land patch based on environmental and terrain features.")

# Title
st.title("🌲 Forest Cover Type Predictor")
st.markdown("Enter environmental and terrain features below to predict the **cover type** of forest land.")

# Session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Layout with tabs
tab1, tab2 = st.tabs(["🔮 Predict", "📜 History"])

with tab1:
    st.subheader("🧭 Terrain and Environmental Features")

    col1, col2, col3 = st.columns(3)
    with col1:
        elevation = st.number_input("🏔️ Elevation (m)", 0, 4000, value=2500)
        aspect = st.number_input("📐 Aspect (degrees)", 0, 360, value=90)
        slope = st.number_input("↘️ Slope (degrees)", 0, 90, value=15)
        hd_hydro = st.number_input("🌊 Horizontal Distance to Hydrology", 0, 10000, value=120)
        vd_hydro = st.number_input("⬆️ Vertical Distance to Hydrology", -500, 500, value=50)

    with col2:
        hd_road = st.number_input("🛣️ Horizontal Distance to Roadways", 0, 10000, value=300)
        hillshade_9am = st.slider("🌞 Hillshade 9am", 0, 255, value=200)
        hillshade_noon = st.slider("🌞 Hillshade Noon", 0, 255, value=220)
        hillshade_3pm = st.slider("🌞 Hillshade 3pm", 0, 255, value=180)
        hd_fire = st.number_input("🔥 Horizontal Distance to Fire Points", 0, 10000, value=250)

    with col3:
        wilderness_area = st.selectbox("🌲 Wilderness Area (0–3)", [0, 1, 2, 3])
        soil_type = st.selectbox("🧱 Soil Type (0–39)", list(range(40)))

    if st.button("🌲 Predict Cover Type"):
        # Build the feature vector
        input_data = [elevation, aspect, slope, hd_hydro, vd_hydro,
                      hd_road, hillshade_9am, hillshade_noon, hillshade_3pm,
                      hd_fire]

        # One-hot encode wilderness and soil type
        wilderness_cols = [1 if i == wilderness_area else 0 for i in range(4)]
        soil_cols = [1 if i == soil_type else 0 for i in range(40)]

        # Final input
        final_input = np.array(input_data + wilderness_cols + soil_cols).reshape(1, -1)
        scaled_input = scaler.transform(final_input)

        # Progress bar
        progress_text = "Predicting forest cover type..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.008)
            my_bar.progress(percent_complete + 1, text=progress_text)

        prediction = model.predict(scaled_input)[0]
        label = cover_mapping[prediction]

        st.success(f"🌳 Predicted Forest Cover Type: **{label}** (Class {prediction})")

        # Save to history
        st.session_state.history.append({
            "Elevation": elevation,
            "Slope": slope,
            "Aspect": aspect,
            "Wilderness": wilderness_area,
            "Soil Type": soil_type,
            "Prediction": label
        })

with tab2:
    st.subheader("📜 Prediction History")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history).iloc[::-1]
        st.dataframe(df_hist, use_container_width=True)

        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download as CSV", csv, "forest_cover_history.csv", "text/csv")

        if st.button("🧹 Clear History"):
            st.session_state.history = []
            st.success("History cleared!")

# Optional banner
st.image("forestcover.jpg", use_container_width=True)
