# 🌲 Forest Cover Type Prediction - Streamlit App

---

This project is a Streamlit web application that predicts the forest cover type for a 30x30 meter patch of land using terrain and environmental data. Trained on real-world data from the Roosevelt National Forest in Colorado, the app leverages a Random Forest Classifier to classify the land into one of seven forest cover types. The interface allows users to input key features like elevation, slope, hillshade values, and distances to hydrology, roads, and fire points. It's built for quick predictions, with an intuitive UI, real-time results, and prediction history tracking.
**.

---

## 🧠 Model
- **Algorithm:** Random Forest Classifier
- **Accuracy:** ~86%
- **Target Classes (Cover Types):**
  1. Spruce/Fir  
  2. Lodgepole Pine  
  3. Ponderosa Pine  
  4. Cottonwood/Willow  
  5. Aspen  
  6. Douglas-fir  
  7. Krummholz

---
##📁 Project Structure
├── app.py                  # Streamlit UI
├── forest_cover_model.pkl  # Trained ML model
├── scaler.pkl              # StandardScaler for input
├── requirements.txt        # Python packages
└── README.md               # Project info
---

