
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('GCF1.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test.csv')

# Define feature names from the new dataset
feature_names = [
    "Length", "L_W_Ratio", "Area", "Perimeter", "Roundness", "R_mean", "R_std", "B_std", "a_mean",
    "a_std", "b_mean", "b_std", "H_mean", "H_std", "S_mean", "Gray_contrast", "Gray_dissimilarity",
    "Gray_homogeneity", "Gray_correlation", "R_contrast", "R_dissimilarity", "R_correlation",
    "R_entropy", "G_contrast", "G_dissimilarity", "G_homogeneity", "G_correlation", "B_dissimilarity",
    "B_correlation", "B_entropy"
]

# Streamlit user interface
st.title("烟叶成熟度判别")

# Length: numerical input
Length = st.number_input("Length:", min_value=417, max_value=745, value=500)
# L_W_Ratio: numerical input
L_W_Ratio = st.number_input("L_W_Ratio:", min_value=1.40, max_value=3.53, value=1.52)
# Area: numerical input
Area = st.number_input("Area:", min_value=41156, max_value=157665, value=70958)
# Perimeter: numerical input
Perimeter = st.number_input("Perimeter:", min_value=1061, max_value=2073, value=1500)
# Roundness: numerical input
Roundness = st.number_input("Roundness:", min_value=0.31, max_value=0.62, value=0.50)
# R_mean: numerical input
R_mean = st.number_input("R_mean:", min_value=76, max_value=204, value=100)
# R_std: numerical input
R_std = st.number_input("R_std:", min_value=10, max_value=29, value=12)
# B_std: numerical input
B_std = st.number_input("B_std:", min_value=10.5, max_value=32.9, value=12.0)
# a_mean: numerical input
a_mean = st.number_input("a_mean:", min_value=0.5, max_value=12.0, value=1.0)
# a_std: numerical input
a_std = st.number_input("a_std:", min_value=0.5, max_value=6.0, value=2.0)
# b_mean: numerical input
b_mean = st.number_input("b_mean:", min_value=24.3, max_value=52.7, value=30.0)
# b_std: numerical input
b_std = st.number_input("b_std:", min_value=3.0, max_value=11.9, value=5.0)
# H_mean: numerical input
H_mean = st.number_input("H_mean:", min_value=26.2, max_value=48.8, value=30.0)
# H_std: numerical input
H_std = st.number_input("H_std:", min_value=2.6, max_value=7.5, value=3.0)
# S_mean: numerical input
S_mean = st.number_input("S_mean:", min_value=80.7, max_value=172.8, value=100.0)
# Gray_contrast: numerical input
Gray_contrast = st.number_input("Gray_contrast:", min_value=5.59, max_value=19.05, value=10.00)
# Gray_dissimilarity: numerical input
Gray_dissimilarity = st.number_input("Gray_dissimilarity:", min_value=0.68, max_value=1.13, value=0.80)
# Gray_homogeneity: numerical input
Gray_homogeneity = st.number_input("Gray_homogeneity:", min_value=0.64, max_value=0.76, value=0.65)
# Gray_correlation: numerical input
Gray_correlation = st.number_input("Gray_correlation:", min_value=0.9982, max_value=0.9985, value=0.9983)
# R_contrast: numerical input
R_contrast = st.number_input("R_contrast:", min_value=4.4, max_value=16.2, value=5.0)
# R_dissimilarity: numerical input
R_dissimilarity = st.number_input("R_dissimilarity:", min_value=0.71, max_value=1.21, value=0.85)
# R_correlation: numerical input
R_correlation = st.number_input("R_correlation:", min_value=0.9979, max_value=0.9996, value=0.9980)
# R_entropy: numerical input
R_entropy = st.number_input("R_entropy:", min_value=4.0, max_value=5.3, value=4.5)
# G_contrast: numerical input
G_contrast = st.number_input("G_contrast:", min_value=6.1, max_value=20.4, value=10.0)
# G_dissimilarity: numerical input
G_dissimilarity = st.number_input("G_dissimilarity:", min_value=0.69, max_value=1.12, value=0.72)
# G_homogeneity: numerical input
G_homogeneity = st.number_input("G_homogeneity:", min_value=0.64, max_value=0.76, value=0.65)
# G_correlation: numerical input
G_correlation = st.number_input("G_correlation:", min_value=0.9986, max_value=0.9995, value=0.9987)
# B_dissimilarity: numerical input
B_dissimilarity = st.number_input("B_dissimilarity:", min_value=0.90, max_value=1.37, value=0.95)
# B_correlation: numerical input
B_correlation = st.number_input("B_correlation:", min_value=0.9906, max_value=0.9987, value=0.9910)
# S_mean: numerical input
B_entropy = st.number_input("B_entropy:", min_value=3.86, max_value=5.33, value=4.00)

# Process inputs and make predictions
feature_values = [
    Length, L_W_Ratio, Area, Perimeter, Roundness, R_mean, R_std, B_std, a_mean,
    a_std, b_mean, b_std, H_mean, H_std, S_mean, Gray_contrast, Gray_dissimilarity,
    Gray_homogeneity, Gray_correlation, R_contrast, R_dissimilarity, R_correlation,
    R_entropy, G_contrast, G_dissimilarity, G_homogeneity, G_correlation, B_dissimilarity,
    B_correlation, B_entropy
]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (2: guoshu, 1: shishu, 0: qianshu)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )

    st.write(advice)
