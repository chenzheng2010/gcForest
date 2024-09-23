
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('GCF1.pkl')
import matplotlib
matplotlib.use('TkAgg')

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
Gray_correlation = st.number_input("Gray_correlation:", min_value=0.10, max_value=1.00, value=0.99)
# R_contrast: numerical input
R_contrast = st.number_input("R_contrast:", min_value=4.4, max_value=16.2, value=5.0)
# R_dissimilarity: numerical input
R_dissimilarity = st.number_input("R_dissimilarity:", min_value=0.71, max_value=1.21, value=0.85)
# R_correlation: numerical input
R_correlation = st.number_input("R_correlation:", min_value=0.10, max_value=1.00, value=0.99)
# R_entropy: numerical input
R_entropy = st.number_input("R_entropy:", min_value=4.0, max_value=5.3, value=4.5)
# G_contrast: numerical input
G_contrast = st.number_input("G_contrast:", min_value=6.1, max_value=20.4, value=10.0)
# G_dissimilarity: numerical input
G_dissimilarity = st.number_input("G_dissimilarity:", min_value=0.69, max_value=1.12, value=0.72)
# G_homogeneity: numerical input
G_homogeneity = st.number_input("G_homogeneity:", min_value=0.64, max_value=0.76, value=0.65)
# G_correlation: numerical input
G_correlation = st.number_input("G_correlation:", min_value=0.10, max_value=1.00, value=0.99)
# B_dissimilarity: numerical input
B_dissimilarity = st.number_input("B_dissimilarity:", min_value=0.90, max_value=1.37, value=0.95)
# B_correlation: numerical input
B_correlation = st.number_input("B_correlation:", min_value=0.10, max_value=1.00, value=0.99)
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
    predicted_class = model.predict(features)
    predicted_proba = model.predict_proba(features)

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (2: 过熟, 1: 适熟, 0: 欠熟)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability0 = predicted_proba[:, 0] * 100
    probability1 = predicted_proba[:, 1] * 100
    probability2 = predicted_proba[:, 2] * 100

    if predicted_class == 0:
        advice = (
            f"中部叶的叶面60%～70%黄绿色，主脉变白1/2左右；上部叶的叶面70%～80%浅黄色，主脉变白2/3 左右。"
            f"模型预测该烟叶样本为欠熟档次的概率是{probability0:.1f}%。"
            "建议延时田间采收烘烤。"
        )
    elif predicted_class == 1:
        advice = (
            f"中部叶的叶面70%～80%浅黄色，主脉变白2/3左右；上部叶的叶面80%～90%浅黄色，主脉变白3/4左右。"
            f"模型预测该烟叶样本为欠熟档次的概率是{probability1:.1f}%。"
            "建议及时进行田间采收烘烤。"
        )
    elif predicted_class == 2:
        advice = (
            f"中部叶的叶面90%～100%黄色，主脉全白；上部叶的叶面90%～100%黄色，主脉全白。"
            f"模型预测该烟叶样本为欠熟档次的概率是{probability2:.1f}%。"
            "建议提前进行采烤。"
        )

    st.write(advice)


    # 使用 SHAP 解释模型
    # 使用一个小的子集作为背景数据（可以是Xtest的一个子集）
    background = shap.sample(X_test, 20)
    # explainer = shap.KernelExplainer(model.predict_proba, X_test.iloc[0:10, :])
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # Display the SHAP force plot for the predicted class
    if predicted_class == 0:
        shap.force_plot(explainer.expected_value[0], shap_values[:, :, 0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    elif predicted_class == 1:
        shap.force_plot(explainer.expected_value[1], shap_values[:, :, 1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    elif predicted_class == 2:
        shap.force_plot(explainer.expected_value[2], shap_values[:, :, 2], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')



    # # 计算测试集的shap值, 限制前50个训练样本是因为计算所有样本时间太长, 这里自己定义用多少个样本或者用全部 运行速度相关 我使用了20个样本
    # shap_values = explainer.shap_values(X=X_test.iloc[0:20, :], nsamples=100)
    # shap_values2 = explainer(X=X_test.iloc[0:20, :])
    
    # pd.DataFrame(feature_values).shape
    # print("shap值维度;", shap_values.shape)
    # shap_values.shape
    # pd.DataFrame([feature_values], columns=feature_names)
    

    # # 绘制 SHAP 总结图
    # plt.figure()
    # plt.title('Class 0 SHAP Summary')
    # shap.summary_plot(shap_values_class_0, X_test, plot_type="dot", cmap="viridis")
    # # plt.savefig("Class 0 SHAP Summary.pdf", format='pdf', bbox_inches='tight')
    # plt.savefig("Class 0 SHAP Summary.png", bbox_inches='tight', dpi=1200)
    # st.image("Class 0 SHAP Summary.png", caption='SHAP Force Plot Explanation')
    #
    # plt.figure()
    # plt.title('Class 1 SHAP Summary')
    # shap.summary_plot(shap_values_class_1, X_test, plot_type="dot", cmap="viridis")
    # plt.savefig("Class 1 SHAP Summary.png", bbox_inches='tight', dpi=1200)
    # st.image("Class 1 SHAP Summary.png", caption='SHAP Force Plot Explanation')
    #
    # plt.figure()
    # plt.title('Class 2 SHAP Summary')
    # shap.summary_plot(shap_values_class_2, X_test, plot_type="dot", cmap="viridis")
    # plt.savefig("Class 2 SHAP Summary.png", bbox_inches='tight', dpi=1200)
    # st.image("Class 2 SHAP Summary.png", caption='SHAP Force Plot Explanation')
