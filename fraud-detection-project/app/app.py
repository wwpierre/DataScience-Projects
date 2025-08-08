import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загрузка модели
model = joblib.load("/content/fraud-detection-project/models/xgb_model.pkl")

# Интерфейс
st.title("🚨 Fraud Detection App")
st.write("Введите параметры транзакции, чтобы узнать, является ли она мошеннической.")

# Ввод признаков
features = {}
for i in range(1, 29):
    features[f"V{i}"] = st.slider(f"V{i}", -20.0, 20.0, 0.0, step=0.1)
features["Amount"] = st.number_input("Сумма транзакции", 0.0, 10000.0, 100.0)
features["Time"] = st.number_input("Время с начала наблюдения (сек)", 0.0, 200000.0, 10000.0)

# Прогноз
if st.button("Проверить транзакцию"):
    input_df = pd.DataFrame([features])

# Упорядочим признаки в том же порядке, как при обучении
    column_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = input_df[column_order]

# Предсказание
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ Внимание! Транзакция подозрительная (вероятность: {prob:.2f})")
    else:
        st.success(f"✅ Транзакция выглядит нормальной (вероятность фрода: {prob:.2f})")
