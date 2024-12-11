import joblib
import pandas as pd
import numpy as np
import streamlit as st

model, columns = joblib.load("real_estate_model.pkl")

st.title("Прогноз стоимости недвижимости")

total_square = st.number_input("Общая площадь", min_value=0.0, value=50.0)
rooms = st.number_input("Количество комнат", min_value=1, value=2)
floor = st.number_input("Этаж", min_value=1, value=3)

if st.button("Рассчитать стоимость"):
    input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    input_data['total_square'] = total_square
    input_data['rooms'] = rooms
    input_data['floor'] = floor

    prediction = model.predict(input_data)[0]
    st.write(f"Предполагаемая стоимость: {prediction:.2f}")
