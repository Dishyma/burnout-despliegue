import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.title('Predicción de riesgo de burnout')
st.write('Ingrese los datos del trabajador para predecir el nivel de riesgo de burnout.')

filename = 'modelo_burnout.pkl'
modelo, labelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

work_hours = st.slider('Horas de trabajo', min_value=3.0, max_value=13.0, value=8.0, step=0.1)
meetings_count = st.slider('Cantidad de reuniones', min_value=0, max_value=10, value=2, step=1)
task_completion_rate = st.slider('Porcentaje de tareas completadas', min_value=0.0, max_value=100.0, value=75.0, step=0.1)
burnout_score = st.slider('Puntaje de burnout', min_value=0.0, max_value=100.0, value=40.0, step=0.1)

datos = [[work_hours, meetings_count, task_completion_rate, burnout_score]]
data = pd.DataFrame(datos, columns=['work_hours', 'meetings_count', 'task_completion_rate', 'burnout_score'])

data_preparada = data.copy()
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

if st.button('Predecir'):
    Y_pred = modelo.predict(data_preparada)
    prediccion = labelencoder.inverse_transform(Y_pred)
    data['Prediccion'] = prediccion
    st.write(data)
    st.success('Riesgo de burnout: ' + str(prediccion[0]))
    st.warning('El modelo tiene una exactitud del 98.52% y un error aproximado del 1.48%')
