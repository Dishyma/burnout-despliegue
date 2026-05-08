# Despliegue - Prediccion de riesgo de burnout

Interfaz grafica con Streamlit para predecir el nivel de riesgo de burnout en trabajadores remotos.

## Ejecucion local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modelo

El modelo es un Arbol de Decision hiperparametrizado con GridSearchCV, entrenado con los datos de burnout limpios.
