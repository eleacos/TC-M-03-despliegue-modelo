# Despliegue del modelo de ML "Hotel Booking - Predicción de Cancelaciones de Reservas" mediante API REST

---

## 🇪🇸 Español

### Descripción del proyecto

Este proyecto tiene como objetivo desplegar un modelo de Machine Learning capaz de predecir si una reserva hotelera será cancelada o no. 

El modelo ha sido expuesto mediante una API REST desarrollada con FastAPI y desplegada en Render, permitiendo su consumo desde cualquier cliente (Python, web, etc.)

---

### Modelo de Machine Learning
El desarrollo completo del modelo (EDA, feature engineering, entrenamiento y evaluación) se encuentra en el siguiente repositorio: 
https://github.com/eleacos/ML_hotel_booking.git

El modelo final utilizada es un **Random Forest optimizado** serializado en formato '.joblib'.  

---

### Estructura del repositorio
├── src/
│   ├── data/            # Muestra del dataset
│   ├── img/             # Visualizaciones
│   ├── models/          # Modelos entrenados (.joblib)
│   ├── notebooks/       # EDA y modelado
|   ├── tests/           # Pruebas y notebooks de testeo
│   └── utils/           # Funciones auxiliares
├── main.py/             # Código de la API (FastAPI)
├── tests/               # Pruebas y notebooks de testeo
├── test_api_fast.ipynb  # Consumo de la API
└── README.md

---

### API REST
La API permite realizar predicciones a partir de datos de entrada estructurados. 

***URL base***: https://tc-m-03-despliegue-modelo-hu9q.onrender.com 

La API incluye **documentación automática** genera con FastAPI: https://tc-m-03-despliegue-modelo-hu9q.onrender.com/docs

Desde esta interfaz puedes: 
- Ver todos los endpoints disponibles
- Consultar los parámetros requeridos
- Probar la API directamente desde el navegador

---

### Endpoint principal: ´POST/predict´
Permite obtener una predicción de cancelación.

---

### Autores

**Brenda Oyola** — https://github.com/Brendaluoyola
**Diana Hoyos** — https://github.com/dianahoyos
**Elena Acosta** — https://github.com/eleacos