import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Datos de la canasta básica
data = {
    'Mes': ['ene-23', 'feb-23', 'mar-23', 'abr-23', 'may-23', 'jun-23', 'jul-23', 'ago-23', 'sep-23', 'oct-23', 'nov-23', 'dic-23', 'ene-24', 'feb-24'],
    'Canasta básica alimentaria': [23315, 26046, 28388, 30469, 32056, 33731, 36130, 42262, 47858, 51975, 59887, 77890, 92415, 104483]
}

# Convertir las fechas a objetos datetime
months = {'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12}
data['Mes'] = [datetime(2000 + int(date.split('-')[1]), months[date.split('-')[0]], 1) for date in data['Mes']]

# Ajustar el modelo ARIMA
model = ARIMA(data['Canasta básica alimentaria'], order=(5,1,0))
model_fit = model.fit()

# Realizar la predicción para los próximos 6 meses
future_months = [data['Mes'][-1] + pd.DateOffset(months=i) for i in range(1, 7)]
forecast = model_fit.forecast(steps=6)

# Visualizar la evolución de la Canasta Básica Alimentaria y la predicción
plt.figure(figsize=(10, 6))
plt.plot(data['Mes'], data['Canasta básica alimentaria'], marker='o', color='b', label='Historia')
plt.plot(future_months, forecast, marker='o', color='r', label='Predicción')
plt.title('Predicción de la Canasta Básica Alimentaria en Argentina')
plt.xlabel('Mes')
plt.ylabel('Valor en $')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
