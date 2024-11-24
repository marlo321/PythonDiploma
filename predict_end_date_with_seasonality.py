import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Завантаження даних
data = pd.read_csv('bmw_car_parts_usage_synthetic.csv')

# Перетворення колонки Date у формат datetime
data['Date'] = pd.to_datetime(data['Date'])

# Додавання стовпця "Year-Month" для групування за місяцями
data['Year-Month'] = data['Date'].dt.to_period('M')

# Групування даних: сума використання запчастин за місяцем та назвою запчастини
grouped_data = data.groupby(['Year-Month', 'Car Part'])['Quantity Used'].sum().reset_index()

# Перетворення "Year-Month" у формат datetime
grouped_data['Year-Month'] = grouped_data['Year-Month'].dt.to_timestamp()

# Додавання трендових ознак: індекс часу
grouped_data['Time Index'] = (grouped_data['Year-Month'] - grouped_data['Year-Month'].min()).dt.days // 30

# Додавання сезонних ознак: місяць
grouped_data['Month'] = grouped_data['Year-Month'].dt.month

# One-hot кодування для сезонності (місяців)
seasonal_features = pd.get_dummies(grouped_data['Month'], prefix='Month', drop_first=True)
grouped_data = pd.concat([grouped_data, seasonal_features], axis=1)

# Вибір цільової змінної (Quantity Used) і ознак (Time Index та місяці)
X = grouped_data[['Time Index'] + [col for col in grouped_data.columns if col.startswith('Month_')]]
y = grouped_data['Quantity Used']

# Розділення на навчальні й тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація моделі
model = LinearRegression()

# Навчання на тренувальних даних
model.fit(X_train, y_train)

# Прогноз на тестових даних
y_pred = model.predict(X_test)

# Обчислення середньоквадратичної помилки
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Збереження моделі у файл
model_filename = 'linear_model_with_seasonality.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_filename}")