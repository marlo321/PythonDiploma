import pandas as pd
import pickle
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_model_and_save_from_csv(data_path, model_save_path):
    """
    Trains SARIMAX models for each BMW model and car part combination and saves them to a file.

    Args:
        data_path (str): Path to the CSV file.
        model_save_path (str): Path to save the trained models.
    """
    # Load the dataset
    data = pd.read_csv(data_path)

    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Group the dataset by 'BMW Model' and 'Car Part'
    grouped_data = data.groupby(['BMW Model', 'Car Part'])
    models = {}

    for (bmw_model, car_part), group in grouped_data:
        # Resample daily, sum quantities, and fill missing days with zero
        group = group.resample('D').sum(numeric_only=True).fillna(0)
        group['DayOfWeek'] = group.index.dayofweek

        # Train a SARIMAX model
        model = SARIMAX(
            group['Quantity Used'],
            exog=group[['DayOfWeek']],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)

        # Save the trained model with its keys
        models[(bmw_model, car_part)] = model_fit

    # Save all trained models to the specified file
    with open(model_save_path, 'wb') as f:
        pickle.dump(models, f)


def predict_end_date_from_csv(model_path, current_stock, bmw_model, car_part):
    """
    Predicts the date when the current stock will run out using a trained SARIMAX model.

    Args:
        model_path (str): Path to the saved model file.
        current_stock (float): Current stock level.
        bmw_model (str): The BMW model to predict for.
        car_part (str): The car part to predict for.

    Returns:
        str: Predicted end date or a message indicating stock longevity.
    """
    # Load trained models
    with open(model_path, 'rb') as f:
        models = pickle.load(f)

    # Retrieve the specific model for the given BMW model and car part
    model_key = (bmw_model, car_part)
    if model_key not in models:
        return "No model found for the specified BMW model and car part."

    model_fit = models[model_key]

    # Generate predictions for the next 30 days
    future_days = pd.DataFrame({
        'DayOfWeek': [(datetime.now() + timedelta(days=i)).weekday() for i in range(30)]
    })

    # Forecast and calculate cumulative consumption
    forecast = model_fit.get_forecast(steps=30, exog=future_days).predicted_mean
    cumulative_consumption = forecast.cumsum()

    # Determine when the stock will run out
    for i, consumption in enumerate(cumulative_consumption):
        if consumption >= current_stock:
            end_date = datetime.now() + timedelta(days=i)
            return end_date.strftime("%Y-%m-%d")

    return "Stock will last for more than 30 days."


# Usage example
data_path = 'bmw_car_parts_usage_synthetic.csv'
model_save_path = 'sarimax_models.pkl'

# Train models and save them
train_model_and_save_from_csv(data_path, model_save_path)

# Predict end date
current_stock = 100
bmw_model = 'BMW 3 Series'
car_part = 'Brake Pads'

result = predict_end_date_from_csv(model_save_path, current_stock, bmw_model, car_part)
print(result)
