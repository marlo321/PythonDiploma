import pandas as pd
import pickle
from datetime import timedelta


def predict_end_date_from_csv(data_path, current_stock, bmw_model, car_part):
    """
    Predicts the end date when stock will run out for a given BMW model and car part.

    Parameters:
        data_path (str): Path to the saved model file.
        current_stock (float): Current stock quantity.
        bmw_model (str): The BMW model for prediction.
        car_part (str): The car part for prediction.

    Returns:
        dict: A dictionary containing predictions including the predicted end date.
    """
    # Load the previously saved model
    with open(data_path, 'rb') as file:
        model = pickle.load(file)
        print(f"Model loaded from {data_path}")

    # Prepare the new data
    new_data = pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01']),  # Assume prediction starts from this date
        'BMW Model': [bmw_model],
        'Car Part': [car_part],
    })

    # Feature engineering
    new_data['Year-Month'] = new_data['Date'].dt.to_period('M')
    new_data['Year-Month'] = new_data['Year-Month'].dt.to_timestamp()
    new_data['Time Index'] = (new_data['Year-Month'] - new_data['Year-Month'].min()).dt.days // 30
    new_data['Month'] = new_data['Year-Month'].dt.month

    # One-hot encoding of months (same process as during training)
    seasonal_features = pd.get_dummies(new_data['Month'], prefix='Month', drop_first=True)

    # Add missing month columns to match the trained model's feature set
    all_months = [f'Month_{i}' for i in range(2, 13)]  # Months 2 to 12 (because drop_first=True)
    for month in all_months:
        if month not in seasonal_features.columns:
            seasonal_features[month] = 0  # Add missing month columns as zeros

    # Ensure the correct column order (same as during training)
    seasonal_features = seasonal_features[all_months]

    # Concatenate the new data with the seasonal features
    new_data = pd.concat([new_data, seasonal_features], axis=1)

    # Select the features (same as during model training)
    X_new = new_data[['Time Index'] + all_months]

    # Make predictions using the loaded model
    predictions = model.predict(X_new)
    new_data['Predicted Quantity Used'] = predictions

    # Calculate Predicted End Date
    if current_stock > 0:
        new_data['Predicted End Date'] = new_data['Date'] + pd.to_timedelta(
            current_stock / new_data['Predicted Quantity Used'].values[0] * 30, unit='D'
        )
    else:
        new_data['Predicted End Date'] = None

    # Prepare the results
    # result = {
    #     'Date': new_data['Date'].iloc[0],
    #     'BMW Model': bmw_model,
    #     'Car Part': car_part,
    #     'Predicted Quantity Used': new_data['Predicted Quantity Used'].iloc[0],
    #     'Predicted End Date': new_data['Predicted End Date'].iloc[0],
    # }

    return new_data['Predicted End Date'].iloc[0]

data_path = 'linear_model_with_seasonality.pkl'
current_stock = 200.0
bmw_model = 'Series 3'
car_part = 'Brake Pads'

result = predict_end_date_from_csv(data_path, current_stock, bmw_model, car_part)
print(result)
