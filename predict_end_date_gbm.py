import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime, timedelta
import numpy as np

def train_model_and_save_from_csv(csv_path, save_path="stock_model.pkl"):
    """
    Loads data from a CSV file, trains a Gradient Boosting model, and saves it.

    Args:
        csv_path (str): Path to the CSV file.
        save_path (str): Path to save the trained model.
    """
    # Load data
    data = pd.read_csv(csv_path)

    # Ensure correct column names
    required_columns = ['Date', 'Day of Week', 'Day of Month', 'Month', 'Quantity Used', 'BMW Model', 'Car Part']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"The dataset must contain the following columns: {required_columns}")

    # Preprocess data
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day of Week'] = data['Date'].dt.dayofweek
    data['Day of Month'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month

    # Encoding categorical variables
    le_model = LabelEncoder()
    le_part = LabelEncoder()
    data['BMW Model'] = le_model.fit_transform(data['BMW Model'])
    data['Car Part'] = le_part.fit_transform(data['Car Part'])

    # Features and target
    X = data[['Day of Week', 'Day of Month', 'Month', 'BMW Model', 'Car Part']]
    y = data['Quantity Used']

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Save the model and label encoders
    with open(save_path, 'wb') as f:
        pickle.dump({'model': model, 'le_model': le_model, 'le_part': le_part}, f)
    print(f"Model saved to {save_path}")


def predict_end_date_from_csv(data_path, current_stock, bmw_model, car_part):
    """
    Predicts the end date of stock depletion using a trained Gradient Boosting model.

    Args:
        data_path (str): Path to the saved model file.
        current_stock (float): Current stock quantity.
        bmw_model (str): The BMW model for prediction.
        car_part (str): The car part for prediction.

    Returns:
        str: Estimated date when the stock will run out.
    """
    # Load the saved model
    with open(data_path, 'rb') as f:
        saved_data = pickle.load(f)

    model = saved_data['model']
    le_model = saved_data['le_model']
    le_part = saved_data['le_part']

    # Handle unseen labels
    if bmw_model not in le_model.classes_:
        print(f"Warning: '{bmw_model}' not found in training data. Using default encoding.")
        le_model.classes_ = np.append(le_model.classes_, bmw_model)
    if car_part not in le_part.classes_:
        print(f"Warning: '{car_part}' not found in training data. Using default encoding.")
        le_part.classes_ = np.append(le_part.classes_, car_part)

    # Encode inputs
    model_encoded = le_model.transform([bmw_model])[0]
    part_encoded = le_part.transform([car_part])[0]

    # Predict stock depletion
    stock_remaining = current_stock
    days_ahead = 0

    while stock_remaining > 0:
        # Generate features for the next day
        future_date = datetime.now() + timedelta(days=days_ahead)
        features = [[future_date.weekday(), future_date.day, future_date.month, model_encoded, part_encoded]]
        daily_consumption = model.predict(features)[0]
        stock_remaining -= daily_consumption
        days_ahead += 1

    end_date = datetime.now() + timedelta(days=days_ahead)
    return end_date.strftime("%Y-%m-%d")


# Example Usage
if __name__ == "__main__":
    # Train and save model from CSV
    csv_path = "bmw_car_parts_usage_synthetic.csv"
    train_model_and_save_from_csv(csv_path)

    # Predict end date
    current_stock = 10
    bmw_model = 'BMW 3 Series'
    car_part = 'Brake Pads'
    model_path = "stock_model.pkl"
    end_date = predict_end_date_from_csv(model_path, current_stock, bmw_model, car_part)
    print(f"Stock will run out on: {end_date}")
