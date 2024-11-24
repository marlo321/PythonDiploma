from predict_end_date_gbm import predict_end_date_from_csv as gbm_predict
from regressionWithSeasons import predict_end_date_from_csv as regression_predict
from sarimax import predict_end_date_from_csv as sarimax_predict
from datetime import datetime, timedelta

def ensemble_predict_end_date( current_stock, bmw_model, car_part):
    """
    Combines predictions from GBM, regression with seasons, and SARIMAX models using weighted averaging.

    Parameters:
        current_stock (int): Current stock level.
        bmw_model (str): The BMW model being analyzed.
        car_part (str): The car part being analyzed.
        weights (list[float], optional): Weights for the predictions of the models. Default is equal weights.

    Returns:
        float: Final ensemble prediction for the end date.
    """
    # Get predictions from individual models
    gbm_prediction = gbm_predict('stock_model.pkl', current_stock, bmw_model, car_part)
    regression_prediction = regression_predict('linear_model_with_seasonality.pkl', current_stock, bmw_model, car_part)
    sarimax_prediction = sarimax_predict('sarimax_models.pkl', current_stock, bmw_model, car_part)

    print(f"sarimax_predictionte: {sarimax_prediction}")
    print(f"regression_prediction: {regression_prediction.date()}")
    print(f"gbm_prediction  : {gbm_prediction}")
    # dates = [date.strftime("%Y-%m-%d") for date in [gbm_prediction, regression_prediction.date(), sarimax_prediction]]

    dates = [sarimax_prediction, regression_prediction.strftime("%Y-%m-%d"), gbm_prediction]
    avg_date = calc_avg_date(dates).strftime('%Y-%m-%d')
    print(f"Average date: {avg_date}")

    return avg_date

def calc_avg_date(date_list):
    from datetime import datetime, timedelta

    # Convert strings to datetime objects
    date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in date_list]

    # Calculate the average date
    total_seconds = sum((date - datetime(1970, 1, 1)).total_seconds() for date in date_objects)
    average_seconds = total_seconds / len(date_objects)

    # Convert the average seconds back to a datetime object
    average_date = datetime(1970, 1, 1) + timedelta(seconds=average_seconds)
    return average_date

# Exampprint("Average Date:", average_date)le usage
if __name__ == "__main__":
    current_stock = 100
    bmw_model = "BMW 3 Series"
    car_part = "Brake Pads"

    # Call ensemble method
    final_prediction = ensemble_predict_end_date(
        current_stock,
        bmw_model,
        car_part
    )
    print(f"Ensemble Prediction for End Date: {final_prediction}")
