from fastapi import FastAPI, HTTPException
from typing import Optional
from ensemble import ensemble_predict_end_date


app = FastAPI()


@app.get("/ensemble_predict_end_date/")
async def ensemble_predict_end_date(
        current_stock: int,
        bmw_model: str,
        car_part: str
):

    if current_stock <= 0:
        raise HTTPException(status_code=400, detail="Current stock must be greater than zero.")

    try:
        return await ensemble_predict_end_date(current_stock, bmw_model, car_part)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# if __name__ == "__main__":
    # current_stock = 10
    # bmw_model = "BMW 3 Series"
    # car_part = "Brake Pads"
    #
    # try:
    #     prediction = ensemble(current_stock, bmw_model, car_part)
    #     print(prediction)
    # except ValueError as e:
    #     print(f"Error: {e}")