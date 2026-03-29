import pickle

# Load model ONCE
with open("models/sarima.pkl", "rb") as f:
    model = pickle.load(f)

def predict_forecast(weeks=4):
    """
    Forecasts future City Hotel demand.
    Note: The underlying model is aggregated weekly. 
    A request of weeks=4 returns demand for the next 4 weeks.
    """
    forecast = model.forecast(steps=weeks)
    return forecast