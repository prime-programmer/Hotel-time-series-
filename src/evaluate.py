import pandas as pd
import numpy as np
import pickle
import lightning.pytorch as pl
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pytorch_forecasting import TemporalFusionTransformer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from src.data_loader import get_sarima_data, get_tft_data
import logging

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

def evaluate_sarima():
    """Evaluates the SARIMA baseline against the last 20 weeks."""
    ts = get_sarima_data()
    
    # Train/Test Split (Isolate the last 20 weeks)
    train = ts.iloc[:-20]
    test = ts.iloc[-20:]

    model = SARIMAX(train, order=(0, 1, 1), seasonal_order=(0, 1, 0, 52), 
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    preds = results.get_forecast(steps=len(test)).predicted_mean

    mape = mean_absolute_percentage_error(test, preds)
    rmse = np.sqrt(mean_squared_error(test, preds))
    
    return mape, rmse

def evaluate_tft():
    """Evaluates the TFT model against the last 10 weeks of known data."""
    try:
        tft_model = TemporalFusionTransformer.load_from_checkpoint("models/tft.ckpt", map_location='cpu')
    except FileNotFoundError:
        print("TFT model not found. Train it first.")
        return None, None

    df = get_tft_data()
    
    # The last 10 weeks are our test set
    test_actuals = df.iloc[-10:]['City_Bookings'].values
    
    # The model needs the 24 weeks strictly PRIOR to the test set to make a prediction
    encoder_data = df.iloc[-34:-10].copy()
    
    # The decoder data provides the time/month structure for the test period, but targets must be 0
    decoder_data = df.iloc[-10:].copy()
    decoder_data['City_Bookings'] = 0.0
    
    inference_df = pd.concat([encoder_data, decoder_data], ignore_index=True)
    
    # Predict (Extract median forecast: index 3)
    raw_predictions = tft_model.predict(inference_df, mode="raw", return_x=False)
    median_forecast = raw_predictions.prediction.cpu().numpy()[0, :, 3]
    
    mape = mean_absolute_percentage_error(test_actuals, median_forecast)
    rmse = np.sqrt(mean_squared_error(test_actuals, median_forecast))
    
    return mape, rmse

def run_all_evaluations():
    print("--- Evaluating SARIMA (Operational Baseline) ---")
    sarima_mape, sarima_rmse = evaluate_sarima()
    print(f"MAPE: {sarima_mape:.2%}")
    print(f"RMSE: {sarima_rmse:.2f}\n")
    
    print("--- Evaluating TFT (Strategic Risk Engine) ---")
    tft_mape, tft_rmse = evaluate_tft()
    if tft_mape is not None:
        print(f"MAPE: {tft_mape:.2%}")
        print(f"RMSE: {tft_rmse:.2f}")

if __name__ == "__main__":
    run_all_evaluations()