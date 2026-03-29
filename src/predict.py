import pandas as pd
import lightning.pytorch as pl
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from pytorch_forecasting import TemporalFusionTransformer
from src.data_loader import get_tft_data

# Load Models (Global to avoid reloading)
try:
    sarima_model = SARIMAXResults.load("models/sarima.pkl")
    tft_model = TemporalFusionTransformer.load_from_checkpoint("models/tft.ckpt", map_location='cpu')
except Exception as e:
    print(f"Model Load Warning: {e}")
    sarima_model, tft_model = None, None

def predict_sarima(weeks: int = 10):
    """Operational Forecast with Date Mapping."""
    if sarima_model is None:
        return {"error": "SARIMA model not found."}
        
    df = get_tft_data()
    last_date = df['date'].max()
    forecast_values = sarima_model.forecast(steps=weeks).tolist()
    
    return {
        (last_date + pd.Timedelta(weeks=i+1)).strftime('%Y-%m-%d'): round(float(val), 2)
        for i, val in enumerate(forecast_values)
    }

def predict_tft(simulated_adr: float = None):
    """Strategic Radar & What-If Simulator with Date Mapping."""
    if tft_model is None:
        raise RuntimeError("TFT model not loaded.")
        
    df = get_tft_data()
    last_time_idx = df["time_idx"].max()
    encoder_data = df[df.time_idx > last_time_idx - 24].copy()
    last_date = encoder_data['date'].max()
    
    # Anchor for the simulation
    baseline_anchor = encoder_data['Seasonal_Baseline'].mean()
    
    future_data = []
    for i in range(1, 11):
        future_date = last_date + pd.Timedelta(weeks=i)
        
        # Simulation Logic
        target_price = simulated_adr if simulated_adr is not None else baseline_anchor
        adr_diff = target_price - baseline_anchor
        
        future_data.append({
            'date': future_date, 'time_idx': last_time_idx + i, 'group': 'City',
            'month': str(future_date.month), 'City_Bookings': 0.0, 
            'Seasonal_Baseline': baseline_anchor, 'ADR_Diff': adr_diff,
            'City_LeadTime': encoder_data['City_LeadTime'].mean(),
            'City_CancelRate': encoder_data['City_CancelRate'].mean()
        })
        
    decoder_data = pd.DataFrame(future_data)
    inference_df = pd.concat([encoder_data, decoder_data], ignore_index=True)
    
    raw_predictions = tft_model.predict(inference_df, mode="raw", return_x=False)
    forecast_values = raw_predictions.prediction.cpu().numpy()[0, :, 3].tolist()
    
    return {
        (last_date + pd.Timedelta(weeks=i+1)).strftime('%Y-%m-%d'): round(float(val), 2)
        for i, val in enumerate(forecast_values)
    }