from fastapi import FastAPI, HTTPException
from src.predict import predict_sarima, predict_tft
from src.data_loader import get_tft_data

app = FastAPI(title="V.Ger Travel Intelligence Engine")

@app.get("/forecast/operational")
def operational_forecast(weeks: int = 10):
    try:
        data = get_tft_data()
        return {
            "model": "SARIMA",
            "data_cutoff": data['date'].max().strftime('%Y-%m-%d'),
            "forecast": predict_sarima(weeks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/strategic")
def strategic_forecast(simulated_adr: float = None):
    try:
        data = get_tft_data()
        # Always get the baseline to show as a 'Radar'
        baseline_forecast = predict_tft(simulated_adr=None)
        
        response = {
            "model": "Temporal Fusion Transformer",
            "data_cutoff": data['date'].max().strftime('%Y-%m-%d'),
            "strategic_radar_baseline": baseline_forecast
        }
        
        if simulated_adr:
            response["mode"] = "What-If Simulation"
            response["simulated_adr"] = simulated_adr
            response["simulated_forecast"] = predict_tft(simulated_adr=simulated_adr)
        else:
            response["mode"] = "Strategic Radar"
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))