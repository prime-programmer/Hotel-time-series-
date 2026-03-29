import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.data_loader import get_sarima_data

def train_sarima():
    print("Loading data...")
    ts = get_sarima_data()

    print("Training SARIMA...")
    model = SARIMAX(ts, order=(0, 1, 1), seasonal_order=(0, 1, 0, 52))
    results = model.fit(disp=False)

    os.makedirs("models", exist_ok=True)
    
    # Save model
    results.save("models/sarima.pkl")
    print("Model saved to models/sarima.pkl")

if __name__ == "__main__":
    train_sarima()