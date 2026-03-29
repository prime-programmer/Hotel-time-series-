import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from src.preprocess import clean_hotel_data

# Load the database URL from your .env file
load_dotenv(override=True)

def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is missing from environment variables.")
    
    # SQLAlchemy requires 'postgresql://', not 'postgres://'
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    return create_engine(db_url)

def load_weekly_base():
    engine = get_db_connection()
    
    # Pull directly from your live Supabase cloud!
    query = "SELECT * FROM bookings"
    df = pd.read_sql(query, engine)

    # The rest of your existing logic stays exactly the same
    df_clean = clean_hotel_data(df)

    weekly = df_clean.resample('W').agg({
        'hotel': 'count',
        'Seasonal_Baseline': 'mean',
        'ADR_Diff': 'mean',
        'lead_time': 'mean',
        'is_canceled': 'mean'
    }).fillna(0)
    
    weekly.columns = ['City_Bookings', 'Seasonal_Baseline', 'ADR_Diff', 'City_LeadTime', 'City_CancelRate']
    weekly['City_Bookings'] = weekly['City_Bookings'].astype(float)
    
    return weekly

def get_tft_data():
    weekly = load_weekly_base().reset_index()
    weekly['time_idx'] = range(len(weekly))
    weekly['group'] = 'City'
    weekly['month'] = weekly['date'].dt.month.astype(str)
    return weekly

def get_sarima_data():
    weekly = load_weekly_base()
    return weekly['City_Bookings']