import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load your secret database URL from the .env file
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise ValueError("DATABASE_URL is missing. Check your .env file.")

def migrate_to_supabase():
    print("Loading local CSV...")
    df = pd.read_csv("data/hotel_bookings.csv")
    
    print("Connecting to Supabase...")
    # SQLAlchemy requires 'postgresql://' instead of 'postgres://' 
    if DB_URL.startswith("postgres://"):
        engine_url = DB_URL.replace("postgres://", "postgresql://", 1)
    else:
        engine_url = DB_URL
        
    engine = create_engine(engine_url)
    
    print("Uploading to cloud database (this may take a minute)...")
    # This automatically creates a table named 'bookings' and uploads the data
    df.to_sql('bookings', engine, if_exists='replace', index=False)
    print("Success! Data is now live in Supabase.")

if __name__ == "__main__":
    migrate_to_supabase()