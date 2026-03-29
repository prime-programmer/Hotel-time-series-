import pandas as pd

def clean_hotel_data(df):
    """Applies assignment-specific cleaning rules and engineers seasonal baselines."""
    # 1. Standard Cleaning & Outlier Management
    df = df[df['adr'] < 5000].copy()
    df = df[df['adults'] > 0].copy()
    for col in ['lead_time', 'adr']:
        cap = df[col].quantile(0.99)
        df.loc[df[col] > cap, col] = cap

    # 2. Isolate City Hotel to prevent signal cancellation
    df_city = df[df['hotel'] == 'City Hotel'].copy()

    # 3. Robust Date Parsing
    month_map = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
                 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
    df_city['month_num'] = df_city['arrival_date_month'].map(month_map)
    date_strs = (df_city['arrival_date_year'].astype(str) + '-' +
                 df_city['month_num'].astype(str) + '-' +
                 df_city['arrival_date_day_of_month'].astype(str))
    
    df_city['date'] = pd.to_datetime(date_strs, errors='coerce')
    df_city = df_city.dropna(subset=['date'])
    
    # 4. Feature Engineering: Seasonal Baseline (The "Radar" Anchor)
    df_city['week_of_year'] = df_city['date'].dt.isocalendar().week
    baseline_map = df_city.groupby('week_of_year')['adr'].mean().to_dict()
    df_city['Seasonal_Baseline'] = df_city['week_of_year'].map(baseline_map)
    
    # 5. Feature Engineering: Price Elasticity Delta (The "What-If" Lever)
    df_city['ADR_Diff'] = df_city['adr'] - df_city['Seasonal_Baseline']

    df_city.set_index('date', inplace=True)
    return df_city