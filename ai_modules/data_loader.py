import pandas as pd
def load_sales_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(inplace=True)
    df['quantity_sold'] = df['quantity_sold'].astype(int)
    if 'stock_quantity' in df.columns:
        df['stock_quantity'] = df['stock_quantity'].astype(int)
    if 'price' in df.columns:
        df['price'] = df['price'].astype(float)
    if 'total_revenue' in df.columns:
        df['total_revenue'] = df['total_revenue'].astype(float)
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month
    return df
def load_reviews(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df
