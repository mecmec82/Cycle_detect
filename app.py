import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_local_minima(df, window_size_days=60, tolerance_days=6):
    """
    Finds local minima (cycle lows) in the 'Close' price data of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'Date' (datetime) and 'Close' (float) columns.
        window_size_days (int): Expected cycle length in days.
        tolerance_days (int): Tolerance for cycle length variation in days.

    Returns:
        pd.DataFrame: DataFrame containing dates and 'Close' prices of local minima.
    """
    minima_dates = []
    minima_prices = []

    for i in range(len(df)):
        current_date = df['Date'].iloc[i]
        current_price = df['Close'].iloc[i]

        # Define the window for checking local minima
        start_date = current_date - pd.Timedelta(days=window_size_days + tolerance_days)
        end_date = current_date + pd.Timedelta(days=window_size_days + tolerance_days)

        window_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        if not window_df.empty:
            min_price_in_window = window_df['Close'].min()
            if current_price == min_price_in_window:
                minima_dates.append(current_date)
                minima_prices.append(current_price)

    minima_df = pd.DataFrame({'Date': minima_dates, 'Close': minima_prices})
    return minima_df

# Streamlit App
st.title('BTC Price with Local Minima (Cycle Lows)')

# Load the CSV file
file_path = 'COINDESK_BTCUSD (1).csv'
df = pd.read_csv(file_path)

# Convert 'Date' to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Convert 'Close' and other price columns to numeric, removing commas
price_columns = ['Open', 'High', 'Low', 'Close']
for col in price_columns:
    df[col] = df[col].str.replace(',', '').astype(float)

# Sort DataFrame by date in ascending order (oldest to newest)
df = df.sort_values(by='Date')
df = df.reset_index(drop=True)

# Find local minima
window_size_days = 60
tolerance_days = 6
minima_df = find_local_minima(df.copy(), window_size_days, tolerance_days)

# Plotting with Matplotlib and display in Streamlit
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['Date'], df['Close'], label='BTC Price', color='blue')
ax.scatter(minima_df['Date'], minima_df['Close'], color='red', label='Local Minima (Cycle Lows)')

ax.set_title('BTC Price with Local Minima (Cycle Lows)')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
