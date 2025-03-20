import streamlit as st
import pandas as pd
import plotly_express as px # Importing Plotly Express
import numpy as np
import datetime
import requests  # Import the requests library for API calls

def find_local_minima_simplified(df, expected_period_days=60, tolerance_days=6, start_date=None):
    """
    Finds local minima using a simplified moving window approach.
    (No changes needed in this function)
    """
    minima_dates = []
    minima_prices = []
    last_low_date = None

    window_size_initial_days = expected_period_days + tolerance_days

    # First Window (from start date or start of data)
    if start_date is None:
        first_start_date = df['Date'].iloc[0]
    else:
        first_start_date = start_date

    first_end_date = first_start_date + pd.Timedelta(days=window_size_initial_days)
    first_window_df = df[(df['Date'] >= first_start_date) & (df['Date'] <= first_end_date)]

    if not first_window_df.empty:
        first_min_price_index = first_window_df['Close'].idxmin()
        first_minima_date = df['Date'].loc[first_min_price_index]
        first_minima_price = df['Close'].loc[first_min_price_index]

        minima_dates.append(first_minima_date)
        minima_prices.append(first_minima_price)
        last_low_date = first_minima_date
    else:
        return pd.DataFrame({'Date': minima_dates, 'Close': minima_prices}) # No minima found in first window, return empty

    # Subsequent Windows
    while True:
        next_start_date = last_low_date + pd.Timedelta(days=expected_period_days - tolerance_days)
        next_end_date = last_low_date + pd.Timedelta(days=expected_period_days + tolerance_days)
        current_window_df = df[(df['Date'] >= next_start_date) & (df['Date'] <= next_end_date)]

        if not current_window_df.empty:
            min_price_index_in_window = current_window_df['Close'].idxmin()
            current_minima_date = df['Date'].loc[min_price_index_in_window]
            current_minima_price = df['Close'].loc[min_price_index_in_window]

            minima_dates.append(current_minima_date)
            minima_prices.append(current_minima_price)
            last_low_date = current_minima_date
        else:
            break # No more data in window, stop

    minima_df = pd.DataFrame({'Date': minima_dates, 'Date': minima_dates, 'Close': minima_prices}) # Corrected DataFrame creation
    return minima_df


def find_half_cycle_lows_relative_to_cycle_lows(df, cycle_lows_df, expected_period_days=60, tolerance_days=6):
    """
    Finds half-cycle lows relative to existing cycle lows.
    (No changes needed in this function)
    """
    half_cycle_minima_dates = []
    half_cycle_minima_prices = []

    for index, cycle_low_row in cycle_lows_df.iterrows():
        cycle_low_date = cycle_low_row['Date']

        half_cycle_start_date = cycle_low_date + pd.Timedelta(days=(expected_period_days / 2) - (tolerance_days / 2))
        half_cycle_end_date = cycle_low_date + pd.Timedelta(days=(expected_period_days / 2) + (tolerance_days / 2))

        half_cycle_window_df = df[(df['Date'] >= half_cycle_start_date) & (df['Date'] <= half_cycle_end_date)]

        if not half_cycle_window_df.empty:
            half_cycle_min_price_index = half_cycle_window_df['Close'].idxmin()
            half_cycle_minima_date = df['Date'].loc[half_cycle_min_price_index]
            half_cycle_minima_price = df['Close'].loc[half_cycle_min_price_index]

            half_cycle_minima_dates.append(half_cycle_minima_date)
            half_cycle_minima_prices.append(half_cycle_minima_price)

    half_cycle_minima_df = pd.DataFrame({'Date': half_cycle_minima_dates, 'Date': half_cycle_minima_dates, 'Close': half_cycle_minima_prices}) # Corrected DataFrame creation
    return half_cycle_minima_df

def find_cycle_highs(df, cycle_lows_df, half_cycle_lows_df):
    """
    Finds cycle highs (highest highs) between cycle and half-cycle lows and labels them 'L' or 'R'.
    (No changes needed in this function)
    """
    cycle_high_dates = []
    cycle_high_prices = []
    cycle_high_labels = [] # List to store 'R' labels

    all_lows_df = pd.concat([cycle_lows_df, half_cycle_lows_df]).sort_values(by='Date').reset_index(drop=True)

    for i in range(len(all_lows_df) - 1):
        start_date = all_lows_df['Date'].iloc[i]
        end_date = all_lows_df['Date'].iloc[i+1]

        high_window_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        if not high_window_df.empty:
            max_high_price_index = high_window_df['High'].idxmax()
            cycle_high_date = df['Date'].loc[max_high_price_index]
            cycle_high_price = df['High'].loc[max_high_price_index]

            cycle_high_dates.append(cycle_high_date)
            cycle_high_prices.append(cycle_high_price)

            # Calculate time differences and label
            time_to_high_from_low = cycle_high_date - start_date
            total_time_between_lows = end_date - start_date
            midpoint_time = total_time_between_lows / 2

            if time_to_high_from_low > midpoint_time:
                cycle_high_labels.append('R') # Right/Late
            else:
                cycle_high_labels.append('L') # Left/Early

    cycle_highs_df = pd.DataFrame({'Date': cycle_high_dates, 'High': cycle_high_prices, 'Label': cycle_high_labels}) # Include labels in df
    return cycle_highs_df, cycle_high_labels # Return both df and labels


# Function to fetch data from Alpha Vantage (FREE Endpoint) - MODIFIED for max data download
@st.cache_data(ttl=3600, persist=True)
def load_data_from_alphavantage(symbol, api_key): # Removed limit_days parameter
    function = 'TIME_SERIES_DAILY' # Using free daily endpoint
    outputsize = 'full'  # Fetch maximum available data - always full now
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        if 'Time Series (Daily)' not in data:
            st.error(f"Error fetching data for symbol '{symbol}' from Alpha Vantage. Check symbol or API key. Raw API response: {data}")
            return None

        daily_data = data['Time Series (Daily)']
        df_records = []
        for date_str, values in daily_data.items():
            df_records.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']), # Using '4. close' for unadjusted close
                'Volume': float(values['5. volume']) # Using '5. volume' for unadjusted volume
            })
        df = pd.DataFrame(df_records)
        df = df.sort_values(by='Date')  # Sort by date
        df = df.reset_index(drop=True)

        st.session_state['full_df_alphavantage'] = df # Store full df in session state
        st.sidebar.write(f"Data length for {symbol}: {len(df)}")
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"API request error for symbol '{symbol}': {e}")
        return None
    except ValueError as e:
        st.error(f"Error parsing JSON response from Alpha Vantage for symbol '{symbol}': {e}. Raw response text: {response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data for '{symbol}': {e}")
        return None


# Streamlit App
st.title('Stock Price Cycle Detection')

# Sidebar for parameters
st.sidebar.header("Parameter Settings")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL") # Default to AAPL
alpha_vantage_api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password") # Password type for API key input
# REMOVED: Days to Download Slider - Data is always fully downloaded now
expected_period_days = st.sidebar.slider("Expected Cycle Period (Days)", min_value=30, max_value=90, value=60, step=5)
tolerance_days = st.sidebar.slider("Tolerance (Days)", min_value=0, max_value=15, value=6, step=1)
show_half_cycle = st.sidebar.checkbox("Show Half-Cycle Lows", value=True)
swap_colors = st.sidebar.checkbox("Swap Colors (Cycle/Half-Cycle)", value=False)

# Date Range Slider - NEW Control
if 'full_df_alphavantage' in st.session_state:
    full_df = st.session_state['full_df_alphavantage']
    min_date = full_df['Date'].min()
    max_date = full_df['Date'].max()
    start_date, end_date = st.sidebar.slider(
        "Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date) # Default range is full data range
    )
else:
    start_date = None
    end_date = None


# Load data from Alpha Vantage - Fetch full data only once initially
if 'full_df_alphavantage' not in st.session_state: # Load only if not already in session state
    df_full = load_data_from_alphavantage(symbol, alpha_vantage_api_key)
else:
    df_full = st.session_state['full_df_alphavantage'] # Retrieve from session state


if df_full is not None:
    if start_date is not None and end_date is not None:
        df_filtered = df_full[(df_full['Date'] >= start_date) & (df_full['Date'] <= end_date)].copy() # Filter based on date range
    else:
        df_filtered = df_full.copy() # Use full df if date range not set

    if not df_filtered.empty: # Proceed only if filtered df is not empty
        df = df_filtered.reset_index(drop=True) # Reset index for filtered df

        minima_df = find_local_minima_simplified(
            df.copy(),
            expected_period_days=expected_period_days,
            tolerance_days=tolerance_days
        )

        half_cycle_minima_df = find_half_cycle_lows_relative_to_cycle_lows(
            df.copy(),
            minima_df,
            expected_period_days=expected_period_days,
            tolerance_days=tolerance_days
        )

        cycle_highs_df, cycle_high_labels = find_cycle_highs(df.copy(), minima_df, half_cycle_minima_df)

        cycle_label = "Cycle Lows"
        half_cycle_label = "Half-Cycle Lows"

        st.sidebar.write(f"Number of {cycle_label} found: {len(minima_df)}")
        st.sidebar.write(f"Number of {half_cycle_label} found: {len(half_cycle_minima_df)}")
        st.sidebar.write(f"Number of Cycle Highs found: {len(cycle_highs_df)}")

        overlap_dates = set(minima_df['Date']).intersection(set(half_cycle_minima_df['Date']))
        half_cycle_minima_df_no_overlap = half_cycle_minima_df[~half_cycle_minima_df['Date'].isin(overlap_dates)]


        # Plotting with Plotly Express - INTERACTIVE CHART
        fig = px.line(df, x='Date', y='Close', title=f'{symbol} Stock Price Chart (Alpha Vantage) - {cycle_label} & {half_cycle_label}') # Base price line
        fig.add_scatter(x=minima_df['Date'], y=minima_df['Close'], mode='markers', marker=dict(color='green' if not swap_colors else 'magenta', size=10), name=cycle_label) # Cycle lows
        if show_half_cycle:
            fig.add_scatter(x=half_cycle_minima_df_no_overlap['Date'], y=half_cycle_minima_df_no_overlap['Close'], mode='markers', marker=dict(color='magenta' if not swap_colors else 'green', size=10), name=half_cycle_label) # Half-cycle lows
        fig.add_scatter(x=cycle_highs_df['Date'], y=cycle_highs_df['High'], mode='markers', marker=dict(color='red', size=10), name='Cycle Highs') # Cycle highs

        # Expected Low Lines - Annotations need to be added separately in Plotly
        if not minima_df.empty:
            most_recent_cycle_low_date = minima_df['Date'].iloc[-1]
            expected_next_low_date = most_recent_cycle_low_date + pd.Timedelta(days=expected_period_days)
            expected_next_half_cycle_low_date = most_recent_cycle_low_date + pd.Timedelta(days=expected_period_days / 2)

            fig.add_vline(x=expected_next_low_date.timestamp() * 1000, line_dash="dash", line_color="grey", annotation_text="Exp. Cycle Low", annotation_position="top left") # Cycle line - Plotly uses timestamps in milliseconds
            fig.add_vline(x=expected_next_half_cycle_low_date.timestamp() * 1000, line_dash="dot", line_color="grey", annotation_text="Exp. Half-Cycle Low", annotation_position="top left") # Half-cycle line

        st.plotly_chart(fig, use_container_width=True) # Display Plotly chart

    else:
        st.warning("No data to display for the selected date range.") # Warn if filtered df is empty

else:
    st.info("Failed to load data from Alpha Vantage API. Please check symbol and API key in the sidebar.") # Updated info message
