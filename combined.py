import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ccxt
import datetime
import requests # Import requests for Alpha Vantage


def find_local_minima_simplified(df, expected_period_days=60, tolerance_days=7, start_date=None):
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

    minima_df = pd.DataFrame({'Date': minima_dates, 'Close': minima_prices})
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

    half_cycle_minima_df = pd.DataFrame({'Date': half_cycle_minima_dates, 'Close': half_cycle_minima_prices})
    return half_cycle_minima_df

def find_cycle_highs(df, cycle_lows_df, half_cycle_lows_df):
    """
    Finds cycle highs (highest highs) between cycle and half-cycle lows and labels them 'L' or 'R'.
    (No changes needed in this function)
    """
    cycle_high_dates = []
    cycle_high_prices = []
    cycle_high_labels = [] # List to store 'L' or 'R' labels

    #all_lows_df = pd.concat([cycle_lows_df, half_cycle_lows_df]).sort_values(by='Date').reset_index(drop=True)

    for i in range(len(cycle_lows_df) - 1):
        #start_date = all_lows_df['Date'].iloc[i]
        #end_date = all_lows_df['Date'].iloc[i+1]
        start_date = cycle_lows_df['Date'].iloc[i]
        end_date = cycle_lows_df['Date'].iloc[i+1]

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
                cycle_high_labels.append('') # Right/Late
            else:
                cycle_high_labels.append('') # Left/Early


    cycle_highs_df = pd.DataFrame({'Date': cycle_high_dates, 'High': cycle_high_prices, 'Label': cycle_high_labels}) # Include labels in df
    return cycle_highs_df, cycle_high_labels # Return both df and labels


# Function to fetch data from Alpha Vantage (FREE Endpoint)
@st.cache_data(ttl=3600, persist=True)
def load_data_from_alphavantage(symbol, api_key, limit_days=300): # Added limit_days parameter
    function = 'TIME_SERIES_DAILY' # Using free daily endpoint
    outputsize = 'full'  # Fetch maximum available data
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

        if len(df) > limit_days: # Limit DataFrame to specified days
            df = df.iloc[-limit_days:].reset_index(drop=True) # Take last 'limit_days' rows

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
#st.title('Cycle Low Detection for Crypto & Stocks')

# Sidebar for parameters
st.sidebar.header("Parameter Settings")
data_source = st.sidebar.selectbox("Select Data Source", ["Crypto (Coinbase/CCXT)", "Stocks (Alpha Vantage)"])

if data_source == "Crypto (Coinbase/CCXT)":
    symbol = st.sidebar.text_input("Symbol", "BTC/USD")
    api_key = None # No API key needed for Coinbase/CCXT
elif data_source == "Stocks (Alpha Vantage)":
    symbol = st.sidebar.text_input("Stock Ticker", "AAPL")
    api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password") # Password type for API key input
else:
    symbol = None
    api_key = None


expected_period_days = st.sidebar.slider("Expected Cycle Period (Days)", min_value=30, max_value=90, value=60, step=5)
tolerance_percentage = st.sidebar.slider("Tolerance (%)", min_value=5, max_value=15, value=10, step=1) # Tolerance as percentage
window_size = st.sidebar.slider("Window size to display (Days)", min_value=30, max_value=300, value=100, step=10)
show_half_cycle = st.sidebar.checkbox("Show Half-Cycle Lows", value=True)
swap_colors = st.sidebar.checkbox("Swap Colors (Cycle/Half-Cycle)", value=False)


# Load Data based on data source
df = None # Initialize df
todays_date = datetime.datetime.now()
last_iteration_date = datetime.datetime.now() -  datetime.timedelta(window_size)


if data_source == "Crypto (Coinbase/CCXT)":
    if symbol: # Only load data if symbol is provided
        @st.cache_data(ttl=3600, persist=True) # Cache data for 1 hour
        def load_data_cached_coinbase(symbol, date):
            exchange = ccxt.coinbase()
            timeframe = '1d'
            limit_days = 300
            limit = limit_days
            since_datetime = date - datetime.timedelta(limit_days)
            since_timestamp = exchange.parse8601(since_datetime.isoformat())
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit)
                df_temp = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df_temp['Date'] = pd.to_datetime(df_temp['Timestamp'], unit='ms')
                df_temp = df_temp[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                st.sidebar.write(f"len (Coinbase): {len(df_temp)}")
                return df_temp
            except ccxt.ExchangeError as e:
                st.error(f"Coinbase API error: {e}")
                return None
            except Exception as e:
                st.error(f"An unexpected error occurred with Coinbase API: {e}")
                return None

        df2 = load_data_cached_coinbase(symbol, todays_date)
        df1 = load_data_cached_coinbase(symbol, last_iteration_date)
        if df2 is not None and df1 is not None:
            df=pd.concat([df1,df2])

elif data_source == "Stocks (Alpha Vantage)":
    if api_key and symbol: # Only load if API key and symbol are provided
        df = load_data_from_alphavantage(symbol, api_key) # using provided function
        if df is None:
            st.stop() # Stop if Alpha Vantage data loading fails
    elif not api_key and symbol:
        st.warning("Please enter your Alpha Vantage API Key in the sidebar to fetch stock data.")
        st.stop() # Stop if no API key

if df is not None: # Proceed only if data is loaded successfully

    # Sort DataFrame by date in ascending order (oldest to newest) - already sorted by API but good practice
    df = df.sort_values(by='Date')
    df = df.reset_index(drop=True)

    # Calculate tolerance days from percentage
    tolerance_days = int((expected_period_days * tolerance_percentage) / 100)

    # Find local minima (full cycle)
    minima_df = find_local_minima_simplified(
        df.copy(),
        expected_period_days=expected_period_days,
        tolerance_days=tolerance_days
    )

    # Find half-cycle lows relative to cycle lows
    half_cycle_minima_df = find_half_cycle_lows_relative_to_cycle_lows(
        df.copy(),
        minima_df, # Pass cycle lows df
        expected_period_days=expected_period_days,
        tolerance_days=tolerance_days
    )

    if swap_colors:
        minima_df_copy = minima_df
        minima_df = half_cycle_minima_df
        half_cycle_minima_df = minima_df_copy


    # Find cycle highs and labels
    cycle_highs_df, cycle_high_labels = find_cycle_highs(df.copy(), minima_df, half_cycle_minima_df)


    cycle_label = "Cycle Lows"
    half_cycle_label = "Half-Cycle Lows"


    st.sidebar.write(f"Number of {cycle_label} found: {len(minima_df)}") # Dynamic counts
    st.sidebar.write(f"Number of {half_cycle_label} found: {len(half_cycle_minima_df)}") # Dynamic counts
    st.sidebar.write(f"Number of Cycle Highs found: {len(cycle_highs_df)}")


    # Identify overlapping dates and filter half-cycle minima to exclude overlaps
    overlap_dates = set(minima_df['Date']).intersection(set(half_cycle_minima_df['Date']))
    half_cycle_minima_df_no_overlap = half_cycle_minima_df[~half_cycle_minima_df['Date'].isin(overlap_dates)]


    # Plotting with Matplotlib and display in Streamlit
    fig, ax = plt.subplots(figsize=(14, 7)) # Reverted figsize back to original
    ax.plot(df['Date'], df['Close'], label='Price', color='blue')

    cycle_low_color = 'green'
    half_cycle_low_color = 'magenta'


    ax.scatter(minima_df['Date'], minima_df['Close'], color=cycle_low_color, label=cycle_label, s=60) # MODIFIED: Increased dot size, s=60
    for index, row in minima_df.iterrows(): # NEW: Annotate Cycle Lows BELOW dot

        ax.annotate('D', (row['Date'], row['Close']), textcoords="offset points", xytext=(0,-20), ha='center', fontsize=12,
                    arrowprops=dict(arrowstyle='-', color='black', linewidth=0.5)) # MODIFIED: xytext=(0,-10) for below


    if show_half_cycle: # Conditionally plot half-cycle lows based on checkbox
        ax.scatter(half_cycle_minima_df_no_overlap['Date'], half_cycle_minima_df_no_overlap['Close'], color=half_cycle_low_color, label=half_cycle_label, s=60) # MODIFIED: Increased dot size, s=60

        for index, row in half_cycle_minima_df_no_overlap.iterrows(): # NEW: Annotate Half-Cycle Lows BELOW dot
            ax.annotate('H', (row['Date'], row['Close']), textcoords="offset points", xytext=(0,-20), ha='center', fontsize=12,
            arrowprops=dict(arrowstyle='-', color='black', linewidth=0.5)) # MODIFIED: xytext=(0,-10) for below



    ax.scatter(cycle_highs_df['Date'], cycle_highs_df['High'], color='red', label='Cycle Highs') # Red dots for cycle highs


    # Add labels to cycle high points - using ax.annotate
    for index, row in cycle_highs_df.iterrows():
        ax.annotate(row['Label'], # The text to annotate
                    xy=(row['Date'], row['High']), # Point to annotate
                    xytext=(0, 10), # Offset for text from the point
                    textcoords='offset points', # How xytext is interpreted
                    ha='center', va='bottom', # Text alignment
                    fontsize=12,
                    arrowprops=dict(arrowstyle='-', color='black', linewidth=0.5)) # Optional arrow, removed arrow


    # Add background color spans for half-cycles
    #all_lows_df = pd.concat([minima_df, half_cycle_minima_df_no_overlap]).sort_values(by='Date').reset_index(drop=True) # Use no_overlap df
    #for i in range(len(all_lows_df) - 1):
    #    start_date = all_lows_df['Date'].iloc[i]
    #    end_date = all_lows_df['Date'].iloc[i+1]
    #    midpoint_date = start_date + (end_date - start_date) / 2

    #    ax.axvspan(start_date, midpoint_date, facecolor='lightgreen', alpha=0.2) # Light green before midpoint
    #    ax.axvspan(midpoint_date, end_date, facecolor='lightpink', alpha=0.2) # Light pink after midpoint

    # Background color after last low
    #last_low_date = all_lows_df['Date'].iloc[-1]
    #today_date = df['Date'].max() # Use the last date in the dataframe as "today" for consistency with data range
    #time_since_last_low = today_date - last_low_date
    #threshold_time = pd.Timedelta(days=expected_period_days / 4)

    #final_bg_color = 'lightgreen' if time_since_last_low < threshold_time else 'lightpink'
    #ax.axvspan(last_low_date, today_date, facecolor=final_bg_color, alpha=0.2) # Background after last low

    # Calculate and plot expected next low line (Cycle) and annotation
    if not minima_df.empty: # Use minima_df to get last cycle low
        most_recent_cycle_low_date = minima_df['Date'].iloc[-1] # Last CYCLE low date
        expected_next_low_date = most_recent_cycle_low_date + pd.Timedelta(days=expected_period_days)
        expected_next_low_date_str = expected_next_low_date.strftime('%Y-%m-%d') # Format date to string
        ax.axvline(x=expected_next_low_date, color='grey', linestyle='--', label='Expected Next Low') # Add vertical line
        ax.annotate(f'Exp. Cycle Low\n{expected_next_low_date_str}', xy=(expected_next_low_date, df['Close'].max()), xytext=(-50, 0), textcoords='offset points',
                    fontsize=10, color='grey', ha='left', va='top') # Annotation for Cycle line with date

        # Calculate and plot expected next half-cycle low line - relative to CYCLE low and annotation
        expected_next_half_cycle_low_date = most_recent_cycle_low_date + pd.Timedelta(days=expected_period_days / 2) # Relative to CYCLE low
        expected_next_half_cycle_low_date_str = expected_next_half_cycle_low_date.strftime('%Y-%m-%d') # Format date to string
        ax.axvline(x=expected_next_half_cycle_low_date, color='grey', linestyle=':', label='Expected Next Half-Cycle Low') # Dotted line for half-cycle
        ax.annotate(f'Exp. Half-Cycle Low\n{expected_next_half_cycle_low_date_str}', xy=(expected_next_half_cycle_low_date,  df['Close'].max()), xytext=(-50, -50), textcoords='offset points',
                    fontsize=10, color='grey', ha='left', va='top') # Annotation for Half-Cycle line with date


    # color upcoming low
    
    next_low_window_start = expected_next_low_date - pd.Timedelta(days=tolerance_days)
    next_low_window_end = expected_next_low_date + pd.Timedelta(days=tolerance_days)
    ax.axvspan(next_low_window_start, next_low_window_end, facecolor='lightgreen', alpha=0.2) # Background after last low


    
    title_suffix = "(Coinbase)" if data_source == "Crypto (Coinbase/CCXT)" else "(Alpha Vantage)"
    ax.set_title(f'{symbol} Price Chart {title_suffix} - {cycle_label} & {half_cycle_label}', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    st.pyplot(fig)

else:
    if data_source == "Crypto (Coinbase/CCXT)":
        st.info("Failed to load data from Coinbase API. Please check symbol and API availability.")
    elif data_source == "Stocks (Alpha Vantage)":
        st.info("Failed to load data from Alpha Vantage API. Please check API key, ticker, and API availability.")
    else:
        st.info("Please select a data source and enter required parameters.")
