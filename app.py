import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ccxt
import datetime

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



# Streamlit App
#st.title('BTC Price with Extended Cycle Background Colors')

# Sidebar for parameters
st.sidebar.header("Parameter Settings")
symbol = st.text_input("Symbol", "BTC/USD")
#timeframe_months = st.sidebar.selectbox("Select Timeframe (Months)", [6, 12, 18, 24, 48, 72, 96], index=3) # Added timeframe selectbox
expected_period_days = st.sidebar.slider("Expected Cycle Period (Days)", min_value=30, max_value=90, value=60, step=5)
tolerance_days = st.sidebar.slider("Tolerance (Days)", min_value=0, max_value=15, value=6, step=1)
show_half_cycle = st.sidebar.checkbox("Show Half-Cycle Lows", value=True) # Control to toggle half-cycle display
swap_colors = st.sidebar.checkbox("Swap Colors (Cycle/Half-Cycle)", value=False) # NEW: Control to swap colors


# Fetch data from Coinbase API
@st.cache_data(ttl=3600, persist=True) # Cache data for 1 hour, use persist=True for session-based caching if needed
#def load_data_from_coinbase(timeframe_months): # Pass timeframe_months to the cached function
def load_data_from_coinbase(symbol): # Pass timeframe_months to the cached function
    exchange = ccxt.coinbase()
    #symbol = 'BTC/USD'
    timeframe = '1d'
    #limit_days = timeframe_months * 31  # Approximate days for selected months (more than enough)
    limit_days = 300
    limit = limit_days # Use calculated limit
    since_datetime = datetime.datetime.now() - datetime.timedelta(days=limit_days) # Use limit_days
    since_timestamp = exchange.parse8601(since_datetime.isoformat())

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']] # Reorder columns

        st.sidebar.write(f"len: {len(df)}")
        return df
    except ccxt.ExchangeError as e:
        st.error(f" API error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


#df = load_data_from_coinbase(timeframe_months) # Pass timeframe_months to data loading function
df = load_data_from_coinbase(symbol) # Pass timeframe_months to data loading function

if df is not None: # Proceed only if data is loaded successfully

    # Sort DataFrame by date in ascending order (oldest to newest) - already sorted by API but good practice
    df = df.sort_values(by='Date')
    df = df.reset_index(drop=True)

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
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['Close'], label='Price', color='blue')

    # Conditional color assignment based on swap_colors checkbox
    cycle_low_color = 'green' if not swap_colors else 'magenta' # Default green, swapped to magenta if checked
    half_cycle_low_color = 'magenta' if not swap_colors else 'green' # Default magenta, swapped to green if checked

    ax.scatter(minima_df['Date'], minima_df['Close'], color=cycle_low_color, label=cycle_label) # Green or magenta dots for cycle lows

    if show_half_cycle: # Conditionally plot half-cycle lows based on checkbox
        ax.scatter(half_cycle_minima_df_no_overlap['Date'], half_cycle_minima_df_no_overlap['Close'], color=half_cycle_low_color, label=half_cycle_label) # Magenta or green dots for half-cycle lows

    ax.scatter(cycle_highs_df['Date'], cycle_highs_df['High'], color='red', label='Cycle Highs') # Red dots for cycle highs


    # Add labels to cycle high points
    for index, row in cycle_highs_df.iterrows():
        ax.text(row['Date'], row['High'], row['Label'], color='black', fontsize=9, ha='left', va='bottom')


    # Add background color spans for half-cycles
    all_lows_df = pd.concat([minima_df, half_cycle_minima_df_no_overlap]).sort_values(by='Date').reset_index(drop=True) # Use no_overlap df
    for i in range(len(all_lows_df) - 1):
        start_date = all_lows_df['Date'].iloc[i]
        end_date = all_lows_df['Date'].iloc[i+1]
        midpoint_date = start_date + (end_date - start_date) / 2

        ax.axvspan(start_date, midpoint_date, facecolor='lightgreen', alpha=0.2) # Light green before midpoint
        ax.axvspan(midpoint_date, end_date, facecolor='lightpink', alpha=0.2) # Light pink after midpoint

    # Background color after last low
    last_low_date = all_lows_df['Date'].iloc[-1]
    today_date = df['Date'].max() # Use the last date in the dataframe as "today" for consistency with data range
    time_since_last_low = today_date - last_low_date
    threshold_time = pd.Timedelta(days=expected_period_days / 4)

    final_bg_color = 'lightgreen' if time_since_last_low < threshold_time else 'lightpink'
    ax.axvspan(last_low_date, today_date, facecolor=final_bg_color, alpha=0.2) # Background after last low


    ax.set_title(f'BTC/USD Price Chart (Coinbase) - {cycle_label} & {half_cycle_label}') # Dynamic title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

else:
    st.info("Failed to load data from Coinbase API. Please check for errors in the sidebar.")
