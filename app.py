import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ccxt
import datetime

# ... (Your functions: find_local_minima_simplified, find_half_cycle_lows_relative_to_cycle_lows, find_cycle_highs - no changes needed) ...


# Streamlit App
#st.title('BTC Price with Extended Cycle Background Colors') # Removed title - can be dynamic later

# Sidebar for parameters
st.sidebar.header("Parameter Settings")

# **NEW: Symbol Input Box**
default_symbol = 'BTC/USD' # Set a default symbol
symbol_input = st.sidebar.text_input("Enter Trading Symbol (e.g., BTC/USD)", default_symbol)
selected_symbol = symbol_input.upper() # Convert input to uppercase for consistency


expected_period_days = st.sidebar.slider("Expected Cycle Period (Days)", min_value=30, max_value=90, value=60, step=5)
tolerance_days = st.sidebar.slider("Tolerance (Days)", min_value=0, max_value=15, value=6, step=1)
show_half_cycle = st.sidebar.checkbox("Show Half-Cycle Lows", value=True)
swap_colors = st.sidebar.checkbox("Swap Colors (Cycle/Half-Cycle)", value=False)


# Fetch data from Coinbase API
@st.cache_data(ttl=3600, persist=True)
#def load_data_from_coinbase(): # Removed - was before symbol input
def load_data_from_coinbase(symbol): # **MODIFIED: Accept symbol as argument**
    exchange = ccxt.coinbase()
    #symbol = 'BTC/USD' # Removed - using input symbol now
    timeframe = '1d'
    limit_days = 300
    limit = limit_days
    since_datetime = datetime.datetime.now() - datetime.timedelta(days=limit_days)
    since_timestamp = exchange.parse8601(since_datetime.isoformat())

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit) # **Use input symbol**
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        st.sidebar.write(f"len: {len(df)}")
        return df
    except ccxt.ExchangeError as e:
        st.error(f" API error for symbol '{symbol}': {e}") # **Include symbol in error message**
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred for symbol '{symbol}': {e}") # **Include symbol in error message**
        return None


#df = load_data_from_coinbase() # Removed - was before symbol input
df = load_data_from_coinbase(selected_symbol) # **MODIFIED: Pass selected_symbol to data loading function**


if df is not None: # Proceed only if data is loaded successfully

    # ... (Rest of your data processing code: sorting, finding minima, highs - no changes needed) ...


    cycle_label = "Cycle Lows"
    half_cycle_label = "Half-Cycle Lows"


    st.sidebar.write(f"Number of {cycle_label} found: {len(minima_df)}")
    st.sidebar.write(f"Number of {half_cycle_label} found: {len(half_cycle_minima_df)}")
    st.sidebar.write(f"Number of Cycle Highs found: {len(cycle_highs_df)}")


    overlap_dates = set(minima_df['Date']).intersection(set(half_cycle_minima_df['Date']))
    half_cycle_minima_df_no_overlap = half_cycle_minima_df[~half_cycle_minima_df['Date'].isin(overlap_dates)]


    # Plotting with Matplotlib
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['Close'], label='Price', color='blue')

    cycle_low_color = 'green' if not swap_colors else 'magenta'
    half_cycle_low_color = 'magenta' if not swap_colors else 'green'

    ax.scatter(minima_df['Date'], minima_df['Close'], color=cycle_low_color, label=cycle_label)

    if show_half_cycle:
        ax.scatter(half_cycle_minima_df_no_overlap['Date'], half_cycle_minima_df_no_overlap['Close'], color=half_cycle_low_color, label=half_cycle_label)

    ax.scatter(cycle_highs_df['Date'], cycle_highs_df['High'], color='red', label='Cycle Highs')

    for index, row in cycle_highs_df.iterrows():
        ax.text(row['Date'], row['High'], row['Label'], color='black', fontsize=9, ha='left', va='bottom')


    all_lows_df = pd.concat([minima_df, half_cycle_minima_df_no_overlap]).sort_values(by='Date').reset_index(drop=True)
    for i in range(len(all_lows_df) - 1):
        start_date = all_lows_df['Date'].iloc[i]
        end_date = all_lows_df['Date'].iloc[i+1]
        midpoint_date = start_date + (end_date - start_date) / 2

        ax.axvspan(start_date, midpoint_date, facecolor='lightgreen', alpha=0.2)
        ax.axvspan(midpoint_date, end_date, facecolor='lightpink', alpha=0.2)

    last_low_date = all_lows_df['Date'].iloc[-1]
    today_date = df['Date'].max()
    time_since_last_low = today_date - last_low_date
    threshold_time = pd.Timedelta(days=expected_period_days / 4)

    final_bg_color = 'lightgreen' if time_since_last_low < threshold_time else 'lightpink'
    ax.axvspan(last_low_date, today_date, facecolor=final_bg_color, alpha=0.2)


    #ax.set_title(f'BTC/USD Price Chart (Coinbase) - {cycle_label} & {half_cycle_label}') # Old title
    ax.set_title(f'{selected_symbol} Price Chart (Coinbase) - {cycle_label} & {half_cycle_label}') # **MODIFIED: Dynamic title with selected symbol**
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

else:
    st.info("Failed to load data from Coinbase API. Please check for errors in the sidebar and ensure the symbol is valid.") # **Improved info message**
