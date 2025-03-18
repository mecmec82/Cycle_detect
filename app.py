import streamlit as st
import pandas as pd
import talib
import matplotlib.pyplot as plt

# --- Function Definitions ---

def load_data_from_csv(uploaded_file):
    """Loads data from CSV using pandas, handling date parsing."""
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['Date']) # Assuming 'Date' column
        df = df.set_index('Date') # Set Date as index for time series analysis
        df.sort_index(inplace=True) # Ensure chronological order
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def calculate_indicators(df):
    """Calculates technical indicators using ta-lib."""
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    # Add more indicators as needed (e.g., Bollinger Bands, AD Line - if you have necessary data)
    return df

def detect_cycle_low_signals(df):
    """Detects potential cycle low signals based on indicators (example rules)."""
    df['CycleLowSignal'] = False # Initialize signal column

    # Example Rule: RSI oversold and MACD bullish crossover
    oversold_rsi = df['RSI'] < 30
    macd_crossover = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))

    df.loc[oversold_rsi & macd_crossover, 'CycleLowSignal'] = True
    return df

def plot_data_with_signals(df, symbol):
    """Generates a plot with price, indicators, and cycle low signals."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True) # 3 subplots

    # Price Chart
    ax1.plot(df['Close'], label='Close Price', color='blue')
    ax1.set_ylabel('Price')
    ax1.set_title(f'{symbol} - Price and Cycle Low Signals')
    ax1.grid(True)

    # Highlight Cycle Low Signals on Price Chart
    cycle_low_dates = df[df['CycleLowSignal']].index
    ax1.scatter(cycle_low_dates, df.loc[cycle_low_dates, 'Close'], color='red', marker='^', s=100, label='Cycle Low Signal')
    ax1.legend()

    # RSI Indicator
    ax2.plot(df['RSI'], label='RSI', color='purple')
    ax2.axhline(30, color='gray', linestyle='--', label='Oversold (30)')
    ax2.axhline(70, color='gray', linestyle='--', label='Overbought (70)')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True)

    # MACD Indicator
    ax3.plot(df['MACD'], label='MACD', color='teal')
    ax3.plot(df['MACD_Signal'], label='MACD Signal', color='orange')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.grid(True)

    plt.xlabel('Date')
    plt.tight_layout() # Adjust layout to prevent overlapping
    return fig # Return the figure for Streamlit to display

# --- Streamlit App ---

st.title("Cycle Low Detection Tool")
st.write("Upload a CSV file with stock data (Date, Open, High, Low, Close, Volume) to detect potential cycle lows.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data_from_csv(uploaded_file)
    if df is not None and not df.empty: # Check if data loading was successful and not empty
        symbol = "Stock Data" # You can try to extract symbol from filename if needed
        st.subheader("Data Preview")
        st.dataframe(df.head()) # Display first few rows of data

        df_with_indicators = calculate_indicators(df.copy()) # Calculate indicators
        df_with_signals = detect_cycle_low_signals(df_with_indicators.copy()) # Detect signals

        st.subheader("Cycle Low Detection Chart")
        fig = plot_data_with_signals(df_with_signals, symbol)
        st.pyplot(fig) # Display the matplotlib plot in Streamlit

        st.subheader("Cycle Low Signals Summary")
        signals_df = df_with_signals[df_with_signals['CycleLowSignal']]
        if not signals_df.empty:
            st.write("Potential Cycle Low Signals detected on the following dates:")
            st.dataframe(signals_df.index.to_frame(index=False)) # Display dates of signals
        else:
            st.write("No Cycle Low Signals detected based on current rules.")

    else:
        st.warning("Please upload a valid CSV file with stock data.")
