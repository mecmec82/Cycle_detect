import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.title("Cycle Low Detection Tool")
st.markdown("Based on principles from [Graddhy's Market Cycles](https://www.graddhy.com/pages/market-cycles)")

# Sidebar for User Inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
lookback_window = st.sidebar.slider("Lookback Window for Cycle Lows (Days)", 5, 90, 20) # Adjust range and default as needed

st.sidebar.markdown("---")
st.sidebar.markdown("Instructions:")
st.sidebar.markdown("1. Enter a stock ticker symbol.")
st.sidebar.markdown("2. Select a date range for analysis.")
st.sidebar.markdown("3. Adjust the 'Lookback Window' to control cycle low sensitivity.")
st.sidebar.markdown("4. Observe the chart for potential cycle lows marked in red.")
st.sidebar.markdown("---")
st.sidebar.markdown("Disclaimer: This is a simplified tool for educational purposes and should not be used for financial advice.")

@st.cache_data  # Cache data for better performance
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error(f"Could not retrieve data for ticker: {ticker}. Please check the ticker symbol.")
        return None
    return data

def detect_cycle_lows(df, window):
    """
    Detects potential cycle lows based on a lookback window.
    A point is considered a cycle low if it's the lowest low within the lookback window.
    """
    cycle_low_dates = []
    for i in range(window, len(df)):
        window_data = df['Low'][i-window:i+1] # Include current point in the window
        current_low = df['Low'][i]
        if current_low == window_data.min():
            cycle_low_dates.append(df.index[i])
    return cycle_low_dates

# Load Data
data = load_data(ticker, start_date, end_date)

if data is not None:
    # Detect Cycle Lows
    cycle_low_dates = detect_cycle_lows(data, lookback_window)
    cycle_low_prices = data.loc[cycle_low_dates, 'Low']

    # Create Plotly Chart
    fig = go.Figure()

    # Price Line
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price (Close)'))

    # Cycle Low Markers
    fig.add_trace(go.Scatter(
        x=cycle_low_dates,
        y=cycle_low_prices,
        mode='markers',
        marker=dict(color='red', size=10),
        name='Potential Cycle Lows'
    ))

    fig.update_layout(
        title=f"{ticker} Stock Price with Potential Cycle Lows",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_orientation="h",
        legend=dict(x=0.1, y=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explanation Section
    st.subheader("Understanding Cycle Lows (Simplified)")
    st.write("This tool attempts to identify potential cycle lows by finding local minimums in the price chart based on a defined lookback window.")
    st.write(f"The **Lookback Window** is set to {lookback_window} days. This means the tool checks if a low price is the lowest within the past {lookback_window} days (including the current day).")
    st.write("Red markers on the chart indicate potential cycle lows identified by this method.")
    st.write("**Important Considerations:**")
    st.write("- **Subjectivity:** Cycle low detection is not an exact science. This tool provides a visual aid, but user judgment is crucial.")
    st.write("- **Timeframe Dependency:** Cycle lows are relative to the timeframe you are analyzing. Adjust the date range and lookback window accordingly.")
    st.write("- **Context is Key:** Consider other factors like market trends, volume, and fundamental analysis when interpreting potential cycle lows.")
    st.write("- **Simplified Algorithm:** This tool uses a very basic algorithm. More sophisticated methods might consider price patterns, momentum, and other indicators for more robust cycle low detection.")
    st.write("Experiment with different tickers and lookback windows to see how the tool identifies potential cycle lows.")
