import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Cycle Low Detection Tool (CSV Upload)")
st.markdown("Based on principles from [Graddhy's Market Cycles](https://www.graddhy.com/pages/market-cycles)")

# Sidebar for User Inputs
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (daily data with Date and Close columns)", type=["csv"])
lookback_window = st.sidebar.slider("Lookback Window for Cycle Lows (Days)", 5, 90, 20) # Adjust range and default as needed

st.sidebar.markdown("---")
st.sidebar.markdown("Instructions:")
st.sidebar.markdown("1. Upload a CSV file containing daily stock data.")
st.sidebar.markdown("   The CSV should have 'Date' and 'Close' columns.")
st.sidebar.markdown("2. Adjust the 'Lookback Window' to control cycle low sensitivity.")
st.sidebar.markdown("3. Observe the chart for potential cycle lows marked in red.")
st.sidebar.markdown("---")
st.sidebar.markdown("Disclaimer: This is a simplified tool for educational purposes and should not be used for financial advice.")

def load_data_from_csv(uploaded_file):
    """
    Loads data from an uploaded CSV file.
    Assumes the CSV has 'Date' and 'Close' columns.
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Check for required columns
            if 'Date' not in df.columns or 'Close' not in df.columns:
                st.error("CSV file must contain 'Date' and 'Close' columns.")
                return None

            # Convert 'Date' to datetime and set as index
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            except ValueError:
                st.error("Error converting 'Date' column to datetime. Please ensure 'Date' column is in a recognizable date format.")
                return None

            # Ensure 'Close' column is numeric, if possible, for calculations later
            try:
                df['Close'] = pd.to_numeric(df['Close'])
            except ValueError:
                st.warning("Could not convert 'Close' column to numeric. Charting will still work, but numerical operations might fail if 'Close' is needed for other calculations later.")


            if 'Low' not in df.columns: # If Low is not in CSV, assume Close as Low for simplicity in cycle low detection
                df['Low'] = df['Close']
                st.warning("Assuming 'Close' column as 'Low' for cycle low detection as 'Low' column not found in CSV. For more accurate results, please include 'Low' column in your CSV.")


            return df
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None
    else:
        st.warning("Please upload a CSV file to proceed.")
        return None


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

# Load Data from CSV
data = load_data_from_csv(uploaded_file)

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
        title=f"Stock Price with Potential Cycle Lows (from CSV)",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_orientation="h",
        legend=dict(x=0.1, y=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explanation Section (same as before)
    st.subheader("Understanding Cycle Lows (Simplified)")
    st.write("This tool attempts to identify potential cycle lows by finding local minimums in the price chart based on a defined lookback window.")
    st.write(f"The **Lookback Window** is set to {lookback_window} days. This means the tool checks if a low price is the lowest within the past {lookback_window} days (including the current day).")
    st.write("Red markers on the chart indicate potential cycle lows identified by this method.")
    st.write("**Important Considerations:**")
    st.write("- **Subjectivity:** Cycle low detection is not an exact science. This tool provides a visual aid, but user judgment is crucial.")
    st.write("- **Timeframe Dependency:** Cycle lows are relative to the timeframe you are analyzing. Ensure your CSV data is daily timeframe.")
    st.write("- **Context is Key:** Consider other factors like market trends, volume, and fundamental analysis when interpreting potential cycle lows.")
    st.write("- **Simplified Algorithm:** This tool uses a very basic algorithm. More sophisticated methods might consider price patterns, momentum, and other indicators for more robust cycle low detection.")
    st.write("Experiment with different lookback windows to see how the tool identifies potential cycle lows.")
