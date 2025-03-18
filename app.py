import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Cycle Low Detection Tool (CSV Upload)")
st.markdown("Based on principles from [Graddhy's Market Cycles](https://www.graddhy.com/pages/market-cycles)")

# Sidebar for User Inputs
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (daily data with Date and Close columns)", type=["csv"])
lookback_window = st.sidebar.slider("Lookback Window for Cycle Lows (Days)", 5, 90, 20)
expected_cycle_length = st.sidebar.number_input("Expected Cycle Length (Days)", min_value=1, value=35, step=1) # Default to 35 days
tolerance_percent = st.sidebar.slider("Cycle Length Tolerance (%)", 0, 50, 10) # Default to 10% tolerance

st.sidebar.markdown("---")
st.sidebar.markdown("Instructions:")
st.sidebar.markdown("1. Upload a CSV file containing daily stock data.")
st.sidebar.markdown("   The CSV should have 'Date' and 'Close' columns (and 'Low' for better results).")
st.sidebar.markdown("2. Adjust 'Lookback Window' for initial low detection sensitivity.")
st.sidebar.markdown("3. Set 'Expected Cycle Length' (e.g., 35 for S&P500).")
st.sidebar.markdown("4. Set 'Cycle Length Tolerance' to control allowed variation.")
st.sidebar.markdown("5. Observe the chart for potential cycle lows marked in red.")
st.sidebar.markdown("---")
st.sidebar.markdown("Disclaimer: This is a simplified tool for educational purposes and should not be used for financial advice.")

def load_data_from_csv(uploaded_file):
    # ... (same load_data_from_csv function as before) ...
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' not in df.columns or 'Close' not in df.columns:
                st.error("CSV file must contain 'Date' and 'Close' columns.")
                return None
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            except ValueError:
                st.error("Error converting 'Date' column to datetime.")
                return None
            try:
                df['Close'] = pd.to_numeric(df['Close'])
            except ValueError:
                st.warning("Could not convert 'Close' to numeric (charting will work).")
            if 'Low' not in df.columns:
                df['Low'] = df['Close']
                st.warning("Assuming 'Close' as 'Low' as 'Low' column not found.")
            return df
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None
    else:
        st.warning("Please upload a CSV file to proceed.")
        return None


def detect_cycle_lows(df, window, expected_cycle_length, tolerance_percent):
    """
    Detects potential cycle lows with a minimum time separation.
    """
    cycle_low_dates = []
    last_cycle_low_date = None  # Keep track of the last detected cycle low date
    min_cycle_interval_days = expected_cycle_length * (1 - tolerance_percent / 100.0) # Calculate minimum interval

    for i in range(window, len(df)):
        window_data = df['Low'][i-window:i+1]
        current_low = df['Low'][i]
        current_date = df.index[i]

        if current_low == window_data.min(): # Found a local minimum
            if last_cycle_low_date is None: # First cycle low found
                cycle_low_dates.append(current_date)
                last_cycle_low_date = current_date
            else:
                time_since_last_low = (current_date - last_cycle_low_date).days
                if time_since_last_low >= min_cycle_interval_days: # Check minimum interval
                    cycle_low_dates.append(current_date)
                    last_cycle_low_date = current_date

    return cycle_low_dates


# Load Data from CSV
data = load_data_from_csv(uploaded_file)

if data is not None:
    # Detect Cycle Lows (with time constraint)
    cycle_low_dates = detect_cycle_lows(data, lookback_window, expected_cycle_length, tolerance_percent)
    cycle_low_prices = data.loc[cycle_low_dates, 'Low']

    # Create Plotly Chart (same as before)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price (Close)'))
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

    # Explanation Section (Updated to include cycle length constraint)
    st.subheader("Understanding Cycle Lows (Simplified)")
    st.write("This tool attempts to identify potential cycle lows by finding local minimums in the price chart based on a defined lookback window, **and now enforces a minimum time separation between cycle lows.**")
    st.write(f"The **Lookback Window** is set to {lookback_window} days, used for initial local minimum detection.")
    st.write(f"The **Expected Cycle Length** is set to {expected_cycle_length} days, and the **Tolerance** to {tolerance_percent}%.")
    st.write(f"This means a new cycle low will only be marked if it occurs at least approximately {expected_cycle_length * (1 - tolerance_percent / 100.0):.1f} days after the previous cycle low.") # Display calculated minimum interval
    st.write("Red markers on the chart indicate potential cycle lows identified by this method.")
    st.write("**Important Considerations:**")
    st.write("- **Subjectivity:** Cycle low detection is not an exact science. This tool provides a visual aid, but user judgment is crucial.")
    st.write("- **Timeframe Dependency:** Cycle lows are relative to the timeframe (daily in this case).")
    st.write("- **Context is Key:** Consider other factors like market trends, volume, and fundamental analysis.")
    st.write("- **Simplified Algorithm:** This tool uses a basic algorithm.  Cycle length is an approximation and market cycles are not perfectly regular.")
    st.write("Experiment with different lookback windows, cycle lengths, and tolerances to see how the tool identifies potential cycle lows.")
