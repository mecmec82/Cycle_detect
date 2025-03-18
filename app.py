import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Cycle Low Detection Tool (CSV Upload)")
st.markdown("Based on principles from [Graddhy's Market Cycles](https://www.graddhy.com/pages/market-cycles)")

# Sidebar for User Inputs
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (daily data with Date and Close columns)", type=["csv"])
lookback_window = st.sidebar.slider("Lookback Window for Cycle Lows (Days)", 5, 90, 20)
expected_cycle_length = st.sidebar.number_input("Expected Cycle Length (Days)", min_value=1, value=35, step=1)
tolerance_percent = st.sidebar.slider("Cycle Length Tolerance (%)", 0, 50, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("Instructions:")
st.sidebar.markdown("1. Upload CSV (daily data with Date, Close). 'Low' column is helpful.")
st.sidebar.markdown("2. Adjust 'Lookback Window' for initial low detection.")
st.sidebar.markdown("3. Set 'Expected Cycle Length' (e.g., 35 for S&P500).")
st.sidebar.markdown("4. Set 'Cycle Length Tolerance'.")
st.sidebar.markdown("5. Observe chart for cycle lows (red markers).")
st.sidebar.markdown("---")
st.sidebar.markdown("Disclaimer: Educational tool, not financial advice.")

def load_data_from_csv(uploaded_file):
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


def detect_cycle_lows(df, lookback_window, expected_cycle_length, tolerance_percent):
    """
    Detects potential cycle lows using a windowed iteration approach with cycle length constraints.
    """
    cycle_low_dates = []
    last_cycle_low_date = None
    min_cycle_interval_days = expected_cycle_length * (1 - tolerance_percent / 100.0)
    data_len = len(df)

    # Iterate with a step size related to expected cycle length
    step_size = max(1, int(expected_cycle_length / 2))  # Step roughly half cycle length, but at least 1
    start_index = lookback_window # Start after the initial lookback window
    for i in range(start_index, data_len, step_size):
        # Define the window for this iteration
        window_start = max(0, i - lookback_window) # Ensure window start is not negative
        window_end = min(data_len, i + lookback_window + 1) # Ensure window end is within data bounds
        window_df = df[window_start:window_end]

        if window_df.empty: # Handle empty window case (shouldn't happen often, but good to check)
            continue

        # Find the date of the minimum 'Low' within this window
        min_low_date_in_window = window_df['Low'].idxmin() # Date of min 'Low' in window
        min_low_price_in_window = window_df.loc[min_low_date_in_window, 'Low'] # Price of min 'Low'

        # Check if this minimum is a local low (using original lookback window around the min_low_date)
        local_window_start_index = df.index.get_loc(min_low_date_in_window) # Get index by date
        local_lookback_start = max(0, local_window_start_index - lookback_window)
        local_lookback_end = min(data_len, local_window_start_index + lookback_window + 1)
        local_lookback_window_data = df['Low'][local_lookback_start:local_lookback_end]

        is_local_low = (min_low_price_in_window == local_lookback_window_data.min())


        if is_local_low: # Found a local minimum in the window
            candidate_low_date = min_low_date_in_window

            if last_cycle_low_date is None: # First cycle low
                cycle_low_dates.append(candidate_low_date)
                last_cycle_low_date = candidate_low_date
            else:
                time_since_last_low = (candidate_low_date - last_cycle_low_date).days
                if time_since_last_low >= min_cycle_interval_days: # Check minimum interval
                    cycle_low_dates.append(candidate_low_date)
                    last_cycle_low_date = candidate_low_date

    return cycle_low_dates


# Load Data from CSV
data = load_data_from_csv(uploaded_file)

if data is not None:
    # Detect Cycle Lows (with time constraint)
    cycle_low_dates = detect_cycle_lows(data, lookback_window, expected_cycle_length, tolerance_percent)
    cycle_low_prices = data.loc[cycle_low_dates, 'Low']

    # Create Plotly Chart
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

    # Explanation Section (Updated for clarity on cycle period)
    st.subheader("Understanding Cycle Lows (Simplified)")
    st.write("This tool identifies potential cycle lows by finding local minimums and ensuring they are spaced out by approximately an 'Expected Cycle Length'.")
    st.write(f"The **Lookback Window** ({lookback_window} days) determines how local minimums are detected.")
    st.write(f"The **Expected Cycle Length** is set to {expected_cycle_length} days, with a **Tolerance** of {tolerance_percent}%.")
    st.write(f"The tool aims to find cycle lows that occur roughly every {expected_cycle_length} days (+/- {tolerance_percent}%).")
    st.write("For each potential low, it checks if it's a local minimum and if enough time has passed since the *last detected cycle low* (at least approximately {expected_cycle_length * (1 - tolerance_percent / 100.0):.1f} days).")
    st.write("Red markers indicate potential cycle lows within a cycle period, spaced out by the defined cycle length.")
    st.write("**Important Considerations:**")
    st.write("- **Cycle Period, not just one low:** The goal is to identify *multiple* cycle lows that define cycle periods.")
    st.write("- **Subjectivity:** Cycle low detection is not exact. This tool is a visual aid.")
    st.write("- **Timeframe Dependency:** Daily timeframe assumed. Adjust parameters for other timeframes.")
    st.write("- **Context is Key:** Consider other market factors.")
    st.write("- **Simplified Algorithm:** Basic algorithm, cycle length is approximate.")
    st.write("Experiment with parameters to refine cycle low detection for your data.")
