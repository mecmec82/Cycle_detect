import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

st.title("Cycle Low Detection Tool (CSV Upload)")
st.markdown("Based on principles from [Graddhy's Market Cycles](https://www.graddhy.com/pages/market-cycles)")

# Sidebar for User Inputs
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (daily data with Date and Close columns)", type=["csv"])
lookback_window = st.sidebar.slider("Lookback Window for Initial Low (Days)", 5, 90, 20) # For the *first* low
expected_cycle_length = st.sidebar.number_input("Expected Cycle Length (Days)", min_value=1, value=35, step=1)
tolerance_percent = st.sidebar.slider("Cycle Length Tolerance (%)", 0, 50, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("Instructions:")
st.sidebar.markdown("1. Upload CSV (daily data with Date, Close). 'Low' column is helpful.")
st.sidebar.markdown("2. Adjust 'Lookback Window' for finding the *first* cycle low.")
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
    Detects cycle lows using iterative forward search based on expected cycle length.
    """
    cycle_low_dates = []
    last_cycle_low_date = None
    tolerance_days = timedelta(days=expected_cycle_length * (tolerance_percent / 100.0))

    # 1. Find the first cycle low in the initial lookback window
    first_window_df = df.iloc[:lookback_window+1] # Include current point
    if not first_window_df.empty:
        initial_low_date = first_window_df['Low'].idxmin()
        initial_low_price = first_window_df.loc[initial_low_date, 'Low']

        # Verify if it's a local low within the lookback window (centered around initial_low_date)
        initial_local_start_index = df.index.get_loc(initial_low_date)
        initial_local_lookback_start = max(0, initial_local_start_index - lookback_window)
        initial_local_lookback_end = min(len(df), initial_local_start_index + lookback_window + 1)
        initial_local_lookback_data = df['Low'][initial_local_lookback_start:initial_local_lookback_end]
        if initial_low_price == initial_local_lookback_data.min(): # Confirm local low
            cycle_low_dates.append(initial_low_date)
            last_cycle_low_date = initial_low_date
        else:
            last_cycle_low_date = initial_low_date # Still set even if not local low, to start search from here.  May need to refine this.


    # 2. Iteratively search for subsequent lows
    if last_cycle_low_date is not None:
        current_low_date = last_cycle_low_date
        while True:
            expected_next_low_date = current_low_date + timedelta(days=expected_cycle_length)
            search_window_start_date = expected_next_low_date - tolerance_days
            search_window_end_date = expected_next_low_date + tolerance_days

            # Define search window DataFrame (handle cases where window goes beyond data range)
            search_window_df = df[search_window_start_date:search_window_end_date]

            if search_window_df.empty: # No data in search window, stop searching forward
                break

            next_low_date_candidate = search_window_df['Low'].idxmin()
            next_low_price_candidate = search_window_df.loc[next_low_date_candidate, 'Low']

            # Verify local low in a lookback window around candidate
            local_start_index = df.index.get_loc(next_low_date_candidate)
            local_lookback_start = max(0, local_start_index - lookback_window)
            local_lookback_end = min(len(df), local_start_index + lookback_window + 1)
            local_lookback_data = df['Low'][local_lookback_start:local_lookback_end]

            if next_low_price_candidate == local_lookback_data.min(): # Confirmed local low
                if next_low_date_candidate > current_low_date: # Ensure it's a *later* low
                    cycle_low_dates.append(next_low_date_candidate)
                    current_low_date = next_low_date_candidate # Move forward from this new low
                else:
                    break # Found a low at or before current, something wrong, stop.  Maybe data issue?
            else:
                break # No local low found in search window, stop searching forward

    return cycle_low_dates


# Load Data from CSV
data = load_data_from_csv(uploaded_file)

if data is not None:
    # Detect Cycle Lows (iterative forward search)
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

    # Explanation Section (Updated for iterative approach)
    st.subheader("Understanding Cycle Lows (Simplified)")
    st.write("This tool now uses an iterative forward search to detect cycle lows based on the 'Expected Cycle Length'.")
    st.write(f"It first finds an initial potential low within the first **Lookback Window** ({lookback_window} days).")
    st.write(f"Then, it iteratively searches for subsequent lows approximately every **Expected Cycle Length** ({expected_cycle_length} days), with a **Tolerance** of {tolerance_percent}%.")
    st.write("For each iteration, it defines a search window around the expected date of the next low and looks for the minimum low within that window.")
    st.write("Red markers indicate potential cycle lows identified by this iterative forward search method.")
    st.write("**Important Considerations:**")
    st.write("- **Iterative Forward Search:**  The tool now uses a more targeted search approach based on cycle length.")
    st.write("- **Initial Low Dependency:** The first detected low influences subsequent low detection.") # Important point
    st.write("- **Subjectivity:** Cycle low detection is not exact. This tool is a visual aid.")
    st.write("- **Timeframe Dependency:** Daily timeframe assumed. Adjust parameters for other timeframes.")
    st.write("- **Context is Key:** Consider other market factors.")
    st.write("- **Simplified Algorithm:** Basic algorithm, cycle length is approximate and real market cycles can vary significantly.")
    st.write("Experiment with parameters, especially 'Lookback Window' (for the initial low) and 'Expected Cycle Length' and 'Tolerance' to refine cycle low detection for your data.")
