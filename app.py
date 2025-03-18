import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_local_minima_simplified(df, expected_period_days=60, tolerance_days=6):
    """
    Finds local minima using a simplified moving window approach.

    Args:
        df (pd.DataFrame): DataFrame with 'Date' (datetime) and 'Close' columns.
        expected_period_days (int): Expected cycle period in days.
        tolerance_days (int): Tolerance for cycle length variation in days.

    Returns:
        pd.DataFrame: DataFrame containing dates and 'Close' prices of local minima.
    """
    minima_dates = []
    minima_prices = []
    last_low_date = None

    window_size_initial_days = expected_period_days + tolerance_days

    # First Window (from start of data)
    first_start_date = df['Date'].iloc[0]
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

# Streamlit App
st.title('BTC Price with Local Minima (Cycle & Half-Cycle Lows)')

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Assume 'Date' and 'Close' columns, adjust if necessary
        if 'Date' not in df.columns or 'Close' not in df.columns:
            st.error("CSV file must contain 'Date' and 'Close' columns.")
        else:
            # Convert 'Date' to datetime objects
            df['Date'] = pd.to_datetime(df['Date'])  # Let pandas infer format

            # Convert 'Close' column to numeric, removing commas if present
            df['Close'] = df['Close'].astype(str).str.replace(',', '').astype(float)

            # Sort DataFrame by date in ascending order (oldest to newest)
            df = df.sort_values(by='Date')
            df = df.reset_index(drop=True)

            # Sidebar for parameters
            st.sidebar.header("Parameter Settings")
            expected_period_days = st.sidebar.slider("Expected Cycle Period (Days)", min_value=30, max_value=90, value=60, step=5)
            tolerance_days = st.sidebar.slider("Tolerance (Days)", min_value=0, max_value=15, value=6, step=1)
            show_half_cycle = st.sidebar.checkbox("Show Half-Cycle Lows", value=True) # Control to toggle half-cycle display

            # Find local minima (full cycle)
            minima_df = find_local_minima_simplified(
                df.copy(),
                expected_period_days=expected_period_days,
                tolerance_days=tolerance_days
            )

            # Find half-cycle lows
            half_cycle_minima_df = find_local_minima_simplified(
                df.copy(),
                expected_period_days=expected_period_days / 2,  # Half cycle period
                tolerance_days=tolerance_days
            )

            st.sidebar.write(f"Number of Cycle Lows found: {len(minima_df)}")
            st.sidebar.write(f"Number of Half-Cycle Lows found: {len(half_cycle_minima_df)}")

            # Identify overlapping dates and filter half-cycle minima to exclude overlaps
            overlap_dates = set(minima_df['Date']).intersection(set(half_cycle_minima_df['Date']))
            half_cycle_minima_df_no_overlap = half_cycle_minima_df[~half_cycle_minima_df['Date'].isin(overlap_dates)]


            # Plotting with Matplotlib and display in Streamlit
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(df['Date'], df['Close'], label='Price', color='blue')
            ax.scatter(minima_df['Date'], minima_df['Close'], color='green', label='Cycle Lows') # Green dots for cycle lows

            if show_half_cycle: # Conditionally plot half-cycle lows based on checkbox
                ax.scatter(half_cycle_minima_df_no_overlap['Date'], half_cycle_minima_df_no_overlap['Close'], color='magenta', label='Half-Cycle Lows') # Magenta dots for half-cycle lows

            ax.set_title('Price Chart with Cycle and Half-Cycle Lows')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

    except pd.errors.ParserError:
        st.error("Error: Could not parse CSV file. Please ensure it is a valid CSV format.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV file to analyze.")
