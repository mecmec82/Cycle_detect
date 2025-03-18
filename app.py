import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_local_minima_non_overlapping(df, window_size_days=66, expected_period_days=60, tolerance_days=6):
    """
    Finds local minima (cycle lows) in non-overlapping windows of the 'Close' price data,
    ensuring the time difference between consecutive minima is within the specified range.

    Args:
        df (pd.DataFrame): DataFrame with 'Date' (datetime) and 'Close' columns.
        window_size_days (int): Size of each non-overlapping window in days.
        expected_period_days (int): Expected cycle period in days.
        tolerance_days (int): Tolerance for cycle length variation in days.

    Returns:
        pd.DataFrame: DataFrame containing dates and 'Close' prices of local minima.
    """
    minima_dates = []
    minima_prices = []
    start_index = 0
    window_delta = pd.Timedelta(days=window_size_days)
    last_minima_date = None

    lower_bound_timedelta = pd.Timedelta(days=expected_period_days - tolerance_days)
    upper_bound_timedelta = pd.Timedelta(days=expected_period_days + tolerance_days)

    while start_index < len(df):
        start_date = df['Date'].iloc[start_index]
        end_date = start_date + window_delta
        print(f"\nWindow: Start Date = {start_date}, End Date = {end_date}") # PRINT WINDOW DATES

        window_df = df[(df['Date'] >= start_date) & (df['Date'] < end_date)]
        if not window_df.empty:
            print(f"Window DataFrame:\n{window_df}") # PRINT WINDOW DATA
            min_price_index_in_window = window_df['Close'].idxmin()
            current_minima_date = df['Date'].loc[min_price_index_in_window]
            current_minima_price = df['Close'].loc[min_price_index_in_window]
            print(f"Min Price in Window: Date = {current_minima_date}, Price = {current_minima_price}") # PRINT MINIMA IN WINDOW

            if last_minima_date is None:
                print("First Minima - Accepted") # PRINT FIRST MINIMA ACCEPTANCE
                minima_dates.append(current_minima_date)
                minima_prices.append(current_minima_price)
                last_minima_date = current_minima_date
                print(f"Last Minima Date updated to: {last_minima_date}") # PRINT LAST MINIMA DATE UPDATE
            else:
                time_diff = current_minima_date - last_minima_date
                print(f"Time Difference from Last Minima: {time_diff}") # PRINT TIME DIFF
                if lower_bound_timedelta <= time_diff <= upper_bound_timedelta:
                    print("Time Diff Within Range - Accepted") # PRINT TIME DIFF ACCEPTANCE
                    minima_dates.append(current_minima_date)
                    minima_prices.append(current_minima_price)
                    last_minima_date = current_minima_date
                    print(f"Last Minima Date updated to: {last_minima_date}") # PRINT LAST MINIMA DATE UPDATE
                else:
                    print("Time Diff Outside Range - Rejected") # PRINT TIME DIFF REJECTION
                    pass

            # Move to the start of the next window
            try:
                next_start_date = end_date
                start_index = df['Date'][df['Date'] >= next_start_date].index[0]
                print(f"Next Start Index: {start_index}") # PRINT NEXT START INDEX
            except IndexError:
                break
        else:
            break

    minima_df = pd.DataFrame({'Date': minima_dates, 'Close': minima_prices})
    return minima_df

# Streamlit App
st.title('BTC Price with Local Minima (Cycle Lows)')

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
            window_size_days = expected_period_days + tolerance_days # Window size is now calculated

            # Find local minima with non-overlapping windows and time difference constraint
            minima_df = find_local_minima_non_overlapping(
                df.copy(),
                window_size_days=window_size_days,
                expected_period_days=expected_period_days,
                tolerance_days=tolerance_days
            )

            st.sidebar.write(f"Number of local minima found: {len(minima_df)}")

            # Plotting with Matplotlib and display in Streamlit
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(df['Date'], df['Close'], label='Price', color='blue')
            ax.scatter(minima_df['Date'], minima_df['Close'], color='red', label='Local Minima (Cycle Lows)')

            ax.set_title('Price Chart with Local Minima (Cycle Lows)')
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
