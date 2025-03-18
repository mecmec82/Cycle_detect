import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_local_minima_simplified(df, expected_period_days=60, tolerance_days=6, start_date=None):
    """
    Finds local minima using a simplified moving window approach, optionally starting from a given date.
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
    Finds cycle highs (highest highs) between cycle and half-cycle lows.

    Args:
        df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns.
        cycle_lows_df (pd.DataFrame): DataFrame of cycle lows.
        half_cycle_lows_df (pd.DataFrame): DataFrame of half-cycle lows.

    Returns:
        pd.DataFrame: DataFrame containing dates and 'High' prices of cycle highs.
    """
    cycle_high_dates = []
    cycle_high_prices = []

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

    cycle_highs_df = pd.DataFrame({'Date': cycle_high_dates, 'High': cycle_high_prices})
    return cycle_highs_df


# Streamlit App
st.title('BTC Price with Cycle Lows & Highs')

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

            # Convert price columns to numeric, removing commas if present
            price_columns = ['Open', 'High', 'Low', 'Close'] # Include 'High'
            for col in price_columns:
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)


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

            # Find half-cycle lows relative to cycle lows
            half_cycle_minima_df = find_half_cycle_lows_relative_to_cycle_lows(
                df.copy(),
                minima_df, # Pass cycle lows df
                expected_period_days=expected_period_days,
                tolerance_days=tolerance_days
            )

            # Find cycle highs
            cycle_highs_df = find_cycle_highs(df.copy(), minima_df, half_cycle_minima_df)


            st.sidebar.write(f"Number of Cycle Lows found: {len(minima_df)}")
            st.sidebar.write(f"Number of Half-Cycle Lows found: {len(half_cycle_minima_df)}")
            st.sidebar.write(f"Number of Cycle Highs found: {len(cycle_highs_df)}")


            # Identify overlapping dates and filter half-cycle minima to exclude overlaps
            overlap_dates = set(minima_df['Date']).intersection(set(half_cycle_minima_df['Date']))
            half_cycle_minima_df_no_overlap = half_cycle_minima_df[~half_cycle_minima_df['Date'].isin(overlap_dates)]


            # Plotting with Matplotlib and display in Streamlit
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(df['Date'], df['Close'], label='Price', color='blue')
            ax.scatter(minima_df['Date'], minima_df['Close'], color='green', label='Cycle Lows') # Green dots for cycle lows

            if show_half_cycle: # Conditionally plot half-cycle lows based on checkbox
                ax.scatter(half_cycle_minima_df_no_overlap['Date'], half_cycle_minima_df_no_overlap['Close'], color='magenta', label='Half-Cycle Lows') # Magenta dots for half-cycle lows

            ax.scatter(cycle_highs_df['Date'], cycle_highs_df['High'], color='red', label='Cycle Highs') # Red dots for cycle highs


            ax.set_title('Price Chart with Cycle Lows & Highs')
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
