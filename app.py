import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.signal import find_peaks

def calculate_fft(data, sampling_rate):
    """
    Calculates the FFT and frequency spectrum of time series data.
    Handles potential data type issues.

    Args:
        data (np.array): Time series data.
        sampling_rate (float): Sampling rate of the data in Hz.

    Returns:
        tuple: Frequencies (np.array), Magnitude spectrum (np.array), Phase spectrum (np.array)
                 Returns None, None, None if there's an error.
    """
    try:
        # Ensure data is numeric and handle potential NaN values
        data = np.array(data, dtype=np.float64) # Explicitly convert to float64
        if np.isnan(data).any(): # Check for NaN values
            st.warning("Warning: Your price data contains NaN (Not a Number) values. These will be replaced with 0 for FFT calculation. Consider cleaning your data for more accurate results.")
            data = np.nan_to_num(data) # Replace NaN with 0

        n = len(data)
        yf = np.fft.fft(data)
        xf = np.fft.fftfreq(n, 1 / sampling_rate) # Correctly calculate frequencies

        # Calculate magnitude spectrum (absolute value of FFT)
        magnitude_spectrum = np.abs(yf)

        # Calculate phase spectrum (angle of FFT) - optional
        phase_spectrum = np.angle(yf)

        return xf[:n//2], magnitude_spectrum[:n//2], phase_spectrum[:n//2] # Return positive frequencies

    except Exception as e:
        st.error(f"Error during FFT calculation: {e}. Please ensure your price data is numeric and does not contain invalid values (e.g., non-numeric characters).")
        return None, None, None # Return None to signal an error


def detect_cycle_lows_fft(frequencies, magnitude_spectrum, sampling_rate):
    """
    Detects dominant cycle periods based on peaks in the FFT magnitude spectrum.
    Returns periods sorted by length (longest first).

    Args:
        frequencies (np.array): Frequencies from FFT.
        magnitude_spectrum (np.array): Magnitude spectrum from FFT.
        sampling_rate (float): Sampling rate (days per sample, here 1.0 for daily data).

    Returns:
        list: List of dominant cycle periods in days, sorted longest to shortest.
    """
    if magnitude_spectrum is None: # Handle case where FFT failed
        return []

    # Find peaks in the magnitude spectrum
    peaks, _ = find_peaks(magnitude_spectrum, prominence=10) # Adjust prominence as needed

    cycle_periods = []
    for peak_index in peaks:
        frequency = frequencies[peak_index]
        if frequency > 0: # Avoid division by zero, ignore DC component (frequency=0)
            period_days = 1 / frequency # Period in days (since frequency is in cycles/day)
            cycle_periods.append(period_days)

    # Sort periods by length (longest first)
    cycle_periods.sort(reverse=True)
    return cycle_periods


def detect_cycle_low_points(data, time, cycle_period_days):
    """
    Detects potential cycle low points based on a moving average.

    Args:
        data (np.array): Time series data.
        time (np.array): Time array (dates).
        cycle_period_days (float): Cycle period in days for moving average.

    Returns:
        tuple: Lists of cycle low dates and corresponding prices.
    """
    window = int(round(cycle_period_days))
    if window <= 1 or window > len(data):
        return [], []  # Invalid window, return empty lists

    moving_average = pd.Series(data).rolling(window=window, center=True).mean().to_numpy()

    cycle_low_dates = []
    cycle_low_prices = []

    # Simple cycle low detection: price below MA and then starts rising
    for i in range(1, len(data)): # Start from index 1 to compare with previous day
        if data[i-1] <= moving_average[i-1] and data[i] > moving_average[i] and data[i] > data[i-1]: # Price crossed MA upwards and is rising
            cycle_low_dates.append(time[i]) # Use time[i] as the date of the low
            cycle_low_prices.append(data[i])

    return cycle_low_dates, cycle_low_prices


def detect_cycle_lows_moving_average(data, time, cycle_periods_days):
    """
    Detects cycle lows using moving averages for visualization. (Unchanged from previous version)
    """
    ma_data = {}
    for period_days in cycle_periods_days:
        window = int(round(period_days)) # Window size for moving average (in days/samples)
        if window > 1 and window <= len(data): # Ensure valid window size
            moving_average = pd.Series(data).rolling(window=window, center=True).mean().to_numpy() # Centered MA for better alignment
            ma_data[period_days] = (time, moving_average)
    return ma_data


def main():
    st.title("Stock Market Cycle Analysis Dashboard")
    st.write("Upload your stock market data (daily time step CSV) to detect daily and weekly cycles and cycle lows.")
    st.write("This dashboard analyzes daily stock data. Ensure your CSV has a 'Date' column and a price column (e.g., 'Close').")

    # Sidebar for settings
    st.sidebar.header("Settings")
    data_source = st.sidebar.radio("Data Source", ["Upload Stock Data CSV File"])
    sampling_rate_daily = 1.0
    sampling_rate = st.sidebar.number_input("Sampling Rate (Days per Sample)", min_value=0.01, value=sampling_rate_daily, format="%.2f", disabled=True, help="Sampling rate is fixed at 1 day per sample for daily stock data.")
    cycle_low_detection_enabled = st.sidebar.checkbox("Detect Cycle Lows", value=True) # Checkbox to enable/disable cycle low detection

    data = None
    time = None

    if data_source == "Upload Stock Data CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload Stock Data CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("Stock data CSV uploaded successfully!")

                date_col_name = st.sidebar.selectbox("Select Date Column", df.columns, index=0)
                price_col_name = st.sidebar.selectbox("Select Price Column (e.g., 'Close')", df.columns)

                if date_col_name and price_col_name:
                    df['Date'] = pd.to_datetime(df[date_col_name])
                    time = df['Date'].to_numpy()
                    data = df[price_col_name].to_numpy()
                else:
                    st.error("Please select both Date and Price columns.")
                    data = None
                    time = None

            except Exception as e:
                st.error(f"Error loading data or parsing CSV. Ensure your CSV has a 'Date' and a price column. Error: {e}")
                data = None
                time = None


    if data is not None and len(data) > 0 and sampling_rate > 0 and time is not None:

        frequencies, magnitude_spectrum, phase_spectrum = calculate_fft(data, sampling_rate)

        if frequencies is not None and magnitude_spectrum is not None: # Proceed only if FFT was successful
            # Detect dominant cycle periods using FFT peaks
            dominant_cycle_periods = detect_cycle_lows_fft(frequencies, magnitude_spectrum, sampling_rate)

            st.subheader("Cycle Analysis Results")
            st.write(f"Detected Dominant Cycle Periods (in days):")
            if dominant_cycle_periods:
                for period in dominant_cycle_periods:
                    st.write(f"- {period:.2f} days")

                fundamental_period = dominant_cycle_periods[0] if dominant_cycle_periods else None # Get the first (longest) period
            else:
                st.write("No significant cycles detected based on FFT peak analysis.")
                fundamental_period = None


            # Data Summary
            st.subheader("Stock Data Summary")
            st.write(f"Number of data points (days): {len(data)}")
            if isinstance(time[0], np.datetime64):
                st.write(f"Time Range: {pd.to_datetime(time[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(time[-1]).strftime('%Y-%m-%d')}")
            else:
                st.write("Time Range: Date range display unavailable due to date format issues.")
            st.write(f"Sampling Rate: Daily (1 sample per day)")

            # Plot Time Domain Data with Moving Averages and Cycle Lows
            st.subheader("Stock Price Time Series with Moving Averages & Cycle Lows")
            fig_time, ax_time = plt.subplots(figsize=(12, 7)) # Adjust figure size
            ax_time.plot(time, data, label="Stock Price", alpha=0.7) # Reduced alpha for original data

            cycle_low_dates = []
            cycle_low_prices = []

            if cycle_low_detection_enabled and fundamental_period: # Only detect and plot if enabled and fundamental period found
                cycle_low_dates, cycle_low_prices = detect_cycle_low_points(data, time, fundamental_period)
                ax_time.plot(cycle_low_dates, cycle_low_prices, 'ro', markersize=5, label=f"Cycle Lows ({fundamental_period:.0f}-Day Cycle)") # Mark cycle lows with red dots

            # Calculate and plot moving averages for detected cycle periods (and some standard periods)
            cycle_periods_to_plot = dominant_cycle_periods + [7, 30, 40] if dominant_cycle_periods else [7, 30, 40] # Include standard periods even if no dominant cycles detected
            ma_data = detect_cycle_lows_moving_average(data, time, cycle_periods_to_plot)

            for period_days, (ma_time, moving_average) in ma_data.items():
                ax_time.plot(ma_time, moving_average, label=f"{period_days:.0f}-Day MA", linestyle='--') # Dashed lines for MAs

            ax_time.set_xlabel("Date")
            ax_time.set_ylabel("Stock Price")
            ax_time.set_title("Stock Price Time Series with Moving Averages and Cycle Lows")
            plt.xticks(rotation=45)
            plt.grid(True, axis='y', linestyle=':') # Grid for better readability
            plt.legend() # Show legend
            plt.tight_layout()
            st.pyplot(fig_time)


            # Plot Frequency Domain (Magnitude Spectrum)
            st.subheader("Frequency Spectrum (Magnitude) - Daily Cycles")
            fig_freq_mag, ax_freq_mag = plt.subplots(figsize=(8, 5)) # Adjust figure size
            ax_freq_mag.plot(frequencies, magnitude_spectrum)
            ax_freq_mag.set_xlabel("Frequency (cycles per day)")
            ax_freq_mag.set_ylabel("Magnitude")
            ax_freq_mag.set_title("Magnitude Spectrum of Stock Price Fluctuations")
            ax_freq_mag.grid(True)
            st.pyplot(fig_freq_mag)

            # Download FFT Results
            st.subheader("Download FFT Results")
            fft_df = pd.DataFrame({'Frequency (cycles/day)': frequencies, 'Magnitude': magnitude_spectrum})

            csv_buffer = StringIO()
            fft_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="Download FFT Results as CSV",
                data=csv_data,
                file_name="fft_results_stock_cycles.csv",
                mime='text/csv',
            )

        else: # FFT calculation failed
            st.error("FFT calculation failed. Please check your data and ensure it's numeric.")


    else:
        st.info("Please upload a stock data CSV file and select the Date and Price columns to start analysis.")

if __name__ == "__main__":
    main()
