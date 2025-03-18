import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.signal import find_peaks

def calculate_fft(data, sampling_rate):
    """
    Calculates the FFT and frequency spectrum of time series data.

    Args:
        data (np.array): Time series data.
        sampling_rate (float): Sampling rate of the data in Hz.

    Returns:
        tuple: Frequencies (np.array), Magnitude spectrum (np.array), Phase spectrum (np.array)
    """
    n = len(data)
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(n, 1 / sampling_rate) # Correctly calculate frequencies

    # Calculate magnitude spectrum (absolute value of FFT)
    magnitude_spectrum = np.abs(yf)

    # Calculate phase spectrum (angle of FFT) - optional
    phase_spectrum = np.angle(yf)

    return xf[:n//2], magnitude_spectrum[:n//2], phase_spectrum[:n//2] # Return positive frequencies


def detect_cycle_lows_fft(frequencies, magnitude_spectrum, sampling_rate):
    """
    Detects dominant cycle periods based on peaks in the FFT magnitude spectrum.

    Args:
        frequencies (np.array): Frequencies from FFT.
        magnitude_spectrum (np.array): Magnitude spectrum from FFT.
        sampling_rate (float): Sampling rate (days per sample, here 1.0 for daily data).

    Returns:
        list: List of dominant cycle periods in days.
    """
    # Find peaks in the magnitude spectrum
    peaks, _ = find_peaks(magnitude_spectrum, prominence=10) # Adjust prominence as needed

    dominant_periods = []
    for peak_index in peaks:
        frequency = frequencies[peak_index]
        if frequency > 0: # Avoid division by zero, ignore DC component (frequency=0)
            period_days = 1 / frequency # Period in days (since frequency is in cycles/day)
            dominant_periods.append(period_days)

    return dominant_periods

def detect_cycle_lows_moving_average(data, time, cycle_periods_days):
    """
    Detects cycle lows using moving averages for visualization.

    Args:
        data (np.array): Time series data.
        time (np.array): Time array (dates).
        cycle_periods_days (list): List of cycle periods in days to calculate moving averages for.

    Returns:
        dict: Dictionary of moving average data for each cycle period.
              Keys are cycle periods (days), values are tuples of (time, moving_average).
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
    st.write("Upload your stock market data (daily time step CSV) to detect daily and weekly cycles.")
    st.write("This dashboard analyzes daily stock data. Ensure your CSV has a 'Date' column and a price column (e.g., 'Close').")

    # Sidebar for settings
    st.sidebar.header("Settings")
    data_source = st.sidebar.radio("Data Source", ["Upload Stock Data CSV File"])
    sampling_rate_daily = 1.0
    sampling_rate = st.sidebar.number_input("Sampling Rate (Days per Sample)", min_value=0.01, value=sampling_rate_daily, format="%.2f", disabled=True, help="Sampling rate is fixed at 1 day per sample for daily stock data.")

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

        # Detect dominant cycle periods using FFT peaks
        dominant_cycle_periods = detect_cycle_lows_fft(frequencies, magnitude_spectrum, sampling_rate)

        st.subheader("Cycle Analysis Results")
        st.write(f"Detected Dominant Cycle Periods (in days):")
        if dominant_cycle_periods:
            for period in dominant_cycle_periods:
                st.write(f"- {period:.2f} days")
        else:
            st.write("No significant cycles detected based on FFT peak analysis.")


        # Data Summary
        st.subheader("Stock Data Summary")
        st.write(f"Number of data points (days): {len(data)}")
        if isinstance(time[0], np.datetime64):
            st.write(f"Time Range: {pd.to_datetime(time[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(time[-1]).strftime('%Y-%m-%d')}")
        else:
            st.write("Time Range: Date range display unavailable due to date format issues.")
        st.write(f"Sampling Rate: Daily (1 sample per day)")

        # Plot Time Domain Data with Moving Averages
        st.subheader("Stock Price Time Series with Moving Averages (for Cycle Visualization)")
        fig_time, ax_time = plt.subplots(figsize=(10, 6)) # Adjust figure size
        ax_time.plot(time, data, label="Stock Price", alpha=0.7) # Reduced alpha for original data

        # Calculate and plot moving averages for detected cycle periods (and some standard periods)
        cycle_periods_to_plot = dominant_cycle_periods + [7, 30, 40] # Include weekly, 30, 40 day cycles
        ma_data = detect_cycle_lows_moving_average(data, time, cycle_periods_to_plot)

        for period_days, (ma_time, moving_average) in ma_data.items():
            ax_time.plot(ma_time, moving_average, label=f"{period_days:.0f}-Day MA", linestyle='--') # Dashed lines for MAs

        ax_time.set_xlabel("Date")
        ax_time.set_ylabel("Stock Price")
        ax_time.set_title("Stock Price Time Series with Moving Averages")
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

    else:
        st.info("Please upload a stock data CSV file and select the Date and Price columns to start analysis.")

if __name__ == "__main__":
    main()
