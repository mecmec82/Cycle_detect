import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

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

    # Calculate phase spectrum (angle of FFT) - optional, can be added if needed
    phase_spectrum = np.angle(yf)

    return xf[:n//2], magnitude_spectrum[:n//2], phase_spectrum[:n//2] # Return only positive frequencies and corresponding spectra (symmetric)

def main():
    st.title("Stock Market Data FFT/Spectral Analysis Dashboard")
    st.write("Upload your stock market data (daily time step CSV) to perform FFT and spectral analysis.")
    st.write("This dashboard is designed for daily stock data. Ensure your CSV has a 'Date' column and a price column (e.g., 'Close').")

    # Sidebar for settings
    st.sidebar.header("Settings")
    data_source = st.sidebar.radio("Data Source", ["Upload Stock Data CSV File"]) # Removed manual input option for stock data context
    sampling_rate_daily = 1.0  # Fixed sampling rate for daily data (1 sample per day)
    sampling_rate = st.sidebar.number_input("Sampling Rate (Days per Sample)", min_value=0.01, value=sampling_rate_daily, format="%.2f", disabled=True, help="Sampling rate is fixed at 1 day per sample for daily stock data.") # Display and disable sampling rate input

    data = None
    time = None  # Initialize time array

    if data_source == "Upload Stock Data CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload Stock Data CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) # Read CSV without parse_dates initially
                st.sidebar.success("Stock data CSV uploaded successfully!")

                # Column selection for stock data - more specific names
                date_col_name = st.sidebar.selectbox("Select Date Column", df.columns, index=0) # Default to first column
                price_col_name = st.sidebar.selectbox("Select Price Column (e.g., 'Close')", df.columns) # No default index, user must select

                if date_col_name and price_col_name: # Ensure both are selected
                    # Explicitly convert the Date column to datetime objects
                    df['Date'] = pd.to_datetime(df[date_col_name]) # Ensure 'Date' column is datetime
                    time = df['Date'].to_numpy() # Use the standardized 'Date' column for time
                    data = df[price_col_name].to_numpy()

                else:
                    st.error("Please select both Date and Price columns.")
                    data = None
                    time = None


            except Exception as e:
                st.error(f"Error loading data or parsing CSV. Ensure your CSV has a 'Date' column and a price column. Error details: {e}")
                data = None
                time = None


    if data is not None and len(data) > 0 and sampling_rate > 0 and time is not None: # Check if time is also valid now
        # Perform FFT calculation
        frequencies, magnitude_spectrum, phase_spectrum = calculate_fft(data, sampling_rate)

        # Display Data Summary
        st.subheader("Stock Data Summary")
        st.write(f"Number of data points (days): {len(data)}")
        # Ensure time[0] and time[-1] are datetime objects before using strftime
        if isinstance(time[0], np.datetime64):
            st.write(f"Time Range: {pd.to_datetime(time[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(time[-1]).strftime('%Y-%m-%d')}") # Display date range, convert np.datetime64 to pandas datetime for strftime
        else:
            st.write("Time Range: Date range display unavailable due to date format issues.") # Fallback message
        st.write(f"Sampling Rate: Daily (1 sample per day)") # Clarify sampling rate

        # Plot Time Domain Data
        st.subheader("Stock Price Time Series")
        fig_time, ax_time = plt.subplots()
        ax_time.plot(time, data)
        ax_time.set_xlabel("Date")
        ax_time.set_ylabel("Stock Price")
        ax_time.set_title("Stock Price Time Series")
        plt.xticks(rotation=45) # Rotate date labels for better readability
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        st.pyplot(fig_time)

        # Plot Frequency Domain (Magnitude Spectrum)
        st.subheader("Frequency Spectrum (Magnitude) - Daily Cycles")
        fig_freq_mag, ax_freq_mag = plt.subplots()
        ax_freq_mag.plot(frequencies, magnitude_spectrum)
        ax_freq_mag.set_xlabel("Frequency (cycles per day)") # Changed x-axis label to cycles per day
        ax_freq_mag.set_ylabel("Magnitude")
        ax_freq_mag.set_title("Magnitude Spectrum of Stock Price Fluctuations")
        ax_freq_mag.grid(True)
        st.pyplot(fig_freq_mag)

        # Download FFT Results (Magnitude and Frequencies)
        st.subheader("Download FFT Results")
        fft_df = pd.DataFrame({'Frequency (cycles/day)': frequencies, 'Magnitude': magnitude_spectrum}) # Updated frequency units in download

        csv_buffer = StringIO()
        fft_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download FFT Results as CSV",
            data=csv_data,
            file_name="fft_results_stock.csv",
            mime='text/csv',
        )

    else:
        st.info("Please upload a stock data CSV file and select the Date and Price columns to start analysis.")

if __name__ == "__main__":
    main()
