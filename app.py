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
    st.title("FFT/Spectral Analysis Dashboard")
    st.write("Upload your time series data to perform FFT and spectral analysis.")

    # Sidebar for settings
    st.sidebar.header("Settings")
    data_source = st.sidebar.radio("Data Source", ["Upload CSV/TXT File", "Enter Data Manually"])
    sampling_rate = st.sidebar.number_input("Sampling Rate (Hz)", min_value=0.01, value=100.0, format="%.2f") # Default 100 Hz, adjust as needed

    data = None
    time = None  # Initialize time array

    if data_source == "Upload CSV/TXT File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else: # .txt or other text-based
                    # Assuming simple text file with one column of data or two columns (time, value)
                    s = str(uploaded_file.read(),"utf-8")
                    data_io_string = StringIO(s)
                    df = pd.read_csv(data_io_string, sep='\s+', header=None) # Flexible separator, handles spaces, tabs

                st.sidebar.success("Data uploaded successfully!")

                # Try to automatically detect time and value columns
                if df.shape[1] >= 2: # Assume at least two columns (time and value)
                    time_col_name = st.sidebar.selectbox("Select Time Column", df.columns, index=0) # Default to first column
                    value_col_name = st.sidebar.selectbox("Select Value Column", df.columns, index=1) # Default to second column

                    time = df[time_col_name].to_numpy()
                    data = df[value_col_name].to_numpy()
                else: # Assume single column of data (no time column provided)
                    data = df.iloc[:, 0].to_numpy() # Take the first column as data
                    time = np.arange(len(data)) / sampling_rate # Create time array based on sampling rate


            except Exception as e:
                st.error(f"Error loading data: {e}")
                data = None
                time = None

    elif data_source == "Enter Data Manually":
        manual_data_input = st.sidebar.text_area("Enter time series data (comma-separated values):")
        if manual_data_input:
            try:
                data_list = [float(x.strip()) for x in manual_data_input.split(',')]
                data = np.array(data_list)
                time = np.arange(len(data)) / sampling_rate # Create time array
                st.sidebar.success("Data entered manually!")
            except ValueError:
                st.error("Invalid data format. Please enter comma-separated numerical values.")
                data = None
                time = None

    if data is not None and len(data) > 0 and sampling_rate > 0:
        # Perform FFT calculation
        frequencies, magnitude_spectrum, phase_spectrum = calculate_fft(data, sampling_rate)

        # Display Data Summary
        st.subheader("Data Summary")
        st.write(f"Number of data points: {len(data)}")
        st.write(f"Sampling Rate: {sampling_rate} Hz")
        if time is not None:
            st.write(f"Total Time Duration: {time[-1]:.2f} seconds")

        # Plot Time Domain Data
        st.subheader("Time Domain Signal")
        fig_time, ax_time = plt.subplots()
        if time is not None:
            ax_time.plot(time, data)
            ax_time.set_xlabel("Time (seconds)")
        else:
            ax_time.plot(data) # If no time, just plot index
            ax_time.set_xlabel("Sample Index")
        ax_time.set_ylabel("Amplitude")
        ax_time.set_title("Time Series Data")
        st.pyplot(fig_time)

        # Plot Frequency Domain (Magnitude Spectrum)
        st.subheader("Frequency Spectrum (Magnitude)")
        fig_freq_mag, ax_freq_mag = plt.subplots()
        ax_freq_mag.plot(frequencies, magnitude_spectrum)
        ax_freq_mag.set_xlabel("Frequency (Hz)")
        ax_freq_mag.set_ylabel("Magnitude")
        ax_freq_mag.set_title("Magnitude Spectrum")
        ax_freq_mag.grid(True)
        st.pyplot(fig_freq_mag)

        # Plot Frequency Domain (Phase Spectrum) - Optional, uncomment if needed
        # st.subheader("Frequency Spectrum (Phase)")
        # fig_freq_phase, ax_freq_phase = plt.subplots()
        # ax_freq_phase.plot(frequencies, phase_spectrum)
        # ax_freq_phase.set_xlabel("Frequency (Hz)")
        # ax_freq_phase.set_ylabel("Phase (radians)")
        # ax_freq_phase.set_title("Phase Spectrum")
        # ax_freq_phase.grid(True)
        # st.pyplot(fig_freq_phase)

        # Download FFT Results (Magnitude and Frequencies)
        st.subheader("Download FFT Results")
        fft_df = pd.DataFrame({'Frequency (Hz)': frequencies, 'Magnitude': magnitude_spectrum})

        csv_buffer = StringIO()
        fft_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download FFT Results as CSV",
            data=csv_data,
            file_name="fft_results.csv",
            mime='text/csv',
        )

    else:
        st.info("Please upload data or enter data manually and set the sampling rate to start analysis.")

if __name__ == "__main__":
    main()
