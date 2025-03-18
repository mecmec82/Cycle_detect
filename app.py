def detect_cycle_lows(df, window, expected_cycle_length, tolerance_percent):
    """
    Detects potential cycle lows with a minimum time separation, allowing for multiple lows.
    (DEBUGGING VERSION WITH PRINT STATEMENTS)
    """
    cycle_low_dates = []
    last_cycle_low_date = None
    min_cycle_interval_days = expected_cycle_length * (1 - tolerance_percent / 100.0)
    print(f"Minimum cycle interval (days): {min_cycle_interval_days:.2f}") # Debug: Show calculated interval

    for i in range(window, len(df)):
        current_date = df.index[i]
        current_low = df['Low'][i]
        window_data = df['Low'][i-window:i+1]
        is_local_low = (current_low == window_data.min())

        print(f"\nDate: {current_date.strftime('%Y-%m-%d')}, Low: {current_low:.2f}, Min in Window: {window_data.min():.2f}, Local Low: {is_local_low}") # Debug: Track current point info

        if is_local_low:
            if last_cycle_low_date is None:
                cycle_low_dates.append(current_date)
                last_cycle_low_date = current_date
                print(f"  - First cycle low detected: {current_date.strftime('%Y-%m-%d')}") # Debug: First low
            else:
                time_since_last_low = (current_date - last_cycle_low_date).days
                print(f"  - Time since last low: {time_since_last_low} days") # Debug: Time since last
                if time_since_last_low >= min_cycle_interval_days:
                    cycle_low_dates.append(current_date)
                    last_cycle_low_date = current_date
                    print(f"  - Cycle low detected: {current_date.strftime('%Y-%m-%d')}, Interval OK") # Debug: Low detected, interval OK
                else:
                    print(f"  - Local low, but interval too short ({time_since_last_low} < {min_cycle_interval_days:.2f} days). Skipped.") # Debug: Low skipped due to interval

    return cycle_low_dates
