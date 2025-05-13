# ... (previous code remains the same) ...

    # Plotting with Matplotlib and display in Streamlit
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['Close'], label='Price', color='blue')

    cycle_low_color = 'green'
    half_cycle_low_color = 'magenta'

    ax.scatter(minima_df['Date'], minima_df['Close'], color=cycle_low_color, label=cycle_label, s=60)
    for index, row in minima_df.iterrows():
        ax.annotate('D', (row['Date'], row['Close']), textcoords="offset points", xytext=(0,-20), ha='center', fontsize=12,
                    arrowprops=dict(arrowstyle='-', color='black', linewidth=0.5))

    if show_half_cycle:
        ax.scatter(half_cycle_minima_df_no_overlap['Date'], half_cycle_minima_df_no_overlap['Close'], color=half_cycle_low_color, label=half_cycle_label, s=60)
        for index, row in half_cycle_minima_df_no_overlap.iterrows():
            ax.annotate('H', (row['Date'], row['Close']), textcoords="offset points", xytext=(0,-20), ha='center', fontsize=12,
            arrowprops=dict(arrowstyle='-', color='black', linewidth=0.5))

    ax.scatter(cycle_highs_df['Date'], cycle_highs_df['High'], color='red', label='Cycle Highs')
    # Add labels to cycle high points - using ax.annotate
    for index, row in cycle_highs_df.iterrows():
        ax.annotate(row['Label'],
                    xy=(row['Date'], row['High']),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=12,
                    arrowprops=dict(arrowstyle='-', color='black', linewidth=0.5))


    # --- REMOVE EXISTING BACKGROUND COLOR SPANS ---
    # # Add background color spans for half-cycles
    # all_lows_df = pd.concat([minima_df, half_cycle_minima_df_no_overlap]).sort_values(by='Date').reset_index(drop=True) # Use no_overlap df
    # for i in range(len(all_lows_df) - 1):
    #     start_date = all_lows_df['Date'].iloc[i]
    #     end_date = all_lows_df['Date'].iloc[i+1]
    #     midpoint_date = start_date + (end_date - start_date) / 2

    #     ax.axvspan(start_date, midpoint_date, facecolor='lightgreen', alpha=0.2) # Light green before midpoint
    #     ax.axvspan(midpoint_date, end_date, facecolor='lightpink', alpha=0.2) # Light pink after midpoint

    # # Background color after last low
    # last_low_date = all_lows_df['Date'].iloc[-1]
    # today_date = df['Date'].max() # Use the last date in the dataframe as "today" for consistency with data range
    # time_since_last_low = today_date - last_low_date
    # threshold_time = pd.Timedelta(days=expected_period_days / 4)

    # final_bg_color = 'lightgreen' if time_since_last_low < threshold_time else 'lightpink'
    # ax.axvspan(last_low_date, today_date, facecolor=final_bg_color, alpha=0.2) # Background after last low
    # --- END OF REMOVED SECTION ---


    # Calculate and plot expected next low line (Cycle) and annotation
    if not minima_df.empty: # Use minima_df to get last cycle low
        most_recent_cycle_low_date = minima_df['Date'].iloc[-1] # Last CYCLE low date
        expected_next_cycle_low_date = most_recent_cycle_low_date + pd.Timedelta(days=expected_period_days)

        # --- ADD NEW BACKGROUND SPAN FOR EXPECTED CYCLE LOW RANGE ---
        expected_low_start_date = expected_next_cycle_low_date - pd.Timedelta(days=tolerance_days)
        expected_low_end_date = expected_next_cycle_low_date + pd.Timedelta(days=tolerance_days)
        ax.axvspan(expected_low_start_date, expected_low_end_date, facecolor='lightcoral', alpha=0.3, label='Expected Low Range')
        # --- END OF NEW BACKGROUND SPAN ---


        expected_next_cycle_low_date_str = expected_next_cycle_low_date.strftime('%Y-%m-%d') # Format date to string
        ax.axvline(x=expected_next_cycle_low_date, color='grey', linestyle='--', label='Expected Next Cycle Low') # Add vertical line
        ax.annotate(f'Exp. Cycle Low\n{expected_next_cycle_low_date_str}', xy=(expected_next_cycle_low_date, df['Close'].max()), xytext=(-50, 0), textcoords='offset points',
                    fontsize=10, color='grey', ha='left', va='top') # Annotation for Cycle line with date

        # Calculate and plot expected next half-cycle low line - relative to CYCLE low and annotation
        if show_half_cycle:
            expected_next_half_cycle_low_date = most_recent_cycle_low_date + pd.Timedelta(days=expected_period_days / 2) # Relative to CYCLE low
            expected_next_half_cycle_low_date_str = expected_next_half_cycle_low_date.strftime('%Y-%m-%d') # Format date to string
            ax.axvline(x=expected_next_half_cycle_low_date, color='grey', linestyle=':', label='Expected Next Half-Cycle Low') # Dotted line for half-cycle
            ax.annotate(f'Exp. Half-Cycle Low\n{expected_next_half_cycle_low_date_str}', xy=(expected_next_half_cycle_low_date,  df['Close'].max()), xytext=(-50, -50), textcoords='offset points',
                        fontsize=10, color='grey', ha='left', va='top') # Annotation for Half-Cycle line with date


    title_suffix = "(Coinbase)" if data_source == "Crypto (Coinbase/CCXT)" else "(Alpha Vantage)"
    ax.set_title(f'{symbol} Price Chart {title_suffix} - {cycle_label} & {half_cycle_label}', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    st.pyplot(fig)

# ... (rest of the code remains the same) ...
