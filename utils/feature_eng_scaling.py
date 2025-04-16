#!/usr/bin/env python

import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks



def geometric_adstock(x: pd.DataFrame, alpha: float) -> float:

    """Apply geometric adstock to array x with decay factor alpha."""

    adx = np.zeros_like(x)
    adx[0] = x.iloc[0]
    for t in range(1, len(x)):
        adx[t] = x.iloc[t] + alpha * adx[t-1]  # carryover from prior period
    return adx


def feature_eng_scale(df: pd.DataFrame, data_path):

    # Apply adstock with some assumed true alphas for simulation

    spend_matrix = df.loc[:,"spend_channel_1":"spend_channel_7"]

    # channel 1 and channel 6 have non-active periods hence we indicate that here as columns
    df['channel_1_active'] = (df['spend_channel_1'] > 0).astype(int)
    df['channel_6_active'] = (df['spend_channel_6'] > 0).astype(int)

    channel_cols = [f'spend_channel_{i+1}' for i in range(7)]

    for channel in channel_cols:
        df[f"{channel}_lag1"] = df[channel].shift(1)
        df[f"{channel}_lag2"] = df[channel].shift(2)

    # Fill NaNs from lags with 0 (assumes no past data before start)
    lag_cols = [col for col in df.columns if "_lag" in col]
    df[lag_cols] = df[lag_cols].fillna(0)

    true_alphas = [0.5, 0.1, 0.7, 0.3, 0.8, 0.2, 0.4]
    adstock_matrix = np.column_stack([
        geometric_adstock(spend_matrix.iloc[:, i], alpha)
        for i, alpha in enumerate(true_alphas)
    ])

    # revenue scaling
    revenue_scaler = df['revenue'].mean()
    scaled_revenue = df['revenue'] / revenue_scaler

    # spend matrix scaling according to spend proportions
    channel_means = spend_matrix.mean()
    spend_matrix_scaled = spend_matrix / channel_means

    # Scale lag features by each channel's original mean
    for channel in channel_cols:
        df[f"{channel}_lag1"] = df[f"{channel}_lag1"] / channel_means[channel]
        df[f"{channel}_lag2"] = df[f"{channel}_lag2"] / channel_means[channel]
    
    ### Control features

    trend = np.arange(104)

    # Extract revenue as a NumPy array for peak detection
    sales = df['revenue'].values

    # Detect peaks in revenue data (you can tweak distance and prominence)
    peaks, _ = find_peaks(sales, distance=4, prominence=5)

    # Create a new DataFrame with only the peak rows
    df_peaks = df.iloc[peaks].copy()
    
    # Step 1: Reset index so all components align on row numbers
    date_series = df['start_of_week'].reset_index(drop=True)
    revenue_series = pd.Series(scaled_revenue.ravel(), name='scaled_revenue', index=range(len(df)))
    trend_series = pd.Series(trend, name='trend', index=range(len(df)))
    spend_df = pd.DataFrame(spend_matrix_scaled, columns=[f"spend_channel_{i+1}" for i in range(7)])
    channel_active = df[['channel_1_active', 'channel_6_active']].reset_index(drop=True)
    spend_channel_lag = df[[f"{ch}_lag1" for ch in channel_cols] + [f"{ch}_lag2" for ch in channel_cols]].reset_index(drop=True)

    # Step 2: Ensure spend_df has the correct index
    spend_df.index = range(len(df))

    # Step 3: Create peak_flag series with 0s, then mark 1s at peak weeks
    peak_flag = pd.Series(0, index=range(len(df)), name='peak_flag')
    peak_flag.iloc[peaks] = 1

    # Step 4: Concatenate all parts
    df_processed = pd.concat([
        date_series,
        revenue_series,
        spend_df,
        trend_series,
        peak_flag,
        channel_active,
        spend_channel_lag
    ], axis=1)


    # Export to CSV
    df_processed.to_csv(os.path.join(data_path,"mmm_processed_dataset.csv"), index=False)
