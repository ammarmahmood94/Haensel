#!/usr/bin/env python

import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from statsmodels.tsa.seasonal import seasonal_decompose



def ROAS(df: pd.DataFrame) -> None:

    total_revenue = df["revenue"].sum(axis=0)
    total_spend = (df.loc[:,"spend_channel_1":"spend_channel_7"].sum(axis=0)).sum()

    print(f"Total Revenue: {total_revenue:.0f}")
    print(f"Total Spend: {total_spend:.0f}")
    print(f"ROAS (Total Revenue / Total Spend): {total_revenue/total_spend:.3f}")


def thousands_formatter(x: float, pos: int) -> int:
    
    """
    Formatter function to convert large numbers to a 'k' format.
    E.g., 400000 becomes 400k.
    """

    if abs(x) >= 1000:
            return f'{int(x/1000)}k'
    return f'{int(x)}'


def seasonality_plot(df : pd.DataFrame) -> None:
    
    # Decompose the time series
    decomposition = seasonal_decompose(
        pd.Series(df['revenue'], index=df['start_of_week']),
        model='additive',
        period=52
    )

    # Extract components
    observed = decomposition.observed
    seasonal = decomposition.seasonal

    colors = {
            'observed': '#0F084B',
            'seasonal': '#3D60A7',
        }

    # Define font properties
    title_font = {'fontsize': 12, 'fontweight': 'bold'}
    label_font = {'fontsize': 10, 'fontweight': 'bold'}
    tick_fontsize = 8

    # Define tick locator and formatter: major ticks every 3 months, minor ticks every month
    major_locator = mdates.MonthLocator(interval=3)
    major_formatter = mdates.DateFormatter('%b %Y')
    minor_locator = mdates.MonthLocator(interval=1)

    # axis locator
    i = [0,1]

    fig, ax = plt.subplots(2,1,figsize=(8, 8), dpi=300)
    fig.patch.set_facecolor('white')

    # -------------------------
    # Plot for Observed Component
    # -------------------------

    ax[0].set_title('Observed', **title_font)
    ax[0].set_xlabel('Months', **label_font)
    ax[0].set_ylabel('Sales Revenue (€)', **label_font)

    # Plot observed data with dashed line
    ax[0].plot(observed.index, observed, linestyle='--', color=colors['observed'], label='Observed', linewidth=1.5)

    # Set x-axis major and minor ticks
    ax[0].xaxis.set_major_locator(major_locator)
    ax[0].xaxis.set_major_formatter(major_formatter)
    ax[0].xaxis.set_minor_locator(minor_locator)

    # Format tick labels
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha='right', fontweight='bold', fontsize=tick_fontsize)
    plt.setp(ax[0].get_yticklabels(), fontweight='bold', fontsize=tick_fontsize)

    # Set custom y-axis formatter for thousands
    ax[0].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # Add grid lines for major and minor ticks
    ax[0].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey', alpha=0.8)
    ax[0].grid(which='minor', linestyle=':', linewidth=0.5, color='grey', alpha=0.5)

    ax[0].legend(fontsize=8)

    # -------------------------
    # Plot for Seasonal Component
    # -------------------------

    ax[1].set_title('Seasonal', **title_font)
    ax[1].set_xlabel('Months', **label_font)
    ax[1].set_ylabel('Seasonality (+/- in €)', **label_font)

    # Plot seasonal data with dashed line
    ax[1].plot(seasonal.index, seasonal, linestyle='--', color=colors['seasonal'], label='Seasonal', linewidth=1.5)

    # Set x-axis major and minor ticks
    ax[1].xaxis.set_major_locator(major_locator)
    ax[1].xaxis.set_major_formatter(major_formatter)
    ax[1].xaxis.set_minor_locator(minor_locator)

    # Format tick labels
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha='right', fontweight='bold', fontsize=tick_fontsize)
    plt.setp(ax[1].get_yticklabels(), fontweight='bold', fontsize=tick_fontsize)

    # Set custom y-axis formatter for thousands
    ax[1].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # Add grid lines for major and minor ticks
    ax[1].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey', alpha=0.8)
    ax[1].grid(which='minor', linestyle=':', linewidth=0.5, color='grey', alpha=0.5)

    ax[1].legend(fontsize=8)

    # Adjust layout: moderate spacing between subplots
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15, hspace=0.7, wspace=0.1)

    plt.show()


def trend_plot(df: pd.DataFrame) -> None:
    
    """
    Analyzes trends by plotting bar charts with moving averages for specified columns.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing columns from 'revenue' to 'spend_channel_7'.
    
    The function generates an 8-panel figure (4 rows x 2 columns) where each subplot
    shows a bar chart for the column values and overlays a moving average line.
    """

    columns_to_plot = df.loc[:,"revenue":"spend_channel_7"].columns.tolist() 

    fig, axs = plt.subplots(4, 2, figsize=(14,12), dpi=300)
    fig.suptitle("Trend Analysis",fontsize=14,fontweight='bold')
    axs = axs.flatten()

    for i, col in enumerate(columns_to_plot):
        ax = axs[i]

        ax.bar(df.index, df[col], width=5, label=col, alpha=0.7,color='blue')

        moving_avg = df[col].rolling(window=8).mean()

        ax.plot(df.index, moving_avg, linestyle='--', color='black', label='Moving Avg (8)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3,6,9,12)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

        ax.set_title(col, fontsize=12,fontweight='bold')
        ax.set_xlabel('Months',fontsize=12,fontweight='bold')
        ax.set_ylabel('Amount (€)', fontsize=12,fontweight='bold')

        # Format tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontweight='bold', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize=10)

        # Set custom y-axis formatter for thousands
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

        # Add grid lines for major and minor ticks
        ax.grid(which='major', linestyle='-', linewidth=0.7, color='darkgrey', alpha=0.8)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='grey', alpha=0.5)

        ax.legend()

    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1, hspace=1.2, wspace=0.4)
    plt.show()

