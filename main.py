import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# Set page layout to wide
st.set_page_config(
    page_title="Stock Data Analysis Dashboard",  # Update the title
    page_icon="📈",  # Update the icon
    layout="wide"
)

# Sidebar title
st.sidebar.header('Input Options')

# Fetch S&P 500 tickers and names
sp500_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500_tickers = sp500_df['Symbol'].tolist()
sp500_names = sp500_df['Security'].tolist()

# Combine tickers and names into a list of tuples
sp500_combined = [f"{name} ({ticker})" for ticker, name in zip(sp500_tickers, sp500_names)]

selection = st.sidebar.selectbox(
    'Select a Company',
    sp500_combined,
    placeholder='Search by Ticker or Company Name',
    help="Choose a company from the S&P 500 index by its ticker symbol or name."
)

# Extract the selected company details
tickerSymbol, companyName = selection[:-1].split(' (')[-1], selection[:-1].split(' (')[0]

# Sidebar: Date input for start date
start_date = st.sidebar.date_input(
    'Start date',
    datetime(2023, 1, 1),
    help="Select the start date for the stock data."
)

# Sidebar: Checkbox for using the current date as the end date
use_current_date = st.sidebar.checkbox(
    'Use current date as end date',
    value=True,
    help="Enable this option to use the current date as the end date for the stock data."
)

# Sidebar: Date input for end date (only if the checkbox is not checked)
end_date = datetime.now() if use_current_date else st.sidebar.date_input(
    'End date',
    datetime(2020, 5, 31),
    help="Select the end date for the stock data. This is only available if 'Use current date as end date' is unchecked."
)

# Sidebar: New data organization
data_options = st.sidebar.selectbox(
    'Select data to plot',
    ['Stock Price - OHLC', 'Returns & Performance', 'Technical Indicators', 'Additional Information'],
    help="Choose the type of data you want to visualize."
)

# OHLC and Candlesticks selection
if data_options == 'Stock Price - OHLC':
    ohlc_option = st.sidebar.selectbox(
        'Select OHLC data',
        ['Candlesticks', 'OHLC'],
        index=0,
        help="Select the type of OHLC data to display: Candlesticks or traditional OHLC chart."
    )

# Ensure end_date is after start_date
if not use_current_date and start_date > end_date:
    st.error('Error: End date must be after start date.')

# Fetch data for the selected ticker
@st.cache_resource
def fetch_data(tickerSymbol, start_date, end_date):
    tickerData = yf.Ticker(tickerSymbol)
    return tickerData.history(period='1d', start=start_date, end=end_date)

tickerDf = fetch_data(tickerSymbol, start_date, end_date)

# Fetch data for S&P 500 Index
sp500Df = fetch_data('^GSPC', start_date, end_date)

# Calculate Returns
@st.cache_data
def calculate_returns(df):
    df['Stock Returns'] = df['Close'].pct_change()
    df['Log Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Cumulative Returns'] = (1 + df['Stock Returns']).cumprod() - 1
    df['Cumulative Log Returns'] = (1 + df['Log Returns']).cumprod() - 1

calculate_returns(tickerDf)
calculate_returns(sp500Df)

# Rolling Alpha and Beta
rolling_window = 252
if data_options == 'Returns & Performance':
    rolling_window = st.sidebar.slider('Lookback window for Alpha and Beta (in days)',
                                       min_value=1, 
                                       max_value=504, 
                                       value=252,
                                       help="Adjust how many past days are used to calculate Alpha and Beta metrics.")

# Rolling Alpha and Beta
window = 30
if data_options == 'Technical Indicators':
    window = st.sidebar.slider('Lookback window for Bollinger bands (in days)',
                                       min_value=1, 
                                       max_value=252, 
                                       value=30,
                                       help="Adjust how many past days are used to calculate Bollinger bands.")


@st.cache_data
def calculate_rolling_metrics(tickerDf, sp500Df, rolling_window):
    tickerDf['Rolling Alpha'] = np.nan
    tickerDf['Rolling Beta'] = np.nan
    for i in range(rolling_window, len(tickerDf)):
        window_stock_returns = tickerDf['Stock Returns'].iloc[i-rolling_window:i].dropna()
        window_benchmark_returns = sp500Df['Stock Returns'].iloc[i-rolling_window:i].dropna()
        if len(window_stock_returns) > 0 and len(window_benchmark_returns) > 0:
            model = LinearRegression()
            X = window_benchmark_returns.values.reshape(-1, 1)
            y = window_stock_returns.values
            model.fit(X, y)
            tickerDf['Rolling Alpha'].iloc[i] = model.intercept_
            tickerDf['Rolling Beta'].iloc[i] = model.coef_[0]
    return tickerDf

tickerDf = calculate_rolling_metrics(tickerDf, sp500Df, rolling_window)

@st.cache_data
def calculate_macd(data, short_span=12, long_span=26, signal_span=9):
    """
    Calculate the MACD (Moving Average Convergence Divergence) and Signal Line.

    Parameters:
    data (pd.Series): A Pandas Series containing the data to smooth.
    short_span (int): The span for the short EMA (default is 12).
    long_span (int): The span for the long EMA (default is 26).
    signal_span (int): The span for the Signal Line EMA (default is 9).

    Returns:
    pd.DataFrame: A DataFrame with MACD Line, Signal Line, and MACD Histogram.
    """
    # Calculate the short and long EMAs
    short_ema = data.ewm(span=short_span, adjust=False).mean()
    long_ema = data.ewm(span=long_span, adjust=False).mean()

    # Calculate the MACD Line
    macd_line = short_ema - long_ema

    # Calculate the Signal Line
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()

    # Calculate the MACD Histogram
    macd_histogram = macd_line - signal_line

    return pd.DataFrame({
        'MACD Line': macd_line,
        'Signal Line': signal_line,
        'MACD Histogram': macd_histogram
    })
    
macd_data = calculate_macd(tickerDf['Close'])

# Reset index to use Date as a column
tickerDf.reset_index(inplace=True)
sp500Df.reset_index(inplace=True)

# Main Page: Title of the app
st.title(f"📈 S&P 500 Stock Data | {companyName} (`{tickerSymbol}`)")
st.write(f"Explore the stock data for **{companyName}**, compare it with the **S&P 500 Index**, and analyze various **financial metrics**.")

def plot_data(tickerDf, sp500Df, data_options, ohlc_option, compare_to_benchmark):
    if data_options == 'Stock Price - OHLC':
        if ohlc_option == 'Candlesticks':
            st.subheader("Candlestick Chart")
            fig = go.Figure(data=[go.Candlestick(x=tickerDf['Date'],
                                                 open=tickerDf['Open'],
                                                 high=tickerDf['High'],
                                                 low=tickerDf['Low'],
                                                 close=tickerDf['Close'],
                                                 name=tickerSymbol)])
            if compare_to_benchmark:
                fig.add_trace(go.Candlestick(x=sp500Df['Date'],
                                             open=sp500Df['Open'],
                                             high=sp500Df['High'],
                                             low=sp500Df['Low'],
                                             close=sp500Df['Close'],
                                             name='^GSPC'))
            fig.update_layout(xaxis_title='Date',
                              xaxis=dict(
                                    rangeslider=dict(visible=True),
                                    showgrid=True
                              ),
                              yaxis_title='Price ($)',
                              xaxis_rangeslider_visible=False,
                              template='plotly_dark',
                              hovermode='x unified',
                              legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                              margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig)
        else:
            st.subheader("OHLC")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Open'], mode='lines', name=f'{tickerSymbol} Open', line=dict(color='cyan')))
            fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['High'], mode='lines', name=f'{tickerSymbol} High', line=dict(color='#4bffb0')))
            fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Low'], mode='lines', name=f'{tickerSymbol} Low', line=dict(color='pink')))
            fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Close'], mode='lines', name=f'{tickerSymbol} Close', line=dict(color='orange')))
            if compare_to_benchmark:
                fig.add_trace(go.Scatter(x=sp500Df['Date'], y=sp500Df['Close'], mode='lines', name='S&P 500 Close', line=dict(color='#FAFAFA')))
            fig.update_layout(xaxis_title='Date',
                              yaxis_title='Price',
                              template='plotly_dark',
                              hovermode='x unified',
                              legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                              margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig)

    elif data_options == 'Returns & Performance':
        st.subheader("Returns and Performance")
        metrics = ['Stock Returns', 'Log Returns', 'Cumulative Returns', 'Cumulative Log Returns', 'Rolling Alpha', 'Rolling Beta']
        for metric in metrics:
            if metric in tickerDf.columns:
                st.subheader(f"{metric}")
                fig = go.Figure()
                if metric == 'Rolling Alpha' or metric == 'Rolling Beta':
                    fig.add_trace(go.Scatter(x=tickerDf['Date'][rolling_window:], y=tickerDf[metric][rolling_window:], mode='lines', name=f'{companyName} ({tickerSymbol})', line=dict(color='#4bffb0')))
                else:
                    fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf[metric], mode='lines', name=tickerSymbol, line=dict(color='#4bffb0')))
                if compare_to_benchmark:
                    if metric in sp500Df.columns:
                        fig.add_trace(go.Scatter(x=sp500Df['Date'], y=sp500Df[metric], mode='lines', name='^GSPC', line=dict(color='#FAFAFA')))
                fig.update_layout(xaxis_title='Date',
                                  yaxis_title=f'{metric}',
                                  template='plotly_dark',
                                 hovermode='x unified',
                                 legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                 margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig)

        st.subheader("Returns Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=tickerDf['Stock Returns'], name=tickerSymbol, marker=dict(color='#4bffb0'), opacity=0.7, nbinsx=50))
        if compare_to_benchmark:
            fig.add_trace(go.Histogram(x=sp500Df['Stock Returns'], name='^GSPC', marker=dict(color='#FAFAFA'), opacity=0.7, nbinsx=50))
        fig.update_layout(xaxis_title='Returns', 
                          yaxis_title='Frequency', 
                          barmode='overlay', 
                          template='plotly_dark',
                          hovermode='x unified',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                          margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig)
    elif data_options == 'Technical Indicators':
        st.subheader("Technical Indicators")
        st.subheader("Bollinger Bands")
        tickerDf['SMA'] = tickerDf['Close'].rolling(window=window).mean()

        # Calculate the standard deviation
        tickerDf['STD'] = tickerDf['Close'].rolling(window=window).std()
    
        # Calculate the upper and lower Bollinger Bands
        tickerDf['Upper Band'] = tickerDf['SMA'] + 2 * tickerDf['STD']
        tickerDf['Lower Band'] = tickerDf['SMA'] - 2 * tickerDf['STD']
    
        # Create a plotly figure
        fig = go.Figure()
    
        # Plot the closing prices
        fig.add_trace(go.Candlestick(x=tickerDf['Date'][window:],
                                                 open=tickerDf['Open'][window:],
                                                 high=tickerDf['High'][window:],
                                                 low=tickerDf['Low'][window:],
                                                 close=tickerDf['Close'][window:],
                                                 name='Candlesticks'))
    
        # Plot the SMA (middle Bollinger Band)
        fig.add_trace(go.Scatter(
            x=tickerDf['Date'][window:],
            y=tickerDf['SMA'][window:],
            mode='lines',
            name=f'{window}-Day SMA',
            line=dict(color='#FAFAFA', width=2)
        ))
    
        # Plot the upper Bollinger Band
        fig.add_trace(go.Scatter(
            x=tickerDf['Date'][window:],
            y=tickerDf['Upper Band'][window:],
            mode='lines',
            name='Upper Band',
            line=dict(color='#4bffb0', width=1, dash='dot'),
            fill=None
        ))
    
        # Plot the lower Bollinger Band and fill the area between upper and lower bands
        fig.add_trace(go.Scatter(
            x=tickerDf['Date'][window:],
            y=tickerDf['Lower Band'][window:],
            mode='lines',
            name='Lower Band',
            line=dict(color='#4bffb0', width=1, dash='dot'),
            fill='tonexty',  # Fill area between Upper Band and Lower Band
            fillcolor='rgba(173, 216, 230, 0.2)'  # Light blue fill with transparency
        ))
    
        # Customize the layout for a fancier appearance
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            #height=600,
        )
        st.plotly_chart(fig)

        st.subheader("Moving Average Convergence Divergence (MACD)")
        fig_macd = go.Figure()
        
        # Plot the MACD Line
        fig_macd.add_trace(go.Scatter(
            x=macd_data.index,
            y=macd_data['MACD Line'],
            mode='lines',
            name='MACD Line',
            line=dict(color='#4bffb0', width=2)
        ))
    
        # Plot the Signal Line
        fig_macd.add_trace(go.Scatter(
            x=macd_data.index,
            y=macd_data['Signal Line'],
            mode='lines',
            name='Signal Line',
            line=dict(color='#FAFAFA', width=2)
        ))
    
        # Plot the MACD Histogram
        fig_macd.add_trace(go.Bar(
            x=macd_data.index,
            y=macd_data['MACD Histogram'],
            name='MACD Histogram',
            marker=dict(color=macd_data['MACD Histogram'].apply(lambda x: 'rgba(255,0,0,0.9)' if x < 0 else 'rgba(0,255,0,0.9)'))
        ))
    
        # Customize the layout
        fig_macd.update_layout(
            xaxis_title='Date',
            yaxis_title='MACD',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            #height=600,
        )
        st.plotly_chart(fig_macd)
        
    elif data_options == 'Additional Information':
        st.subheader("Volume")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Volume'], mode='lines', name=tickerSymbol, line=dict(color='#4bffb0')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Volume ($)', template='plotly_dark')
        st.plotly_chart(fig)

        st.subheader("Dividends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Dividends'], mode='lines', name=tickerSymbol, line=dict(color='#FAFAFA')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Dividends ($)', template='plotly_dark')
        st.plotly_chart(fig)


# Sidebar: Checkbox to compare to S&P 500 Index
compare_to_benchmark = st.sidebar.checkbox(
    'Compare to S&P 500 Index (`^GSPC`)',
    value=False,
    help="Enable this option to compare the selected company's stock data with the S&P 500 index."
)

# Plot the data based on user selection
plot_data(tickerDf, sp500Df, data_options, ohlc_option if data_options == 'Stock Price - OHLC' else None, compare_to_benchmark)

# Footer
st.markdown("""
    <footer style='text-align: center; color: #ffffff; padding: 1rem; background-color: #1a1a1a;'>
        Developed by Yosri Ben Halima | Financial Data Analytics Platform © 2024
    </footer>
""", unsafe_allow_html=True)
