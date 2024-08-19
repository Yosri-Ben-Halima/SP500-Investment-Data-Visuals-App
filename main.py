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
sp500_combined = [ticker+' - '+name for ticker, name in zip(sp500_tickers, sp500_names)]

selection = st.sidebar.selectbox('Select a Company', sp500_combined, placeholder='Search by Ticker or Company Name')

# Extract the selected company details
tickerSymbol, companyName = selection.split(' - ')[0], selection.split(' - ')[1]

# Sidebar: Date input for start date
start_date = st.sidebar.date_input('Start date', datetime(2018, 5, 31))

# Sidebar: Checkbox for using the current date as the end date
use_current_date = st.sidebar.checkbox('Use current date as end date', value=True)

# Sidebar: Date input for end date (only if the checkbox is not checked)
end_date = datetime.now() if use_current_date else st.sidebar.date_input('End date', datetime(2020, 5, 31))

# Sidebar: New data organization
data_options = st.sidebar.radio('Select data to plot', ['Stock Price - OHLC', 'Returns & Performance', 'Additional Information'])

# OHLC and Candlesticks selection
if data_options == 'Stock Price - OHLC':
    ohlc_option = st.sidebar.selectbox('Select OHLC data', ['Candlesticks', 'OHLC'], index=0)

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
def calculate_returns(df):
    df['Stock Returns'] = df['Close'].pct_change()
    df['Log Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Cumulative Returns'] = (1 + df['Stock Returns']).cumprod() - 1
    df['Cumulative Log Returns'] = (1 + df['Log Returns']).cumprod() - 1

calculate_returns(tickerDf)
calculate_returns(sp500Df)

# Rolling Alpha and Beta
rolling_window = st.sidebar.slider('Lookback window for Alpha and Beta (in days)', min_value=1, max_value=504, value=252)

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
            fig.update_layout(xaxis_title='Date', yaxis_title='Price ($)', xaxis_rangeslider_visible=False, template='plotly_dark')
            st.plotly_chart(fig)
        else:
            st.subheader("OHLC")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Open'], mode='lines', name=f'{tickerSymbol} Open', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['High'], mode='lines', name=f'{tickerSymbol} High', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Low'], mode='lines', name=f'{tickerSymbol} Low', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Close'], mode='lines', name=f'{tickerSymbol} Close', line=dict(color='orange')))
            if compare_to_benchmark:
                fig.add_trace(go.Scatter(x=sp500Df['Date'], y=sp500Df['Close'], mode='lines', name='S&P 500 Close', line=dict(color='cyan')))
            fig.update_layout(xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
            st.plotly_chart(fig)

    elif data_options == 'Returns & Performance':
        st.subheader("Returns and Performance")
        metrics = ['Stock Returns', 'Log Returns', 'Cumulative Returns', 'Cumulative Log Returns', 'Rolling Alpha', 'Rolling Beta']
        for metric in metrics:
            if metric in tickerDf.columns:
                st.subheader(f"{metric}")
                fig = go.Figure()
                if metric == 'Rolling Alpha' or metric == 'Rolling Beta':
                    fig.add_trace(go.Scatter(x=tickerDf['Date'][rolling_window:], y=tickerDf[metric][rolling_window:], mode='lines', name=f'{companyName} ({tickerSymbol})', line=dict(color='white')))
                else:
                    fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf[metric], mode='lines', name=tickerSymbol, line=dict(color='#4bffb0')))
                if compare_to_benchmark:
                    if metric in sp500Df.columns:
                        fig.add_trace(go.Scatter(x=sp500Df['Date'], y=sp500Df[metric], mode='lines', name='^GSPC', line=dict(color='#4bffb0')))
                fig.update_layout(xaxis_title='Date', yaxis_title=f'{metric}', template='plotly_dark')
                st.plotly_chart(fig)

        st.subheader("Returns Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=tickerDf['Stock Returns'], name=tickerSymbol, marker=dict(color='white'), opacity=0.7, nbinsx=50))
        if compare_to_benchmark:
            fig.add_trace(go.Histogram(x=sp500Df['Stock Returns'], name='^GSPC', marker=dict(color='#4bffb0'), opacity=0.7, nbinsx=50))
        fig.update_layout(xaxis_title='Returns', yaxis_title='Frequency', barmode='overlay', template='plotly_dark')
        st.plotly_chart(fig)

    elif data_options == 'Additional Information':
        st.subheader("Volume")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Volume'], mode='lines', name=tickerSymbol, line=dict(color='#4bffb0')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Volume ($)', template='plotly_dark')
        st.plotly_chart(fig)

        st.subheader("Dividends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Dividends'], mode='lines', name=tickerSymbol, line=dict(color='white')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Dividends ($)', template='plotly_dark')
        st.plotly_chart(fig)


# Sidebar: Checkbox to compare to S&P 500 Index
compare_to_benchmark = st.sidebar.checkbox('Compare to S&P 500 Index (`^GSPC`)')

# Plot the data based on user selection
plot_data(tickerDf, sp500Df, data_options, ohlc_option if data_options == 'Stock Price - OHLC' else None, compare_to_benchmark)

# Footer
st.markdown("""
    <footer style='text-align: center; color: #ffffff; padding: 1rem; background-color: #1a1a1a;'>
        Developed by Yosri Ben Halima | Financial Data Analytics Platform © 2024
    </footer>
""", unsafe_allow_html=True)
