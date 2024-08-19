import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# Set page layout to wide
st.set_page_config(layout="wide")

# Sidebar title
st.sidebar.header('Input Options')

# Fetch S&P 500 tickers and names
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
sp500_names = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Security'].tolist()

# Sidebar: Dropdown menu for selecting ticker
tickerSymbol = st.sidebar.selectbox('Select a Company', sp500_tickers, placeholder='Search Company by Ticker')
companyName = sp500_names[sp500_tickers.index(tickerSymbol)]

# Sidebar: Date input for start date
start_date = st.sidebar.date_input('Start date', datetime(2018, 5, 31))

# Sidebar: Checkbox for using the current date as the end date
use_current_date = st.sidebar.checkbox('Use current date as end date', value=True)

# Sidebar: Date input for end date (only if the checkbox is not checked)
if use_current_date:
    end_date = datetime.now()
else:
    end_date = st.sidebar.date_input('End date', datetime(2020, 5, 31))

# Sidebar: Multiselect box for selecting data to plot
data_options = [
    'Candlestick',
    'Open',
    'High',
    'Low',
    'Close',
    'Stock Returns',
    'Log Returns',
    'Cumulative Returns',
    'Cumulative Log Returns',
    'Rolling Alpha and Beta',
    'Returns Distribution',
    'Volume',
    'Dividends'
]
selected_data = st.sidebar.multiselect('Select data to plot', data_options, default=['Close'],placeholder='Select Data to Plot')

# Sidebar: Checkbox to compare to S&P 500 Index
compare_to_benchmark = st.sidebar.checkbox('Compare to S&P 500 Index (Benchmark)')

# Ensure end_date is after start_date
if not use_current_date and start_date > end_date: # type: ignore
    st.error('Error: End date must be after start date.')

# Fetch data for the selected ticker
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

# Fetch data for S&P 500 Index
sp500Data = yf.Ticker('^GSPC')
sp500Df = sp500Data.history(period='1d', start=start_date, end=end_date)

# Calculate Stock Returns and Log Returns and Cumulative Returns
tickerDf['Stock Returns'] = tickerDf['Close'].pct_change()
tickerDf['Log Returns'] = np.log(tickerDf['Close'] / tickerDf['Close'].shift(1))
tickerDf['Cumulative Returns'] = (1 + tickerDf['Stock Returns']).cumprod() - 1
tickerDf['Cumulative Log Returns'] = (1 + tickerDf['Log Returns']).cumprod() - 1

sp500Df['Stock Returns'] = sp500Df['Close'].pct_change()
sp500Df['Log Returns'] = np.log(sp500Df['Close'] / sp500Df['Close'].shift(1))
sp500Df['Cumulative Returns'] = (1 + sp500Df['Stock Returns']).cumprod() - 1
sp500Df['Cumulative Log Returns'] = (1 + sp500Df['Log Returns']).cumprod() - 1

# Calculate Rolling Alpha and Beta
rolling_window = st.sidebar.slider('Lookback window for Alpha and Beta (in days)', min_value=1, max_value=504, value=252) #252 e.g., 1 year of trading days

tickerDf['Rolling Alpha'] = np.nan
tickerDf['Rolling Beta'] = np.nan

for i in range(rolling_window, len(tickerDf)):
    window_stock_returns = tickerDf['Stock Returns'].iloc[i-rolling_window:i].dropna()
    window_benchmark_returns = sp500Df['Stock Returns'].iloc[i-rolling_window:i].dropna()
    
    if len(window_stock_returns) > 0 and len(window_benchmark_returns) > 0:
        model = LinearRegression()
        X = window_benchmark_returns.values.reshape(-1, 1) # type: ignore
        y = window_stock_returns.values
        model.fit(X, y) # type: ignore
        alpha = model.intercept_
        beta = model.coef_[0]
        tickerDf['Rolling Alpha'].iloc[i] = alpha
        tickerDf['Rolling Beta'].iloc[i] = beta

# Reset index to use Date as a column
tickerDf.reset_index(inplace=True)
sp500Df.reset_index(inplace=True)

# Main Page: Title of the app
st.write(f"""
# S&P 500 Stock Data | {companyName} ({tickerSymbol})
""")

# Plot the selected data
if 'Candlestick' in selected_data:
    st.write("""
    ## Candlestick Chart
    """)
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
                                     name='S&P 500'))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#F0F8FF'
    )
    st.plotly_chart(fig)

# Plot OHLC, Volume, Dividends, Stock Returns, Log Returns, and Stock Splits if not showing Candlestick
for data in selected_data:
    if data in tickerDf.columns and data != 'Candlestick':
        st.write(f"""
        ## {data}
        """)
        if data in ['Open', 'High', 'Low', 'Close', 'Stock Returns', 'Log Returns', 'Cumulative Returns','Cumulative Log Returns']:
            # Line chart for OHLC prices and returns
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf[data], mode='lines', name=f'{companyName} ({tickerSymbol})', line=dict(color='blue')))
            if compare_to_benchmark:
                fig.add_trace(go.Scatter(x=sp500Df['Date'], y=sp500Df[data], mode='lines', name='S&P 500 (Benchmark)', line=dict(color='orange')))
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title=f'{data}',
                plot_bgcolor='#F0F8FF'
            )
            st.plotly_chart(fig)

        elif data == 'Volume':
            # Bar chart for Volume
            fig = go.Figure()
            fig.add_trace(go.Bar(x=tickerDf['Date'], y=tickerDf['Volume'], name=f'{companyName} ({tickerSymbol})',  marker_color='blue'))
            if compare_to_benchmark:
                fig.add_trace(go.Bar(x=sp500Df['Date'], y=sp500Df['Volume'], name='S&P 500 (Benchmark)', marker_color='orange'))
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Volume',
                plot_bgcolor='#F0F8FF'
            )
            st.plotly_chart(fig)

        elif data == 'Dividends':
            # Bar chart for Dividends
            fig = go.Figure()
            fig.add_trace(go.Bar(x=tickerDf['Date'], y=tickerDf['Dividends'], name=f'{companyName} ({tickerSymbol})', marker_color='blue'))
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Dividends',
                plot_bgcolor='#F0F8FF'    
            )
            st.plotly_chart(fig)

# Plot Monthly Returns Distribution
if 'Returns Distribution' in selected_data:
    st.write("""
    ## Returns Distribution
    """)
    fig = go.Figure()

    # Plot stock returns distribution
    fig.add_trace(go.Histogram(x=tickerDf['Stock Returns'],
                               name=f'{companyName} ({tickerSymbol})',
                               marker=dict(color='blue'),
                               opacity=0.7,
                               nbinsx=50))
    
    # Plot S&P 500 returns distribution
    if compare_to_benchmark:
        fig.add_trace(go.Histogram(x=sp500Df['Stock Returns'],
                                   name='S&P 500 (Benchmark)',
                                   marker=dict(color='orange'),
                                   opacity=0.7,
                                   nbinsx=50))

    fig.update_layout(xaxis_title='Returns',
                      yaxis_title='Frequency',
                      barmode='overlay',
                      plot_bgcolor='#F0F8FF'
                      )
    st.plotly_chart(fig)

# Rolling Alpha and Beta
if 'Rolling Alpha and Beta' in selected_data:
    st.markdown("""
    ## Rolling Alpha
    """)
    fig_alpha = go.Figure()

    fig_alpha.add_trace(go.Scatter(x=tickerDf['Date'][rolling_window:],
                                   y=tickerDf['Rolling Alpha'][rolling_window:],
                                   mode='lines',
                                   name='Rolling Alpha',
                                   line=dict(color='blue')))
    
    # Add a dashed horizontal line at y=0
    fig_alpha.add_shape(
    go.layout.Shape(
        type="line",
        x0=tickerDf['Date'].tolist()[rolling_window], x1=tickerDf['Date'].tolist()[-1],
        y0=0, y1=0,
        line=dict(color="darkgrey", width=2, dash="dash")
        )
    )

    
    fig_alpha.update_layout(
        xaxis_title='Date',
        yaxis_title='Alpha',
        plot_bgcolor='#F0F8FF'
    )
    st.plotly_chart(fig_alpha)

# Plot Rolling Beta
if 'Rolling Alpha and Beta' in selected_data:
    st.write("""
    ## Rolling Beta
    """)
    fig_beta = go.Figure()

    fig_beta.add_trace(go.Scatter(x=tickerDf['Date'][rolling_window:],
                                  y=tickerDf['Rolling Beta'][rolling_window:],
                                  mode='lines',
                                  name='Rolling Beta',
                                  line=dict(color='orange')))
    
    # Add a dashed horizontal line at y=0
    fig_beta.add_shape(
    go.layout.Shape(
        type="line",
        x0=tickerDf['Date'].tolist()[rolling_window], x1=tickerDf['Date'].tolist()[-1],
        y0=0, y1=0,
        line=dict(color="darkgrey", width=2, dash="dash")
        )
    )

    # Add a dashed horizontal line at y=1
    fig_beta.add_shape(
    go.layout.Shape(
        type="line",
        x0=tickerDf['Date'].tolist()[rolling_window], x1=tickerDf['Date'].tolist()[-1],
        y0=1, y1=1,
        line=dict(color="darkgrey", width=2, dash="dash")
        )
    )
    
    fig_beta.update_layout(
        xaxis_title='Date',
        yaxis_title='Beta',
        plot_bgcolor='#F0F8FF'
    )
    st.plotly_chart(fig_beta)
