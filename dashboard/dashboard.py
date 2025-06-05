"""
Interactive financial dashboard using Streamlit.
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spark_processing.analyzer import FinancialDataAnalyzer

# Page config
st.set_page_config(
    page_title="Financial Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("Real-Time Financial Market Dashboard")
st.markdown("*A data engineering project for monitoring financial markets*")

# Function to load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(db_path):
    """Load and analyze data from the database."""
    analyzer = FinancialDataAnalyzer(db_path)
    results = analyzer.analyze_market_data()
    
    # Convert all Spark DataFrames to pandas DataFrames to ensure pickle-serializability
    for key in results:
        if hasattr(results[key], 'toPandas'):
            results[key] = results[key].toPandas()
    
    return results

# Database path (use test database if available, otherwise use main database)
if os.path.exists("test_finance.db"):
    db_path = "test_finance.db"
elif os.path.exists("finance.db"):
    db_path = "finance.db"
else:
    st.error("No database file found. Please run the market data producer first.")
    st.stop()

# Load data
data = load_data(db_path)
price_df = data['price_df']
ohlcv_df = data['ohlcv_df']
ma_df = data['ma_df']
rsi_df = data['rsi_df']
vol_df = data['vol_df']
market_summary = data['market_summary']
top_gainers = data['top_gainers']
top_losers = data['top_losers']

# Sidebar for stock selection
st.sidebar.header("Select Stock")
selected_symbol = st.sidebar.selectbox(
    "Choose a stock to analyze",
    options=price_df['symbol'].unique()
)

# Sidebar for technical indicators
st.sidebar.header("Technical Indicators")
show_sma = st.sidebar.checkbox("Show Simple Moving Averages", value=True)
show_rsi = st.sidebar.checkbox("Show Relative Strength Index", value=True)
show_vol = st.sidebar.checkbox("Show Volatility", value=True)

# Create dashboard layout
col1, col2 = st.columns([2, 1])

# Market summary in second column
with col2:
    st.subheader("Market Summary")
    st.metric(
        label="Average Price",
        value=f"${market_summary.iloc[0]['average_price']:.2f}"
    )
    st.metric(
        label="Average Change",
        value=f"{market_summary.iloc[0]['average_change_percent']:.2f}%",
        delta=f"{market_summary.iloc[0]['average_change_percent']:.2f}%"
    )
    st.metric(
        label="Gainers vs. Losers",
        value=f"{int(market_summary.iloc[0]['gainers'])} ↑ / {int(market_summary.iloc[0]['losers'])} ↓"
    )
    
    # Top gainers
    st.subheader("Top Gainers")
    for _, row in top_gainers.iterrows():
        st.metric(
            label=row['symbol'],
            value=f"${row['price']:.2f}",
            delta=f"{row['change_percent']:.2f}%"
        )
    
    # Top losers
    st.subheader("Top Losers")
    for _, row in top_losers.iterrows():
        st.metric(
            label=row['symbol'],
            value=f"${row['price']:.2f}",
            delta=f"{row['change_percent']:.2f}%"
        )

# Stock price chart in first column
with col1:
    # Selected stock information
    latest_price = price_df[price_df['symbol'] == selected_symbol].sort_values('timestamp').iloc[-1]
    
    st.subheader(f"{selected_symbol} - ${latest_price['price']:.2f}")
    st.metric(
        label="Change",
        value=f"${latest_price['change']:.2f}",
        delta=f"{latest_price['change_percent']:.2f}%"
    )
    
    # Price chart with technical indicators
    st.subheader(f"{selected_symbol} Price Chart with Technical Indicators")
    
    # Get data for the selected symbol
    symbol_ohlcv = ohlcv_df[ohlcv_df['symbol'] == selected_symbol].sort_values('timestamp')
    symbol_ma = ma_df[ma_df['symbol'] == selected_symbol].sort_values('timestamp')
    symbol_rsi = rsi_df[rsi_df['symbol'] == selected_symbol].sort_values('timestamp')
    symbol_vol = vol_df[vol_df['symbol'] == selected_symbol].sort_values('timestamp')
    
    # Create price chart
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=symbol_ohlcv['timestamp'],
            open=symbol_ohlcv['open'],
            high=symbol_ohlcv['high'],
            low=symbol_ohlcv['low'],
            close=symbol_ohlcv['close'],
            name="OHLC"
        )
    )
    
    # Add moving averages if selected
    if show_sma:
        fig.add_trace(
            go.Scatter(
                x=symbol_ma['timestamp'],
                y=symbol_ma['sma_5'],
                mode='lines',
                name='SMA 5',
                line=dict(color='blue', width=1)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=symbol_ma['timestamp'],
                y=symbol_ma['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1)
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{selected_symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI chart if selected
    if show_rsi:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(
            go.Scatter(
                x=symbol_rsi['timestamp'],
                y=symbol_rsi['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            )
        )
        
        # Add overbought/oversold lines
        rsi_fig.add_shape(
            type="line", 
            x0=symbol_rsi['timestamp'].iloc[0], 
            y0=70, 
            x1=symbol_rsi['timestamp'].iloc[-1], 
            y1=70,
            line=dict(color="red", width=1, dash="dash")
        )
        rsi_fig.add_shape(
            type="line", 
            x0=symbol_rsi['timestamp'].iloc[0], 
            y0=30, 
            x1=symbol_rsi['timestamp'].iloc[-1], 
            y1=30,
            line=dict(color="green", width=1, dash="dash")
        )
        
        rsi_fig.update_layout(
            title=f"{selected_symbol} Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI",
            height=200,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(rsi_fig, use_container_width=True)
    
    # Volatility chart if selected
    if show_vol:
        vol_fig = go.Figure()
        vol_fig.add_trace(
            go.Scatter(
                x=symbol_vol['timestamp'],
                y=symbol_vol['volatility'],
                mode='lines',
                name='Volatility',
                line=dict(color='red', width=2)
            )
        )
        
        vol_fig.update_layout(
            title=f"{selected_symbol} Volatility (20-day)",
            xaxis_title="Date",
            yaxis_title="Volatility",
            height=200
        )
        
        st.plotly_chart(vol_fig, use_container_width=True)

# Volume bar chart (below price chart)
vol_bar = go.Figure()
vol_bar.add_trace(
    go.Bar(
        x=symbol_ohlcv['timestamp'],
        y=symbol_ohlcv['volume'],
        name='Volume',
        marker=dict(color='rgba(0, 128, 0, 0.7)')
    )
)

vol_bar.update_layout(
    title=f"{selected_symbol} Trading Volume",
    xaxis_title="Date",
    yaxis_title="Volume",
    height=200
)

st.plotly_chart(vol_bar, use_container_width=True)

# Data tables section
st.header("Market Data Tables")
tab1, tab2, tab3 = st.tabs(["Price Data", "OHLCV Data", "Technical Indicators"])

with tab1:
    st.subheader("Latest Price Data")
    st.dataframe(price_df.sort_values(['symbol', 'timestamp'], ascending=[True, False]).head(20))

with tab2:
    st.subheader("OHLCV Data")
    st.dataframe(ohlcv_df.sort_values(['symbol', 'timestamp'], ascending=[True, False]).head(20))

with tab3:
    st.subheader("Technical Indicators")
    
    # Combine technical indicators for the selected symbol
    tech_indicators = symbol_ma.merge(
        symbol_rsi[['timestamp', 'rsi']], 
        on='timestamp'
    ).merge(
        symbol_vol[['timestamp', 'volatility']], 
        on='timestamp'
    )
    
    # Select relevant columns
    tech_df = tech_indicators[[
        'timestamp', 'open', 'high', 'low', 'close', 
        'sma_5', 'sma_10', 'sma_20', 'rsi', 'volatility'
    ]]
    
    st.dataframe(tech_df.sort_values('timestamp', ascending=False))

# Footer
st.markdown("---")
st.markdown("**Financial Data Platform** - Real-time data analytics for financial markets")
st.markdown("Data refreshes every 5 minutes. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
