"""
Advanced financial dashboard using Streamlit.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import logging
import sqlite3
import pytz
from streamlit_autorefresh import st_autorefresh
import asyncio
from market_data.data.realtime_data import RealtimeDataFetcher

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="金融行情中心",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from market_data.data.analyzer import FinancialDataAnalyzer
# from spark_processing.analyzer import FinancialDataAnalyzer
from market_data.analytics.financial_metrics import FinancialMetrics
from market_data.analytics.anomaly_detection import MarketAnomalyDetector
from market_data.analytics.predictive_models import MarketPredictor
from market_data.storage.database_manager import DatabaseManager
from market_data.storage.database_manager_realtime import DatabaseManagerRealtime
from market_data.analysis.stock_analyzer import StockAnalyzer
# from mcp.market_chat_assistant import MarketChatAssistant

# Add custom CSS for the chatbot icon
st.markdown("""
<style>
.chatbot-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background-color: #1E88E5;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: all 0.2s ease;
    margin: 10px auto;
}

.chatbot-icon:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}

.chatbot-icon svg {
    width: 24px;
    height: 24px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Dashboard Configuration")

# Add chatbot icon to sidebar
st.sidebar.markdown("""
<div style="text-align: center;">
    <a href="http://localhost:8052" target="_blank" class="chatbot-icon">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
        </svg>
    </a>
</div>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define color scheme
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#FFC107',
    'positive': '#4CAF50',
    'negative': '#E53935',
    'neutral': '#757575',
    'background': '#F5F5F5',
    'text': '#212121'
}

# Title and description
st.title("金融市场行情中心")

# Database selection
db_files = [f for f in os.listdir('.') if f.endswith('.db')]
if not db_files:
    st.error("No database files found. Please run the market data producer first.")
    st.stop()

selected_db = st.sidebar.selectbox("Select Database", db_files, index=0)
db_path = selected_db
data_by_symbol={}
# Initialize database manager
db_manager = DatabaseManager(db_path)
db_manager_realtime=DatabaseManagerRealtime()

# Initialize RealtimeDataFetcher with token
realtime_fetcher = RealtimeDataFetcher(token="e3cc2496231dfdec51d5aa1c3f331891-c-app")

# Initialize stock analyzer
analyzer = StockAnalyzer(db_path)

# Get available symbols
try:
    # latest_prices = db_manager.get_latest_prices()
    # Import SYMBOLS from config
    from market_data.config import US_SYMBOLS, A_SHARE_SYMBOLS
    available_symbols = US_SYMBOLS + A_SHARE_SYMBOLS
    available_symbols_U=US_SYMBOLS
except Exception as e:
    st.error(f"Error loading symbols from config: {e}")
    available_symbols = []

# Add market selection
market_type = st.sidebar.radio(
    "选择市场",
    ["美股", "A股", "全部"],
    index=2
)

# Filter symbols based on market selection
if market_type == "美股":
    filtered_symbols = US_SYMBOLS
elif market_type == "A股":
    filtered_symbols = A_SHARE_SYMBOLS
else:
    filtered_symbols = available_symbols
if market_type=="A股":
    selected_symbols = st.sidebar.multiselect(
    "选择股票",
    options=filtered_symbols,
    default=filtered_symbols[:3] if len(filtered_symbols) >= 3 else filtered_symbols
    )

    latest_prices_A = db_manager_realtime.get_latest_prices(selected_symbols)
    # Create empty DataFrame with correct structure for US stocks
    latest_prices = pd.DataFrame(columns=['symbol', 'price', 'change', 'change_percent', 'volume', 'timestamp'])
# Symbol selection
else:
    selected_symbols = st.sidebar.multiselect(
        "选择股票",
        options=filtered_symbols,
        default=filtered_symbols[:3] if len(filtered_symbols) >= 3 else filtered_symbols

    )
    latest_prices = db_manager_realtime.get_latest_prices(selected_symbols)
if not selected_symbols:
    st.warning("请至少选择一只股票")
    st.stop()

# Get latest prices for selected symbols
# latest_prices = db_manager_realtime.get_latest_prices(selected_symbols)

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("市场表现")
    
    # Create a placeholder for volume display
    volume_placeholder = st.empty()
    
    # Display real-time volume for each stock
    with volume_placeholder.container():
        # Group stocks into sets of 3
        for i in range(0, len(selected_symbols), 3):
            group_symbols = selected_symbols[i:i+3]
            volume_cols = st.columns(len(group_symbols))
            for j, symbol in enumerate(group_symbols):
                with volume_cols[j]:
                    # Get data based on market type
                    if symbol in A_SHARE_SYMBOLS:
                        realtime_data = db_manager.get_latest_prices([symbol])
                        # Get real-time level 2 data for A-shares using AllTick API
                        level2_data = realtime_fetcher.get_level2_data(symbol)
                        logger.info(f"level2 data:{level2_data}!!!!!!!!!!")
                    else:
                        realtime_data = latest_prices[latest_prices['symbol'] == symbol]
           
                    if not realtime_data.empty:
                        volume = realtime_data.iloc[0]['volume']
                        # Format volume based on market type
                        if symbol in A_SHARE_SYMBOLS:
                            volume_str = f"{volume}手"  # Convert to 手 for A-shares
                        else:
                            volume_str = f"{volume:,}"  # Regular format for US stocks
                        st.metric(
                            label=f"{symbol} 成交量",
                            value=volume_str
                        )
                        
                        # Display level 2 data for A-shares
                        if symbol in A_SHARE_SYMBOLS and level2_data:
                            st.markdown("---")
                            st.markdown("**五档行情**")
                            
                            # Create a container for level 2 data
                            level2_container = st.container()
                            
                            with level2_container:
                                # Display bid data
                                st.markdown("**买盘**")
                                bid_data = []
                                for i in range(1, 6):
                                    price = level2_data[f'bid_price_{i}']
                                    volume = level2_data[f'bid_volume_{i}']
                                    if price > 0:  # Only show non-zero prices
                                        bid_data.append(f"买{i}: ¥{price:.2f} ({volume}手)")
                                st.markdown("\n".join(bid_data))
                                
                                # Display ask data
                                st.markdown("**卖盘**")
                                ask_data = []
                                for i in range(1, 6):
                                    price = level2_data[f'ask_price_{i}']
                                    volume = level2_data[f'ask_volume_{i}']
                                    if price > 0:  # Only show non-zero prices
                                        ask_data.append(f"卖{i}: ¥{price:.2f} ({volume}手)")
                                st.markdown("\n".join(ask_data))

    # Create a placeholder for price display
    price_placeholder = st.empty()
    
    # Display real-time prices for each stock
    with price_placeholder.container():
        # Group stocks into sets of 3
        for i in range(0, len(selected_symbols), 3):
            group_symbols = selected_symbols[i:i+3]
            price_cols = st.columns(len(group_symbols))
            for j, symbol in enumerate(group_symbols):
                with price_cols[j]:
                    # Get data based on market type
                    if symbol in A_SHARE_SYMBOLS:
                        realtime_data = db_manager.get_latest_prices([symbol])
                       
                    else:
                        realtime_data = latest_prices[latest_prices['symbol'] == symbol]
                    if not realtime_data.empty:
                        price = realtime_data.iloc[0]['price']
                        change = realtime_data.iloc[0]['change']
                        change_percent = realtime_data.iloc[0]['change_percent']
                        # Format price based on market type
                        if symbol in A_SHARE_SYMBOLS:
                            price_str = f"¥{price:.2f}"  # Use ¥ for A-shares
                        else:
                            price_str = f"${price:.2f}"  # Use $ for US stocks
                        st.metric(
                            label=f"{symbol} 价格",
                            value=price_str,
                            delta=f"{change_percent:.2f}%"
                        )

for symbol in selected_symbols:
    if symbol in A_SHARE_SYMBOLS:
        selected_symbols.remove(symbol)
latest_price_r = db_manager_realtime.get_latest_prices(selected_symbols)


if not available_symbols:
    st.error("No symbols found in the database. Please run the market data producer first.")
    st.stop()
# Custom stock input
st.sidebar.markdown("---")
st.sidebar.subheader("增加自定义股票")
custom_symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL, MSFT)").upper()

if custom_symbol:
    if custom_symbol in available_symbols:
        if custom_symbol not in selected_symbols:
            selected_symbols.append(custom_symbol)
            st.sidebar.success(f"Added {custom_symbol} to selected stocks!")
        else:
            st.sidebar.info(f"{custom_symbol} is already selected.")
    else:
        # Try to fetch data for this symbol
        try:
            # Import YahooFinance directly
            import yfinance as yf
            
            # Try to get quote from yfinance directly
            ticker = yf.Ticker(custom_symbol)
            info = ticker.info
            
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                # Update config.py to add new symbol
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'market_data', 'config.py')
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                # Find the SYMBOLS list and add the new symbol
                import re
                symbols_pattern = r'SYMBOLS\s*=\s*\[(.*?)\]'
                match = re.search(symbols_pattern, config_content, re.DOTALL)
                
                if match:
                    symbols_str = match.group(1)
                    symbols = [s.strip().strip('"\'') for s in symbols_str.split(',') if s.strip()]
                    
                    if custom_symbol not in symbols:
                        # Add new symbol to the list
                        symbols.append(custom_symbol)
                        # Sort symbols alphabetically
                        symbols.sort()
                        # Create new symbols string
                        new_symbols_str = ',\n    '.join([f'"{s}"' for s in symbols])
                        # Replace old symbols list with new one
                        new_config_content = re.sub(
                            symbols_pattern,
                            f'SYMBOLS = [\n    {new_symbols_str}\n]',
                            config_content,
                            flags=re.DOTALL
                        )
                        
                        # Write updated content back to config.py
                        with open(config_path, 'w') as f:
                            f.write(new_config_content)
                        logger.info(f"Added {custom_symbol} to config.py SYMBOLS list")
                
                # Get price information
                current_price = info.get('regularMarketPrice', 0)
                previous_close = info.get('previousClose', 0)
                
                # Calculate change
                change = current_price - previous_close
                change_percent = (change / previous_close * 100) if previous_close else 0
                
                # Create price data
                price_data = {
                    "symbol": custom_symbol,
                    "price": current_price,
                    "change": change,
                    "change_percent": change_percent,
                    "volume": info.get('regularMarketVolume', 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Get historical data directly   这里也会更新数据
                hist = ticker.history(period="1mo")
                
                # Convert to list of dictionaries
                historical_data = []
                for idx, row in hist.iterrows():
                    historical_data.append({
                        "symbol": custom_symbol,
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": int(row['Volume']),
                        "period": "1d",
                        "timestamp": idx.to_pydatetime().isoformat()
                    })
                
                # Save to database directly without validation
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Add price data
                cursor.execute('''
                INSERT INTO stock_prices 
                (symbol, price, change, change_percent, volume, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    price_data["symbol"],
                    price_data["price"],
                    price_data["change"],
                    price_data["change_percent"],
                    price_data["volume"],
                    price_data["timestamp"]
                ))
                
                # Add historical data
                for data in historical_data:
                    cursor.execute('''
                    INSERT INTO ohlcv 
                    (symbol, open, high, low, close, volume, period, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data["symbol"],
                        data["open"],
                        data["high"],
                        data["low"],
                        data["close"],
                        data["volume"],
                        data["period"],
                        data["timestamp"]
                    ))
                
                conn.commit()
                conn.close()
                
                # Add to available symbols
                available_symbols.append(custom_symbol)
                selected_symbols.append(custom_symbol)
                # for symbol in selected_symbols:
                #     data_by_symbol[symbol] = load_stock_data(symbol)
                
                # Update latest prices to include the new symbol
                latest_price_r = db_manager_realtime.get_latest_prices(selected_symbols)
                
                # Load data for the new symbol - FIXED
                # Instead of calling load_stock_data, directly create a DataFrame
                ohlcv_data = pd.DataFrame()
                try:
                    # Using direct DB query instead of load_stock_data
                    conn = sqlite3.connect(db_path)
                    query = f"SELECT * FROM ohlcv WHERE symbol = '{custom_symbol}'"
                    if start_date:
                        query += f" AND timestamp >= '{start_date.isoformat()}'"
                    if end_date:
                        query += f" AND timestamp <= '{end_date.isoformat()}'"
                    query += " ORDER BY timestamp ASC"
                    
                    ohlcv_data = pd.read_sql_query(query, conn)
                    
                    # Convert timestamp to datetime
                    if 'timestamp' in ohlcv_data.columns:
                        ohlcv_data['timestamp'] = pd.to_datetime(ohlcv_data['timestamp'])
                    
                    conn.close()
                except Exception as e:
                    st.sidebar.warning(f"Added stock but error loading data: {str(e)}")
                
                data_by_symbol[custom_symbol] = ohlcv_data
                
                # Force a page refresh to show the new data
                st.sidebar.success(f"Added {custom_symbol} to selected stocks! Refresh page to see data.")
                st.sidebar.button("Refresh Data")
            else:
                st.sidebar.error(f"Stock symbol '{custom_symbol}' not found")
        except Exception as e:
            st.sidebar.error(f"Error adding stock: {str(e)}")
# Date range selection
date_ranges = {
    "1 Day": timedelta(days=1),
    "1 Week": timedelta(days=7),
    "1 Month": timedelta(days=30),
    "3 Months": timedelta(days=90),
    "6 Months": timedelta(days=180),
    "1 Year": timedelta(days=365),
    "All": None
}

# Define interval mapping based on time range
interval_mapping = {
    "1 Day": "1m",      # 1小时间隔
    "1 Week": "1h",     # 1小时间隔
    "1 Month": "1d",    # 1天间隔
    "3 Months": "1d",   # 1天间隔
    "6 Months": "1d",   # 1天间隔
    "1 Year": "1d",     # 1天间隔
    "All": "1d"         # 1天间隔
}

selected_range = st.sidebar.selectbox("Select Time Range", list(date_ranges.keys()), index=2)
date_range = date_ranges[selected_range]
selected_interval = interval_mapping[selected_range]

# Calculate start and end dates
end_date = datetime.now()
logger.info(f"now time {end_date} is used")

# Set start_date based on selected range
if selected_range == "1 Day":
    # Convert end_date to Eastern Time
    eastern_tz = pytz.timezone('US/Eastern')
    end_date_eastern = end_date.astimezone(eastern_tz)
    ##
    # If current time is before market open, use previous trading day
    market_open = end_date_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    if end_date_eastern < market_open:
        # Go back to previous trading day
        end_date_eastern = end_date_eastern - timedelta(days=1)
        # Handle weekends differently
        if end_date_eastern.weekday() == 5:  # Saturday
            end_date_eastern = end_date_eastern - timedelta(days=1)  # Go back to Friday
        elif end_date_eastern.weekday() == 6:  # Sunday
            end_date_eastern = end_date_eastern - timedelta(days=2)  # Go back to Friday
        # Set time to market close (16:00)
        end_date_eastern = end_date_eastern.replace(hour=16, minute=0, second=0, microsecond=0)
    ##
    # Set start_date to market open of the same day
    start_date = end_date_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    end_date = end_date_eastern
else:
    start_date = end_date - date_range if date_range else None

# Technical indicator selection
st.sidebar.header("技术指标")
show_sma = st.sidebar.checkbox("简单移动平均线", value=True)
show_bollinger = st.sidebar.checkbox("布林带", value=False)
show_rsi = st.sidebar.checkbox("相对强弱指数 (RSI)", value=True)
show_macd = st.sidebar.checkbox("MACD", value=False)
show_volume = st.sidebar.checkbox("交易量", value=True)

# Analysis options
st.sidebar.header("分析选项")
show_anomalies = st.sidebar.checkbox("检测异常", value=True)
show_predictions = st.sidebar.checkbox("价格预测", value=True)
show_correlation = st.sidebar.checkbox("相关度分析", value=False)
show_stats = st.sidebar.checkbox("数据分析", value=True)

# Main content
# Create tabs for different sections
tabs = st.tabs([
    "市场总览", 
    "股票分析", 
    "技术指标", 
    "异常检测",
    "价格预测",
    "数据探索",
    "对话助手"  # Add new tab
])

# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stock_data(symbol, start_date=None, end_date=None):
    """Load stock data for a given symbol and date range."""
    try:
        if start_date:
            ohlcv_data = db_manager.get_historical_prices(symbol, start_date, end_date)
        else:
            ohlcv_data = db_manager.get_historical_prices(symbol)
        
        return ohlcv_data
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

# Load data for all selected symbols
data_by_symbol = {}
for symbol in selected_symbols:
    data_by_symbol[symbol] = load_stock_data(symbol, start_date, end_date)
# Add an auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 300, 60)
    
    # 会每 refresh_interval 秒自动刷新一次页面
    count = st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")

    st.sidebar.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
# Tab 1: Market Overview
with tabs[0]:
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    # logger.info(f"latest_price_r 列名：{latest_price_r.columns}  dffewfwef{latest_price_r}")
    # latest_data = latest_price_r[latest_price_r['symbol'].isin(selected_symbols)] if not latest_price_r.empty else pd.DataFrame(columns=['symbol', 'price', 'change', 'change_percent', 'volume', 'timestamp'])
    with col1:
        st.subheader("市场表现")
        
        # Add auto-refresh for 1-day view
        if selected_range == "1 Day":
            # Add a placeholder for the last update time
            last_update = st.empty()
            
            # Create a placeholder for the chart
            chart_placeholder = st.empty()
            
            # Update last update time
            last_update.write(f"Last update: {datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M:%S')} ET")
            
            # Define a list of colors for different stocks
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # Function to update the chart
            def update_chart():
                # Get latest real-time prices
                latest_prices = db_manager_realtime.get_realtime_prices(selected_symbols)
                logger.info(f"now latest_prices is {latest_prices['timestamp']}")
                
                # Create performance chart
                fig = go.Figure()
                
                for i, symbol in enumerate(selected_symbols):
                    logger.info(f"now symbol_data is {start_date}!!!!!!!and end_date is {end_date}")
                    # Get historical stock prices
                    # symbol_data = db_manager_realtime.get_historical_stock_prices(
                    #     symbol, 
                    #     start_date, 
                    #     end_date,
                    #     interval=selected_interval
                    # )
                    # symbol_data = db_manager_realtime.get_historical_stock_prices(
                    #     symbol, 
                    #     "2025-5-23", 
                    #     "2025-5-24",
                    #     interval=selected_interval
                    # )
                    symbol_data = db_manager_realtime.get_historical_stock_prices(
                        symbol, 
                        start_date, 
                        end_date,
                        interval=selected_interval
                    )
                    
                    if not symbol_data.empty:
                        # Convert timestamp to Eastern Time
                        timestamps = pd.to_datetime(symbol_data['timestamp'])
                        logger.info(f"now symbol_data is {symbol_data.tail()}!!!!!!!and end_date is {end_date}")
                        if timestamps.dt.tz is None:
                            symbol_data['timestamp'] = timestamps.dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                        else:
                            symbol_data['timestamp'] = timestamps.dt.tz_convert('US/Eastern')
                        
                        # Get real-time data for this symbol
                        realtime_data = latest_prices[latest_prices['symbol'] == symbol]
                        
                        if not realtime_data.empty:
                            # Convert real-time data to match historical data format
                            latest_price = realtime_data
                            latest_row = pd.DataFrame({
                                'symbol': [symbol],
                                'price': [latest_price['price']],
                                'change': [latest_price['change']],
                                'change_percent': [latest_price['change_percent']],
                                'volume': [latest_price['volume']],
                                'timestamp': [latest_price['timestamp']]
                            })
                            
                            # Append real-time data to historical data
                            symbol_data = pd.concat([symbol_data, latest_row], ignore_index=True)
                            symbol_data = symbol_data.sort_values('timestamp')
                        
                        # Normalize price to start at 100 for comparison
                        first_price = symbol_data['price'].iloc[0]
                        normalized_price = symbol_data['price'] / first_price * 100
                        
                        # Add trace to the figure
                        fig.add_trace(go.Scatter(
                            x=symbol_data['timestamp'],
                            y=normalized_price,
                            mode='lines',
                            name=symbol,
                            line=dict(width=2)
                        ))
                
                # Update layout after all traces are added
                fig.update_layout(
                    title="相对表现 (归一化到 100)",
                    xaxis_title="日期",
                    yaxis_title="归一化价格",
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Update last update time
                last_update.write(f"Last update: {datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M:%S')} ET")
            
            # Initial chart update
            update_chart()
        # Create the figure before the loop
        fig = go.Figure()
        logger.info(f"now symbols{selected_symbols}")
        if selected_range != "1 Day":
            for symbol in selected_symbols:
            # Get historical stock prices instead of OHLCV data with appropriate interval
                symbol_data = db_manager_realtime.get_historical_stock_prices(
                    symbol, 
                    start_date, 
                    end_date,
                    interval=selected_interval
                )
        
                
                if not symbol_data.empty:
                # Convert timestamp to Eastern Time (handle both tz-aware and tz-naive timestamps)
                    timestamps = pd.to_datetime(symbol_data['timestamp'])
                    logger.info(f"{symbol}is not empty")
                    if timestamps.dt.tz is None:
                        symbol_data['timestamp'] = timestamps.dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                    else:
                        symbol_data['timestamp'] = timestamps.dt.tz_convert('US/Eastern')
                
                # Normalize price to start at 100 for comparison
                    first_price = symbol_data['price'].iloc[0]
                    normalized_price = symbol_data['price'] / first_price * 100
                
                # Add trace to the existing figure
                    fig.add_trace(go.Scatter(
                        x=symbol_data['timestamp'],
                        y=normalized_price,
                        mode='lines',
                        name=symbol,
                        line=dict(width=2)
                    ))
        
        # Update layout after all traces are added
            fig.update_layout(
                title="相对表现 (归一化到 100)",
                xaxis_title="日期",
                yaxis_title="归一化价格",
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        
            st.plotly_chart(fig, use_container_width=True)

        # Create a placeholder for volume display
        volume_placeholder = st.empty()
        
        # Display real-time volume for each stock
        with volume_placeholder.container():
            # Group stocks into sets of 4
            for i in range(0, len(selected_symbols), 3):
                group_symbols = selected_symbols[i:i+3]
                volume_cols = st.columns(len(group_symbols))
                for j, symbol in enumerate(group_symbols):
                    with volume_cols[j]:
                        realtime_data = latest_prices[latest_prices['symbol'] == symbol]
                        if not realtime_data.empty:
                            volume = realtime_data.iloc[0]['volume']
                            st.metric(
                                label=f"{symbol} Volume",
                                value=f"{volume:,}"  # Add thousand separator
                            )

        # Create a placeholder for price display
        price_placeholder = st.empty()
        
        # Display real-time prices for each stock
        with price_placeholder.container():
            # Group stocks into sets of 4
            for i in range(0, len(selected_symbols), 3):
                group_symbols = selected_symbols[i:i+3]
                price_cols = st.columns(len(group_symbols))
                for j, symbol in enumerate(group_symbols):
                    with price_cols[j]:
                        realtime_data = latest_prices[latest_prices['symbol'] == symbol]
                        if not realtime_data.empty:
                            price = realtime_data.iloc[0]['price']
                            change = realtime_data.iloc[0]['change']
                            change_percent = realtime_data.iloc[0]['change_percent']
                            st.metric(
                                label=f"{symbol} Price",
                                value=f"${price:.2f}",
                                delta=f"{change_percent:.2f}%"
                            )
        
        # Add auto-refresh JavaScript
        # st.markdown(
        #     """
        #     <script>
        #         setInterval(function() {
        #             window.location.reload();
        #         }, 2000);  // Refresh every 2 seconds
        #     </script>
        #     """,
        #     unsafe_allow_html=True
        # )

        # Show correlation matrix if selected
        if show_correlation and len(selected_symbols) > 1:
            st.subheader("相关矩阵")
            
            # Create DataFrame with close prices for all symbols
            price_df = pd.DataFrame()
            
            for symbol in selected_symbols:
                if not data_by_symbol[symbol].empty:
                    price_df[symbol] = data_by_symbol[symbol].set_index('timestamp')['close']
            
            # Calculate correlation matrix
            corr_matrix = price_df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=px.colors.diverging.RdBu_r,
                color_continuous_midpoint=0
            )
            
            fig.update_layout(
                title="价格相关矩阵",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        # Add portfolio management section
        st.subheader("投资组合管理")
        
        # Create a form for portfolio updates
        with st.form("portfolio_update_form"):
            # Get current portfolio
            portfolio_df = db_manager_realtime.get_portfolio()
            
            # Create columns for the form
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                selected_stock = st.selectbox(
                    "Select Stock",
                    options=selected_symbols,
                    key="portfolio_stock"
                )
            
            with form_col2:
                shares = st.number_input(
                    "Number of Shares",
                    min_value=0,
                    value=0,
                    step=1
                )
            
            # Submit button
            submitted = st.form_submit_button("Update Position")

            # Add portfolio summary section with auto-refresh
            st.subheader("投资组合总结")
        
        # Create placeholders for portfolio data
            portfolio_value_placeholder = st.empty()
            portfolio_table_placeholder = st.empty()
            
            if submitted:
                if shares > 0:
                    success = db_manager_realtime.update_portfolio(
                        selected_stock,
                        shares
                    )
                    if success:
                        st.success(f"Updated position for {selected_stock}")
                    else:
                        st.error(f"Failed to update position for {selected_stock}")
                else:
                    st.warning("Please enter valid number of shares")

        # Function to update portfolio display
        def update_portfolio_display():
            # Get portfolio summary
            portfolio_summary = db_manager_realtime.get_portfolio_summary()
            
            # Display portfolio metrics
            portfolio_value_placeholder.metric(
                label="Total Portfolio Value",
                value=f"${portfolio_summary['total_value']:,.2f}",
                delta=f"${portfolio_summary['value_change']:,.2f}"
            )
            
            # Get and display current positions
            portfolio_df = db_manager_realtime.get_portfolio()
            if not portfolio_df.empty:
                # Format the DataFrame for display
                display_df = portfolio_df.copy()
                display_df['market_value'] = display_df['market_value'].map('${:,.2f}'.format)
                display_df['current_price'] = display_df['current_price'].map('${:.2f}'.format)
                
                # Display the positions table
                portfolio_table_placeholder.dataframe(
                    display_df[['symbol', 'shares', 'current_price', 'market_value']],
                    hide_index=True
                )
            else:
                portfolio_table_placeholder.info("No positions in portfolio")

        # Add Top Movers section
        st.subheader("市场异动榜")
        
        # Create placeholders for top movers data
        top_movers_placeholder = st.empty()
        
        def update_top_movers_display():
            # Get current top movers data
            top_gainers, top_losers, is_market_open = analyzer.get_current_top_movers()
            
            # Create columns for gainers and losers
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 涨幅榜")
                if not top_gainers.empty:
                    # Filter out A-share stocks
                    us_gainers = top_gainers[~top_gainers['symbol'].str.endswith(('.SH', '.SZ'))]
                    if not us_gainers.empty:
                        gainers_df = us_gainers[['symbol', 'price', 'change_percent']].copy()
                        gainers_df['price'] = gainers_df['price'].map('${:.2f}'.format)
                        gainers_df['change_percent'] = gainers_df['change_percent'].map('{:.2f}%'.format)
                        st.dataframe(gainers_df, hide_index=True)
                    else:
                        st.info("No US stock gainers data available")
                else:
                    st.info("No gainers data available")
            
            with col2:
                st.markdown("### 跌幅榜")
                if not top_losers.empty:
                    # Filter out A-share stocks
                    us_losers = top_losers[~top_losers['symbol'].str.endswith(('.SH', '.SZ'))]
                    if not us_losers.empty:
                        losers_df = us_losers[['symbol', 'price', 'change_percent']].copy()
                        losers_df['price'] = losers_df['price'].map('${:.2f}'.format)
                        losers_df['change_percent'] = losers_df['change_percent'].map('{:.2f}%'.format)
                        st.dataframe(losers_df, hide_index=True)
                    else:
                        st.info("No US stock losers data available")
                else:
                    st.info("No losers data available")
            
            # Show market status
            st.markdown(f"**市场状态:** {'Open' if is_market_open else 'Closed'}")
            st.markdown(f"**最后更新:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initial display for both sections
        if market_type!="A股":
            update_portfolio_display()
            update_top_movers_display()
        
        # Add auto-refresh for both sections
        st.markdown(
            """
            <script>
                setInterval(function() {
                    window.location.reload();
                }, 60000);  // Refresh every minute
            </script>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.subheader("市场总结")
        
        # Calculate market statistics
        avg_price = latest_prices['price'].mean()
        logger.info(f"all prices: {latest_prices['price']}")

        avg_change = latest_prices['change_percent'].mean()
        gainers = len(latest_prices[latest_prices['change'] > 0])
        losers = len(latest_prices[latest_prices['change'] < 0])
        total_volume = latest_prices['volume'].sum()
        
        # Display metrics
        st.metric(
            label="Average Price",
            value=f"${avg_price:.2f}"
        )
        st.metric(
            label="Average Change",
            value=f"{avg_change:.2f}%",
            delta=f"{avg_change:.2f}%"
        )
        st.metric(
            label="Gainers vs. Losers",
            value=f"{gainers} ↑ / {losers} ↓"
        )
        st.metric(
            label="Total Volume",
            value=f"{total_volume:,}"
        )
        
        # Top gainers and losers
        st.subheader("涨幅榜")
        top_gainers = latest_prices.sort_values('change_percent', ascending=False).head(3)
        for _, row in top_gainers.iterrows():
            st.metric(
                label=row['symbol'],
                value=f"${row['price']:.2f}",
                delta=f"{row['change_percent']:.2f}%"
            )
        
        st.subheader("跌幅榜")
        # Filter for stocks with negative change_percent
        losers_data = latest_prices[latest_prices['change_percent'] < 0]
        if not losers_data.empty:
            top_losers = losers_data.sort_values('change_percent', ascending=True).head(3)
            for _, row in top_losers.iterrows():
                st.metric(
                    label=row['symbol'],
                    value=f"${row['price']:.2f}",
                    delta=f"{row['change_percent']:.2f}%"
                )
        else:
            st.info("No stocks with negative change")

        
    
# Tab 2: Stock Analysis
with tabs[1]:
    # Select a symbol for detailed analysis
    selected_symbol_analysis = st.selectbox(
        "Select a stock for detailed analysis",
        options=selected_symbols
    )
    # for symbol in selected_symbols:
    #     data_by_symbol[symbol] = load_stock_data(symbol, start_date, end_date)
    
    # Load data for selected symbol
    stock_data = data_by_symbol[selected_symbol_analysis]
    
    if stock_data.empty:
        st.warning(f"No data available for {selected_symbol_analysis}")
    else:
        # Get latest price information

        latest_price = latest_prices[latest_prices['symbol'] == selected_symbol_analysis].iloc[0]
        
        # Create columns for layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="实时价格",
                value=f"${latest_price['price']:.2f}",
                delta=f"{latest_price['change']:.2f}"
            )
        
        with col2:
            st.metric(
                label="变化率",
                value=f"{latest_price['change_percent']:.2f}%",
                delta=f"{latest_price['change_percent']:.2f}%"
            )
        
        with col3:
            st.metric(
                label="实时交易量",
                value=f"{latest_price['volume']:,}"
            )
        
        # Price chart
        # st.subheader(f"{selected_symbol_analysis} 价格图表")
        
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=stock_data['timestamp'],
                open=stock_data['open'],
                high=stock_data['high'],
                low=stock_data['low'],
                close=stock_data['close'],
                name="OHLC"
            )
        )
        # Add moving averages if selected
        if show_sma:
            # Calculate SMAs
            stock_data['sma_20'] = stock_data['close'].rolling(window=20).mean()
            stock_data['sma_50'] = stock_data['close'].rolling(window=50).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['timestamp'],
                    y=stock_data['sma_20'],
                    mode='lines',
                    name='SMA (20)',
                    line=dict(color='blue', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['timestamp'],
                    y=stock_data['sma_50'],
                    mode='lines',
                    name='SMA (50)',
                    line=dict(color='orange', width=1)
                )
            )
        
        # Add Bollinger Bands if selected
        if show_bollinger:
            # Calculate Bollinger Bands
            window = 20
            stock_data['middle_band'] = stock_data['close'].rolling(window=window).mean()
            stock_data['std'] = stock_data['close'].rolling(window=window).std()
            stock_data['upper_band'] = stock_data['middle_band'] + 2 * stock_data['std']
            stock_data['lower_band'] = stock_data['middle_band'] - 2 * stock_data['std']
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['timestamp'],
                    y=stock_data['upper_band'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='rgba(0, 255, 0, 0.6)', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['timestamp'],
                    y=stock_data['middle_band'],
                    mode='lines',
                    name='Middle Band (SMA 20)',
                    line=dict(color='rgba(0, 0, 255, 0.6)', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['timestamp'],
                    y=stock_data['lower_band'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='rgba(255, 0, 0, 0.6)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.2)'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_symbol_analysis} 价格图表",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart if selected
        if show_volume:
            vol_fig = go.Figure()
            
            vol_fig.add_trace(
                go.Bar(
                    x=stock_data['timestamp'],
                    y=stock_data['volume'],
                    name='Volume',
                    marker=dict(color='rgba(0, 128, 0, 0.7)')
                )
            )
            
            vol_fig.update_layout( 
                title=f"{selected_symbol_analysis} 交易量",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            
            st.plotly_chart(vol_fig, use_container_width=True)
        
        # Statistical analysis if selected
        if show_stats:
            st.subheader("数据分析")
            
            # Calculate returns
            stock_data['daily_return'] = stock_data['close'].pct_change() * 100
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Summary statistics
                stats = stock_data['daily_return'].describe()
                
                metrics = {
                    "平均每日收益": f"{stats['mean']:.2f}%",
                    "收益率波动": f"{stats['std']:.2f}%",
                    "最小收益": f"{stats['min']:.2f}%",
                    "最大收益": f"{stats['max']:.2f}%"
                }
                
                for label, value in metrics.items():
                    st.metric(label=label, value=value)
            
            with col2:
                # Return distribution
                fig = px.histogram(
                    stock_data,
                    x='daily_return',
                    nbins=30,
                    title="收益率分布特征",
                    labels={'daily_return': 'Daily Return (%)'}
                )
                
                # Add vertical line at mean
                fig.add_vline(
                    x=stats['mean'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean",
                    annotation_position="top"
                )
                
                fig.update_layout(height=300)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate cumulative returns
            stock_data['cumulative_return'] = (1 + stock_data['daily_return'] / 100).cumprod() * 100 - 100
            
            # Plot cumulative returns
            fig = px.line(
                stock_data,
                x='timestamp',
                y='cumulative_return',
                title=f"{selected_symbol_analysis} 累计收益 (%)",
                labels={'cumulative_return': 'Cumulative Return (%)', 'timestamp': 'Date'}
            )
            
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
# Tab 3: Technical Indicators
with tabs[2]:
    # Select a symbol for technical analysis
    selected_symbol_tech = st.selectbox(
        "Select a stock for technical analysis",
        options=selected_symbols,
        key="tech_analysis_symbol"
    )
    logger.info(f"now symbols are{selected_symbol_analysis}")
    # Load data for selected symbol
    tech_data = data_by_symbol[selected_symbol_tech]
    
    if tech_data.empty:
        st.warning(f"No data available for {selected_symbol_tech}")
    else:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI
            if show_rsi:
                st.subheader("相对强弱指数 (RSI)")
                
                # Calculate RSI using Financial Metrics
                close_prices = tech_data['close'].values
                rsi_values = FinancialMetrics.calculate_rsi(close_prices)
                tech_data['rsi'] = rsi_values
                
                # Create RSI chart
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=tech_data['timestamp'],
                        y=tech_data['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    )
                )
                
                # Add overbought/oversold lines
                fig.add_shape(
                    type="line",
                    x0=tech_data['timestamp'].iloc[0],
                    y0=70,
                    x1=tech_data['timestamp'].iloc[-1],
                    y1=70,
                    line=dict(color="red", width=1, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=tech_data['timestamp'].iloc[0],
                    y0=30,
                    x1=tech_data['timestamp'].iloc[-1],
                    y1=30,
                    line=dict(color="green", width=1, dash="dash")
                )
                
                fig.update_layout(
                    title=f"{selected_symbol_tech} RSI (14)",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            logger.info("now MACD")
            # MACD
            # if show_macd:
            #     st.subheader("Moving Average Convergence Divergence (MACD)")
                
                # Calculate MACD using Financial Metrics
                # MACD
            if show_macd:
                    st.subheader("移动平均收敛曲线 (MACD)")
                    
                    # Get close prices
                    close_prices = tech_data['close'].values
                    
                    # Check if we have enough data for MACD
                    if len(close_prices) >= 26:
                        # Calculate MACD using Financial Metrics
                        try:
                            macd_data = FinancialMetrics.calculate_macd(close_prices)
                            tech_data['macd'] = macd_data['macd']
                            tech_data['signal'] = macd_data['signal']
                            tech_data['histogram'] = macd_data['histogram']
                            
                            # Create MACD chart
                            fig = make_subplots(rows=1, cols=1)
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=tech_data['timestamp'],
                                    y=tech_data['macd'],
                                    mode='lines',
                                    name='MACD',
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=tech_data['timestamp'],
                                    y=tech_data['signal'],
                                    mode='lines',
                                    name='Signal',
                                    line=dict(color='red', width=1)
                                )
                            )
                            
                            # Add histogram if available
                            if 'histogram' in tech_data.columns:
                                colors = ['green' if val >= 0 else 'red' for val in tech_data['histogram']]
                                
                                fig.add_trace(
                                    go.Bar(
                                        x=tech_data['timestamp'],
                                        y=tech_data['histogram'],
                                        name='Histogram',
                                        marker=dict(color=colors)
                                    )
                                )
                            
                            fig.update_layout(
                                title=f"{selected_symbol_tech} MACD (12,26,9)",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                height=300,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error calculating MACD: {str(e)}")
                            st.info("MACD calculation requires more data than is currently available.")
                    else:
                        # Not enough data
                        st.warning(f"MACD calculation requires at least 26 data points. Currently only have {len(close_prices)} points.")
                        logger.info(f"currently available{len(close_prices)}")
                        # Show a simplified version if possible
                        if len(close_prices) >= 10:
                            # Calculate simple moving averages as an alternative
                            short_ma = tech_data['close'].rolling(window=5).mean()
                            long_ma = tech_data['close'].rolling(window=10).mean()
                            tech_data['simple_macd'] = short_ma - long_ma
                            
                            # Create simple chart
                            fig = go.Figure()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=tech_data['timestamp'],
                                    y=tech_data['simple_macd'],
                                    mode='lines',
                                    name='Simple MACD (5,10)',
                                    line=dict(color='purple', width=2)
                                )
                            )
                            
                            fig.update_layout(
                                title=f"{selected_symbol_tech} Simplified MACD (5,10)",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.info("Showing simplified MACD using shorter periods (5,10) instead of standard (12,26,9)")
                

        
        # Bollinger Bands
        if show_bollinger:
            st.subheader("布林带")
            
            # Calculate Bollinger Bands using Financial Metrics
            close_prices = tech_data['close'].values
            bb_data = FinancialMetrics.calculate_bollinger_bands(close_prices)
            
            tech_data['middle_band'] = bb_data['middle']
            tech_data['upper_band'] = bb_data['upper']
            tech_data['lower_band'] = bb_data['lower']
            
            # Create Bollinger Bands chart
            fig = go.Figure()
            
            # Add price
            fig.add_trace(
                go.Scatter(
                    x=tech_data['timestamp'],
                    y=tech_data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='lightskyblue', width=1)
                )
            )
            
            # Add bands
            fig.add_trace(
                go.Scatter(
                    x=tech_data['timestamp'],
                    y=tech_data['upper_band'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='rgba(0, 255, 0, 0.6)', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data['timestamp'],
                    y=tech_data['middle_band'],
                    mode='lines',
                    name='Middle Band (SMA 20)',
                    line=dict(color='rgba(0, 0, 255, 0.6)', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data['timestamp'],
                    y=tech_data['lower_band'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='rgba(255, 0, 0, 0.6)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.2)'
                )
            )
            
            fig.update_layout(
                title=f"{selected_symbol_tech} Bollinger Bands (20,2)",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Technical analysis summary
        st.subheader("技术分析总结")
        
        # Create a dataframe with the latest values
        latest_tech = tech_data.iloc[-1]
        
        # Create columns for layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Close Price",
                value=f"${latest_tech['close']:.2f}"
            )
            
            if 'sma_20' in tech_data.columns and 'sma_50' in tech_data.columns:
                # Calculate SMAs if not already calculated
                if 'sma_20' not in tech_data.columns:
                    tech_data['sma_20'] = tech_data['close'].rolling(window=20).mean()
                if 'sma_50' not in tech_data.columns:
                    tech_data['sma_50'] = tech_data['close'].rolling(window=50).mean()
                    
                latest_tech = tech_data.iloc[-1]
                
                st.metric(
                    label="SMA (20)",
                    value=f"${latest_tech['sma_20']:.2f}",
                    delta=f"{(latest_tech['close'] - latest_tech['sma_20']):.2f}"
                )
                
                st.metric(
                    label="SMA (50)",
                    value=f"${latest_tech['sma_50']:.2f}",
                    delta=f"{(latest_tech['close'] - latest_tech['sma_50']):.2f}"
                )
        
        with col2:
            if 'rsi' in tech_data.columns:
                rsi_value = latest_tech['rsi']
                rsi_color = (
                    "red" if rsi_value > 70 else 
                    "green" if rsi_value < 30 else 
                    "white"
                )
                
                st.markdown(f"<h3 style='color:{rsi_color}'>RSI (14): {rsi_value:.2f}</h3>", unsafe_allow_html=True)
                
                rsi_signal = (
                    "超买" if rsi_value > 70 else
                    "超卖" if rsi_value < 30 else
                    "中性"
                )
                
                st.markdown(f"**Signal:** {rsi_signal}")
            
            if 'macd' in tech_data.columns and 'signal' in tech_data.columns:
                macd_value = latest_tech['macd']
                signal_value = latest_tech['signal']
                macd_color = "green" if macd_value > signal_value else "red"
                
                st.markdown(f"<h3 style='color:{macd_color}'>MACD: {macd_value:.4f}</h3>", unsafe_allow_html=True)
                
                macd_signal = (
                    "看涨" if macd_value > signal_value else
                    "看跌"
                )
                
                st.markdown(f"**Signal:** {macd_signal}")
        
        with col3:
            if 'upper_band' in tech_data.columns and 'lower_band' in tech_data.columns:
                upper_band = latest_tech['upper_band']
                lower_band = latest_tech['lower_band']
                close_price = latest_tech['close']
                
                bb_position = (close_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0.5
                
                bb_signal = (
                    "超买" if close_price > upper_band else
                    "超卖" if close_price < lower_band else
                    "中性"
                )
                
                st.markdown(f"<h3>BB Position: {bb_position:.2f}</h3>", unsafe_allow_html=True)
                st.markdown(f"**Signal:** {bb_signal}")
# Tab 4: Anomaly Detection
with tabs[3]:
    if show_anomalies:
        st.subheader("市场异常检测")
        
        # Select symbol for anomaly detection
        anomaly_symbol = st.selectbox(
            "Select a stock for anomaly detection",
            options=selected_symbols,
            key="anomaly_symbol"
        )
        
        # Load data for selected symbol
        anomaly_data = data_by_symbol[anomaly_symbol]
        
        if anomaly_data.empty:
            st.warning(f"No data available for {anomaly_symbol}")
        else:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Parameters for anomaly detection
            with col1:
                st.subheader("检测参数")
                
                anomaly_window = st.slider(
                    "检测窗口",
                    min_value=5,
                    max_value=60,
                    value=20,
                    step=5,
                    help="Number of periods to use for baseline calculation"
                )
                
                anomaly_threshold = st.slider(
                    "异常阈值",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    help="Threshold in standard deviations for anomaly detection"
                )
                
                anomaly_types = st.multiselect(
                    "异常监测指标",
                    options=["Price", "Volume", "Volatility", "Momentum"],
                    default=["Price", "Volume"]
                )
            
            with col2:
                st.subheader("检测结果")
                
                # Initialize anomaly detector
                detector = MarketAnomalyDetector()
                
                # Detect price anomalies
                total_anomalies = 0
                
                if "Price" in anomaly_types:
                    price_anomalies = detector.detect_price_anomalies(
                        anomaly_data['close'].values,
                        window=anomaly_window,
                        threshold=anomaly_threshold
                    )
                    
                    anomaly_data['price_anomaly'] = price_anomalies
                    price_anomaly_count = np.sum(price_anomalies)
                    total_anomalies += price_anomaly_count
                    
                    st.metric(
                        label="价格异常",
                        value=price_anomaly_count
                    )
                
                # Detect volume anomalies
                if "Volume" in anomaly_types:
                    volume_anomalies = detector.detect_volume_anomalies(
                        anomaly_data['volume'].values,
                        window=anomaly_window,
                        threshold=anomaly_threshold
                    )
                    
                    anomaly_data['volume_anomaly'] = volume_anomalies
                    volume_anomaly_count = np.sum(volume_anomalies)
                    total_anomalies += volume_anomaly_count
                    
                    st.metric(
                        label="交易量异常",
                        value=volume_anomaly_count
                    )
                
                # Detect volatility regime changes
                if "Volatility" in anomaly_types:
                    # Calculate returns
                    returns = np.diff(anomaly_data['close'].values) / anomaly_data['close'].values[:-1]
                    # Add a 0 at the beginning to maintain the same array length
                    returns = np.insert(returns, 0, 0)
                    
                    volatility_regimes = detector.detect_volatility_regime_changes(
                        returns,
                        window=anomaly_window,
                        threshold=anomaly_threshold
                    )
                    
                    anomaly_data['high_volatility'] = volatility_regimes['high_volatility']
                    anomaly_data['low_volatility'] = volatility_regimes['low_volatility']
                    
                    vol_anomaly_count = np.sum(volatility_regimes['high_volatility']) + np.sum(volatility_regimes['low_volatility'])
                    total_anomalies += vol_anomaly_count
                    
                    st.metric(
                        label="Volatility Regime Changes",
                        value=vol_anomaly_count
                    )
                
                # Detect momentum anomalies
                if "Momentum" in anomaly_types:
                    momentum_anomalies = detector.detect_momentum_anomalies(
                        anomaly_data['close'].values,
                        short_window=5,
                        long_window=20,
                        threshold=anomaly_threshold
                    )
                    
                    anomaly_data['positive_momentum'] = momentum_anomalies['positive_momentum']
                    anomaly_data['negative_momentum'] = momentum_anomalies['negative_momentum']
                    
                    momentum_anomaly_count = np.sum(momentum_anomalies['positive_momentum']) + np.sum(momentum_anomalies['negative_momentum'])
                    total_anomalies += momentum_anomaly_count
                    
                    st.metric(
                        label="Momentum Anomalies",
                        value=momentum_anomaly_count
                    )
                
                st.metric(
                    label="总异常",
                    value=total_anomalies
                )
            
            # Visualization of anomalies
            st.subheader("异常可视化")
            
            # Create price chart with anomalies
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=anomaly_data['timestamp'],
                    y=anomaly_data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=1)
                )
            )
            
            # Add price anomalies if detected
            if "Price" in anomaly_types:
                # Get indices of price anomalies
                price_anomaly_indices = np.where(anomaly_data['price_anomaly'])[0]
                
                if len(price_anomaly_indices) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_data.iloc[price_anomaly_indices]['timestamp'],
                            y=anomaly_data.iloc[price_anomaly_indices]['close'],
                            mode='markers',
                            name='Price Anomalies',
                            marker=dict(
                                color='red',
                                size=10,
                                symbol='circle',
                                line=dict(color='black', width=1)
                            )
                        )
                    )
            
            # Add momentum anomalies if detected
            if "Momentum" in anomaly_types and 'positive_momentum' in anomaly_data.columns:
                # Get indices of positive momentum anomalies
                pos_momentum_indices = np.where(anomaly_data['positive_momentum'])[0]
                
                if len(pos_momentum_indices) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_data.iloc[pos_momentum_indices]['timestamp'],
                            y=anomaly_data.iloc[pos_momentum_indices]['close'],
                            mode='markers',
                            name='Positive Momentum',
                            marker=dict(
                                color='green',
                                size=10,
                                symbol='triangle-up',
                                line=dict(color='black', width=1)
                            )
                        )
                    )
                
                # Get indices of negative momentum anomalies
                neg_momentum_indices = np.where(anomaly_data['negative_momentum'])[0]
                
                if len(neg_momentum_indices) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_data.iloc[neg_momentum_indices]['timestamp'],
                            y=anomaly_data.iloc[neg_momentum_indices]['close'],
                            mode='markers',
                            name='Negative Momentum',
                            marker=dict(
                                color='red',
                                size=10,
                                symbol='triangle-down',
                                line=dict(color='black', width=1)
                            )
                        )
                    )
            
            # Update layout
            fig.update_layout(
                title=f"{anomaly_symbol} 价格异常",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume anomalies chart
            if "Volume" in anomaly_types:
                vol_fig = go.Figure()
                
                # Add volume bars
                vol_fig.add_trace(
                    go.Bar(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data['volume'],
                        name='Volume',
                        marker=dict(color='rgba(0, 128, 0, 0.5)')
                    )
                )
                
                # Add volume anomalies
                volume_anomaly_indices = np.where(anomaly_data['volume_anomaly'])[0]
                
                if len(volume_anomaly_indices) > 0:
                    vol_fig.add_trace(
                        go.Bar(
                            x=anomaly_data.iloc[volume_anomaly_indices]['timestamp'],
                            y=anomaly_data.iloc[volume_anomaly_indices]['volume'],
                            name='Volume Anomalies',
                            marker=dict(color='rgba(255, 0, 0, 0.8)')
                        )
                    )
                
                # Update layout
                vol_fig.update_layout(
                    title=f"{anomaly_symbol} 交易量异常",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=300,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(vol_fig, use_container_width=True)
            
            # Add real-time Z-score analysis
            st.subheader("实时Z-score分析")
            
            # Get historical data for the last 30 days
            historical_data = db_manager.get_historical_prices(anomaly_symbol, 
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            if not historical_data.empty:
                # Calculate mean and std of closing prices
                mean_price = historical_data['close'].mean()
                std_price = historical_data['close'].std()
                
                # Get latest price
                latest_price = latest_prices[latest_prices['symbol'] == anomaly_symbol]['price'].iloc[0]
                
                # Calculate Z-score
                z_score = (latest_price - mean_price) / std_price if std_price != 0 else 0
                
                # Create warning message based on Z-score
                if abs(z_score) > 3:
                    st.error(f"⚠️ Extreme Z-score detected: {z_score:.2f}")
                    st.markdown(f"""
                    - Current Price: ${latest_price:.2f}
                    - 30-day Mean: ${mean_price:.2f}
                    - 30-day Std: ${std_price:.2f}
                    - Z-score: {z_score:.2f}
                    """)
                elif abs(z_score) > 2:
                    st.warning(f"⚠️ High Z-score detected: {z_score:.2f}")
                    st.markdown(f"""
                    - Current Price: ${latest_price:.2f}
                    - 30-day Mean: ${mean_price:.2f}
                    - 30-day Std: ${std_price:.2f}
                    - Z-score: {z_score:.2f}
                    """)
                else:
                    st.success(f"Normal Z-score: {z_score:.2f}")
                    st.markdown(f"""
                    - Current Price: ${latest_price:.2f}
                    - 30-day Mean: ${mean_price:.2f}
                    - 30-day Std: ${std_price:.2f}
                    - Z-score: {z_score:.2f}
                    """)
                
                # Add Z-score chart
                fig = go.Figure()
                
                # Add historical prices
                fig.add_trace(go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['close'],
                    mode='lines',
                    name='Historical Prices',
                    line=dict(color='blue', width=1)
                ))
                
                # Add mean line
                fig.add_trace(go.Scatter(
                    x=[historical_data['timestamp'].min(), historical_data['timestamp'].max()],
                    y=[mean_price, mean_price],
                    mode='lines',
                    name='30-day Mean',
                    line=dict(color='green', width=1, dash='dash')
                ))
                
                # Add latest price point
                fig.add_trace(go.Scatter(
                    x=[datetime.now()],
                    y=[latest_price],
                    mode='markers',
                    name='Latest Price',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='star'
                    )
                ))
                
                # # Update layout
                # fig.update_layout(
                #     title=f"{anomaly_symbol} Price with Z-score Analysis",
                #     xaxis_title="Date",
                #     yaxis_title="Price ($)",
                #     height=400,
                #     legend=dict(
                #         orientation="h",
                #         yanchor="bottom",
                #         y=1.02,
                #         xanchor="right",
                #         x=1
                #     )
                # )
                
                # st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient historical data for Z-score calculation")
    else:
        st.info("Enable 'Detect Anomalies' in the sidebar to use this feature.")
# Tab 5: Price Prediction
with tabs[4]:
    if show_predictions:
        st.subheader("价格预测模型")
        
        # Select symbol for price prediction
        prediction_symbol = st.selectbox(
            "Select a stock for price prediction",
            options=selected_symbols,
            key="prediction_symbol"
        )
        
        # Load data for selected symbol
        prediction_data = data_by_symbol[prediction_symbol]
        
        if prediction_data.empty:
            st.warning(f"No data available for {prediction_symbol}")
        else:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Parameters for prediction
            with col1:
                st.subheader("模型参数")
                
                pred_horizon = st.slider(
                    "Prediction Horizon (days)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1,
                    help="Number of days to predict into the future"
                )
                
                pred_features = st.multiselect(
                    "Features to Use",
                    options=["Price", "Volume", "Technical Indicators"],
                    default=["Price", "Technical Indicators"]
                )
                
                test_size = st.slider(
                    "Test Set Size (%)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Percentage of data to use for testing"
                ) / 100
            
            with col2:
                st.subheader("运行预测")
                
                run_prediction = st.button("Run Price Prediction Model")
                
                if run_prediction:
                    with st.spinner("Training prediction models..."):
                        # Initialize predictor
                        predictor = MarketPredictor()
                        
                        # Prepare data
                        # Get close prices
                        prices = prediction_data['close']
                        
                        # Create additional features if selected
                        additional_features = None
                        
                        if "Volume" in pred_features or "Technical Indicators" in pred_features:
                            additional_features = pd.DataFrame(index=prediction_data.index)
                            
                            if "Volume" in pred_features:
                                # Add volume features
                                additional_features['volume'] = prediction_data['volume']
                                additional_features['volume_change'] = prediction_data['volume'].pct_change()
                                additional_features['volume_ma5'] = prediction_data['volume'].rolling(window=5).mean()
                            
                            if "Technical Indicators" in pred_features:
                                # Add technical indicators
                                # Calculate RSI
                                additional_features['rsi'] = FinancialMetrics.calculate_rsi(prediction_data['close'].values)
                                
                                # Calculate SMAs
                                additional_features['sma_5'] = prediction_data['close'].rolling(window=5).mean()
                                additional_features['sma_20'] = prediction_data['close'].rolling(window=20).mean()
                                
                                # Calculate volatility
                                returns = prediction_data['close'].pct_change().values
                                additional_features['volatility'] = FinancialMetrics.calculate_volatility(returns, window=20, annualize=False)
                        
                        # Run prediction model
                        results = predictor.predict_price_movement(
                            prices,
                            additional_features=additional_features,
                            n_lags=pred_horizon,
                            test_size=test_size
                        )
                        
                        if 'error' in results:
                            st.error(f"Error in prediction: {results['error']}")
                        else:
                            st.success("Prediction model trained successfully!")
                            
                            # Show best model
                            best_model = results['best_model']
                            st.info(f"Best model: {best_model}")
                            
                            # Show model performance
                            model_results = results['results'][best_model]
                            
                            metrics = {
                                "Train RMSE": f"${model_results['train_rmse']:.3f}",
                                "Test RMSE": f"${model_results['test_rmse']:.3f}",
                                "Train MAE": f"${model_results['train_mae']:.3f}",
                                "Test MAE": f"${model_results['test_mae']:.3f}",
                                "Train R²": f"{model_results['train_r2']:.3f}",
                                "Test R²": f"{model_results['test_r2']:.3f}"
                            }
                            
                            # Create metrics display
                            metric_cols = st.columns(3)
                            
                            for i, (label, value) in enumerate(metrics.items()):
                                col_idx = i % 3
                                metric_cols[col_idx].metric(label=label, value=value)
                            
                            # Show future prediction
                            if results['future_prediction'] is not None:
                                future_price = results['future_prediction']
                                latest_price = prediction_data['close'].iloc[-1]
                                price_change = future_price - latest_price
                                pct_change = price_change / latest_price * 100
                                
                                st.subheader("Price Prediction")
                                st.metric(
                                    label=f"Predicted Price in {pred_horizon} days",
                                    value=f"${future_price:.2f}",
                                    delta=f"{pct_change:.2f}%"
                                )
                            
                            # Show prediction chart
                            st.subheader("Model Performance")
                            
                            # Create plot of actual vs predicted values
                            fig = go.Figure()
                            
                            # Get actual and predicted values
                            test_actual = model_results['test_actual']
                            test_pred = model_results['test_predictions']
                            
                            # Get the dates for the test set (last x% of the data)
                            test_size_points = int(len(prediction_data) * test_size)
                            test_dates = prediction_data['timestamp'].iloc[-test_size_points:].reset_index(drop=True)
                            
                            # Add actual prices
                            fig.add_trace(
                                go.Scatter(
                                    x=test_dates,
                                    y=test_actual,
                                    mode='lines',
                                    name='Actual',
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            # Add predicted prices
                            fig.add_trace(
                                go.Scatter(
                                    x=test_dates,
                                    y=test_pred,
                                    mode='lines',
                                    name='Predicted',
                                    line=dict(color='red', width=2, dash='dash')
                                )
                            )
                            
                            # Add future prediction point if available
                            if results['future_prediction'] is not None:
                                future_date = prediction_data['timestamp'].iloc[-1] + pd.Timedelta(days=pred_horizon)
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=[future_date],
                                        y=[future_price],
                                        mode='markers',
                                        name='Future Prediction',
                                        marker=dict(
                                            color='green',
                                            size=10,
                                            symbol='star',
                                            line=dict(color='black', width=1)
                                        )
                                    )
                                )
                            
                            # Update layout
                            fig.update_layout(
                                title=f"{prediction_symbol} Price Prediction",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                height=400,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Click 'Run Price Prediction Model' to train and evaluate prediction models.")
    else:
        st.info("Enable 'Price Predictions' in the sidebar to use this feature.")
# Tab 6: Data Explorer
with tabs[5]:
    st.subheader("Financial Data Explorer")
    
    # Select data type
    data_type = st.radio(
        "Select Data Type",
        options=["Price Data", "OHLCV Data", "Statistics"],
        horizontal=True
    )
    
    # Select symbol
    explorer_symbol = st.selectbox(
        "Select Symbol",
        options=selected_symbols,
        key="explorer_symbol"
    )
    
    # Load data for selected symbol
    explorer_data = data_by_symbol[explorer_symbol]
    
    if explorer_data.empty:
        st.warning(f"No data available for {explorer_symbol}")
    else:
        if data_type == "Price Data":
            # Get price data from database
            price_data = db_manager.get_latest_prices([explorer_symbol])
            
            st.subheader(f"Latest Price Data for {explorer_symbol}")
            st.dataframe(price_data)
            
            # Download button
            csv = price_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Price Data as CSV",
                data=csv,
                file_name=f"{explorer_symbol}_prices.csv",
                mime="text/csv"
            )
        
        elif data_type == "OHLCV Data":
            st.subheader(f"OHLCV Data for {explorer_symbol}")
            
            # Date filter
            col1, col2 = st.columns(2)
            
            with col1:
                start_filter = st.date_input(
                    "Start Date",
                    value=explorer_data['timestamp'].min().date() if not explorer_data.empty else None
                )
            
            with col2:
                end_filter = st.date_input(
                    "End Date",
                    value=explorer_data['timestamp'].max().date() if not explorer_data.empty else None
                )
            
            # Filter data
            filtered_data = explorer_data[
                (explorer_data['timestamp'].dt.date >= start_filter) &
                (explorer_data['timestamp'].dt.date <= end_filter)
            ]
            
            st.dataframe(filtered_data.sort_values('timestamp', ascending=False))
            
            # Download button
            csv = filtered_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download OHLCV Data as CSV",
                data=csv,
                file_name=f"{explorer_symbol}_ohlcv.csv",
                mime="text/csv"
            )
        
        elif data_type == "Statistics":
            st.subheader(f"Statistical Analysis for {explorer_symbol}")
            
            # Calculate returns
            explorer_data['daily_return'] = explorer_data['close'].pct_change() * 100
            
            # Get monthly returns
            explorer_data['year_month'] = explorer_data['timestamp'].dt.to_period('M')
            monthly_returns = explorer_data.groupby('year_month')['daily_return'].agg(['mean', 'std', 'min', 'max']).reset_index()
            monthly_returns['year_month'] = monthly_returns['year_month'].astype(str)
            
            # Get yearly returns
            explorer_data['year'] = explorer_data['timestamp'].dt.year
            yearly_returns = explorer_data.groupby('year')['daily_return'].agg(['mean', 'std', 'min', 'max']).reset_index()
            

            # Create tabs for different statistics
            stat_tabs = st.tabs(["Summary", "Monthly", "Yearly", "Distribution"])
            
            with stat_tabs[0]:
                st.subheader("Summary Statistics")
                
                # Calculate statistics
                summary = explorer_data['daily_return'].describe()
                
                # Calculate additional metrics
                total_trading_days = len(explorer_data)
                positive_days = (explorer_data['daily_return'] > 0).sum()
                negative_days = (explorer_data['daily_return'] < 0).sum()
                positive_pct = positive_days / total_trading_days * 100
                
                # Calculate annualized return and volatility
                ann_return = explorer_data['daily_return'].mean() * 252
                ann_volatility = explorer_data['daily_return'].std() * np.sqrt(252)
                sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="Annualized Return", value=f"{ann_return:.2f}%")
                    st.metric(label="Daily Avg Return", value=f"{summary['mean']:.2f}%")
                    st.metric(label="Total Trading Days", value=f"{total_trading_days}")
                
                with col2:
                    st.metric(label="Annualized Volatility", value=f"{ann_volatility:.2f}%")
                    st.metric(label="Daily Volatility", value=f"{summary['std']:.2f}%")
                    st.metric(label="Positive Days", value=f"{positive_days} ({positive_pct:.1f}%)")
                
                with col3:
                    st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")
                    st.metric(label="Max Daily Gain", value=f"{summary['max']:.2f}%")
                    st.metric(label="Max Daily Loss", value=f"{summary['min']:.2f}%")
            
            with stat_tabs[1]:
                st.subheader("Monthly Returns")
                
                # Display monthly returns
                st.dataframe(monthly_returns)
                
                # Create heatmap of monthly returns
                monthly_returns['year'] = monthly_returns['year_month'].str.split('-').str[0]
                monthly_returns['month'] = monthly_returns['year_month'].str.split('-').str[1]
                
                # Create pivot table
                pivot_df = monthly_returns.pivot(index='year', columns='month', values='mean')
                
                # Create heatmap
                fig = px.imshow(
                    pivot_df,
                    text_auto=".2f",
                    color_continuous_scale=px.colors.diverging.RdBu_r,
                    color_continuous_midpoint=0,
                    labels=dict(x="Month", y="Year", color="Return (%)")
                )
                
                fig.update_layout(
                    title="Monthly Returns Heatmap (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with stat_tabs[2]:
                st.subheader("Yearly Returns")
                
                # Display yearly returns
                st.dataframe(yearly_returns)
                
                # Create bar chart of yearly returns
                fig = px.bar(
                    yearly_returns,
                    x='year',
                    y='mean',
                    error_y='std',
                    labels=dict(x="Year", y="Return (%)", mean="Average Return"),
                    title=f"{explorer_symbol} Yearly Returns",
                    color='mean',
                    color_continuous_scale=px.colors.diverging.RdBu_r,
                    color_continuous_midpoint=0
                )
                
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with stat_tabs[3]:
                st.subheader("Return Distribution")
                
                # Create histogram of returns
                fig = px.histogram(
                    explorer_data,
                    x='daily_return',
                    nbins=50,
                    labels=dict(x="Daily Return (%)", y="Frequency"),
                    title=f"{explorer_symbol} Return Distribution"
                )
                
                # Add a vertical line at the mean
                fig.add_vline(
                    x=explorer_data['daily_return'].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean",
                    annotation_position="top"
                )
                
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Clean returns data
                returns = explorer_data['daily_return'].dropna()

                # Check if we have sufficient data
                if len(returns) > 3 and returns.var() > 0:
                    # Calculate and display skewness and kurtosis
                    skew = returns.skew()
                    kurt = returns.kurtosis()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Skewness", value=f"{skew:.4f}")
                        skew_interpretation = (
                            "Positive (right-skewed)" if skew > 0 else
                            "Negative (left-skewed)" if skew < 0 else
                            "Zero (symmetric)"
                        )
                        st.markdown(f"**Interpretation:** {skew_interpretation}")
                    
                    with col2:
                        st.metric(label="Kurtosis", value=f"{kurt:.4f}")
                        kurt_interpretation = (
                            "Leptokurtic (heavy tails)" if kurt > 0 else
                            "Platykurtic (thin tails)" if kurt < 0 else
                            "Mesokurtic (normal)"
                        )
                        st.markdown(f"**Interpretation:** {kurt_interpretation}")
                else:
                    st.warning("Insufficient data for calculating skewness and kurtosis. Need at least 4 valid data points with variation.")
                # QQ plot
                import scipy.stats as stats
                
                # Calculate QQ plot data
                returns = explorer_data['daily_return'].dropna()
                qq = stats.probplot(returns, dist="norm")
                
                # Extract data
                x = np.array([point[0] for point in qq[0]])
                y = np.array([point[1] for point in qq[0]])
                
                # Create QQ plot
                fig = go.Figure()
                
                # Add scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        name='Returns',
                        marker=dict(color='blue')
                    )
                )
                
                # Add the line representing normal distribution
                slope = qq[1][0]
                intercept = qq[1][1]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=slope * x + intercept,
                        mode='lines',
                        name='Normal',
                        line=dict(color='red')
                    )
                )
                
                fig.update_layout(
                    title="Normal Q-Q Plot",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Normality test
                st.subheader("Normality Test (Shapiro-Wilk)")

                # Check if we have sufficient data (Shapiro-Wilk requires at least 3 data points)
                if len(returns) >= 3 and returns.var() > 0:
                    try:
                        stat, p_value = stats.shapiro(returns)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(label="Test Statistic", value=f"{stat:.4f}")
                        
                        with col2:
                            st.metric(label="p-value", value=f"{p_value:.8f}")
                        
                        # Interpret the result
                        if p_value < 0.05:
                            st.markdown("**Conclusion:** Reject the null hypothesis. Returns are **not normally distributed**.")
                        else:
                            st.markdown("**Conclusion:** Fail to reject the null hypothesis. Returns may follow a normal distribution.")
                    except Exception as e:
                        st.warning(f"Could not perform Shapiro-Wilk test: {str(e)}")
                else:
                    st.warning("Insufficient data for Shapiro-Wilk test. Need at least 3 valid data points with variation.")
# Add Chat Assistant Tab
# with tabs[6]:
#     st.subheader("市场分析助手")
    
#     # Initialize chat assistant
#     chat_assistant = MarketChatAssistant()
    
#     # Create a container for chat messages
#     chat_container = st.container()
    
#     # Initialize chat history in session state if it doesn't exist
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
    
#     # Display chat history
#     with chat_container:
#         for message in st.session_state.chat_history:
#             if message["role"] == "user":
#                 st.write(f"👤 You: {message['content']}")
#             else:
#                 st.write(f"🤖 Assistant: {message['content']}")
    
#     # Chat input
#     user_input = st.text_input("Ask about market analysis:", key="chat_input")
    
#     if user_input:
#         # Add user message to chat history
#         st.session_state.chat_history.append({"role": "user", "content": user_input})
        
#         # Get assistant's response
#         with st.spinner("Thinking..."):
#             response = asyncio.run(chat_assistant.process_message(user_input))
            
#             # Add assistant's response to chat history
#             st.session_state.chat_history.append({"role": "assistant", "content": response})
        
#         # Clear the input box
#         st.experimental_rerun()
    
#     # Add a clear chat button
#     if st.button("Clear Chat"):
#         st.session_state.chat_history = []
#         st.experimental_rerun()