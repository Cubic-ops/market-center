"""
Configuration settings for the financial data platform.
"""

from .stock_list import get_stocks_from_database

# US Market data settings
US_SYMBOLS = [
    "AAPL",
    "AMZN",
    "BAC",
    "CERO",
    "CIB",
    "GOOGL",
    "JPM",
    "META",
    "MSFT",
    "NVDA",
    "PBR",
    "TSLA",
    "WMT"
]

# A-share Market data settings
A_SHARE_SYMBOLS = [
    "600941.SH",
    "600519.SH",
    "601398.SH",
    "601857.SH",
    "601939.SH",
    "601288.SH",
    "600938.SH",
    "601988.SH",
    "300750.SZ"  
]   
#不能超过10个否则报错level not enough

# Get all stocks from database
SYMBOLS_ALL = get_stocks_from_database()

# Data fetch intervals (in seconds)
PRICE_FETCH_INTERVAL = 60
OHLCV_FETCH_INTERVAL = 300  # 5 minutes

# Database settings
DATABASE_PATH = "finance.db"  # SQLite database path
