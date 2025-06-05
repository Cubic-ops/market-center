"""
Stock list management module for handling US stock symbols.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional
import sqlite3
from datetime import datetime
import os

def get_all_stocks() -> List[str]:
    """
    Fetch all US stock symbols using yfinance.
    Returns a list of stock symbols.
    """
    try:
        # Get all tickers from yfinance
        tickers = yf.Tickers('^GSPC ^DJI ^IXIC')  # Get major indices first
        all_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        symbols = all_tickers['Symbol'].tolist()
        
        # Add more symbols from other sources
        # You can expand this by adding more sources or using other APIs
        return symbols
    except Exception as e:
        print(f"Error fetching stock symbols: {e}")
        return []

def filter_stocks_by_criteria(
    min_market_cap: Optional[float] = None,
    min_volume: Optional[int] = None,
    sectors: Optional[List[str]] = None
) -> List[str]:
    """
    Filter stocks based on specific criteria.
    
    Args:
        min_market_cap: Minimum market capitalization in millions
        min_volume: Minimum average daily volume
        sectors: List of sectors to include
    
    Returns:
        List of filtered stock symbols
    """
    # Implementation will depend on your data source
    # This is a placeholder for the actual implementation
    return []

def update_stock_database():
    """
    Update the local database with current stock information.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'finance.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stocks (
        symbol TEXT PRIMARY KEY,
        name TEXT,
        sector TEXT,
        industry TEXT,
        last_updated TIMESTAMP
    )
    ''')
    
    # Get current stock list
    symbols = get_all_stocks()
    
    # Update database
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            cursor.execute('''
            INSERT OR REPLACE INTO stocks (symbol, name, sector, industry, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol,
                info.get('longName', ''),
                info.get('sector', ''),
                info.get('industry', ''),
                datetime.now()
            ))
        except Exception as e:
            print(f"Error updating {symbol}: {e}")
    
    conn.commit()
    conn.close()

def get_stocks_from_database(
    limit: int = 5000,
    offset: int = 0
) -> List[str]:
    """
    Get stocks from the local database with pagination.
    
    Args:
        limit: Maximum number of stocks to return
        offset: Number of stocks to skip
    
    Returns:
        List of stock symbols
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'finance.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT symbol FROM stocks
    ORDER BY symbol
    LIMIT ? OFFSET ?
    ''', (limit, offset))
    
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

if __name__ == "__main__":
    # Update the database when run directly
    update_stock_database() 