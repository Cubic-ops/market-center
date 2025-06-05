"""
Real-time database manager for financial data using yfinance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import pytz
import sqlite3
import os

logger = logging.getLogger(__name__)

class DatabaseManagerRealtime:
    """Manage real-time financial data operations using yfinance."""
    
    def __init__(self):
        """Initialize the real-time database manager."""
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.db_path = 'market_data.db'
        # self._reset_database()
        self._init_portfolio_table()
    
    def _reset_database(self):
        """Reset the database to ensure correct schema."""
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                logger.info("Removed old database file")
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
    
    def _init_portfolio_table(self):
        """Initialize the portfolio table in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create portfolio table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                shares INTEGER NOT NULL,
                last_updated TEXT NOT NULL,
                UNIQUE(symbol)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Portfolio table initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing portfolio table: {e}")
    
    def update_portfolio(self, symbol: str, shares: int) -> bool:
        """
        Update portfolio position for a symbol.
        
        Args:
            symbol: Stock symbol
            shares: Number of shares
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update or insert portfolio position
            cursor.execute('''
            INSERT OR REPLACE INTO portfolio (symbol, shares, last_updated)
            VALUES (?, ?, ?)
            ''', (
                symbol,
                shares,
                datetime.now(self.eastern_tz).isoformat()
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Updated portfolio position for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating portfolio position for {symbol}: {e}")
            return False
    
    def get_portfolio(self) -> pd.DataFrame:
        """
        Get current portfolio positions.
        
        Returns:
            DataFrame with portfolio positions containing:
            - symbol: Stock symbol
            - shares: Number of shares
            - current_price: Current market price
            - market_value: Current market value
            - last_updated: Last update timestamp
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get portfolio positions
            portfolio_df = pd.read_sql_query('''
            SELECT * FROM portfolio
            ''', conn)
            
            if portfolio_df.empty:
                return pd.DataFrame(columns=[
                    'symbol', 'shares', 'current_price',
                    'market_value', 'last_updated'
                ])
            
            # Get current prices for all symbols
            symbols = portfolio_df['symbol'].tolist()
            current_prices = self.get_latest_prices(symbols)
            
            # Merge portfolio with current prices
            portfolio_df = portfolio_df.merge(
                current_prices[['symbol', 'price']],
                on='symbol',
                how='left'
            )
            
            # Calculate portfolio metrics
            portfolio_df['current_price'] = portfolio_df['price']
            portfolio_df['market_value'] = portfolio_df['shares'] * portfolio_df['current_price']
            
            # Convert timestamp to datetime
            portfolio_df['last_updated'] = pd.to_datetime(portfolio_df['last_updated'])
            
            # Drop temporary columns
            portfolio_df = portfolio_df.drop('price', axis=1)
            
            conn.close()
            return portfolio_df
            
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return pd.DataFrame(columns=[
                'symbol', 'shares', 'current_price',
                'market_value', 'last_updated'
            ])
    
    def get_portfolio_summary(self) -> Dict[str, float]:
        """
        Get portfolio summary statistics.
        
        Returns:
            Dictionary containing:
            - total_value: Total portfolio value
            - value_change: Change in portfolio value (positive for gain, negative for loss)
        """
        try:
            portfolio_df = self.get_portfolio()
            logger.info(f"the portfolio is{portfolio_df}")
            
            if portfolio_df.empty:
                return {
                    'total_value': 0.0,
                    'value_change': 0.0
                }
            
            # Calculate total value
            total_value = portfolio_df['market_value'].sum()
            
            # Get latest prices with change percentages
            symbols = portfolio_df['symbol'].tolist()
            latest_prices = self.get_latest_prices(symbols)
            
            # Calculate value change based on price changes
            value_change = 0.0
            for _, row in portfolio_df.iterrows():
                # Get the change percentage from latest prices
                symbol_data = latest_prices[latest_prices['symbol'] == row['symbol']]
                if not symbol_data.empty:
                    change_percent = symbol_data.iloc[0]['change_percent']
                    # Calculate the value change for this position
                    position_value_change = row['market_value'] * (change_percent / 100)
                    value_change += position_value_change
            
            summary = {
                'total_value': total_value,
                'value_change': value_change
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {
                'total_value': 0.0,
                'value_change': 0.0
            }
    
    def get_latest_prices(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get latest stock prices using yfinance.
        
        Args:
            symbols: Optional list of symbols to filter
            
        Returns:
            DataFrame with latest prices containing:
            - symbol: Stock symbol
            - price: Current price
            - change: Price change
            - change_percent: Percentage change
            - volume: Trading volume
            - timestamp: Timestamp of the price (Eastern Time)
        """
        try:
            if not symbols:
                # If no symbols provided, return empty DataFrame
                return pd.DataFrame(columns=['symbol', 'price', 'change', 'change_percent', 'volume', 'timestamp'])
            
            # Get data for all symbols at once
            try:
                # Download data for all symbols
                data = yf.download(
                    symbols,
                    period="1d",
                    interval="1m",
                    group_by='ticker',
                    auto_adjust=True
                )
                
                results = []
                current_time = datetime.now(self.eastern_tz)
                
                # Process each symbol's data
                for symbol in symbols:
                    try:
                        if isinstance(data, pd.DataFrame):
                            # Single symbol case
                            if len(symbols) == 1:
                                df = data
                            else:
                                # Multiple symbols case
                                df = data[symbol]
                            
                            if not df.empty:
                                # Get the latest data point
                                latest = df.iloc[-1]
                                
                                # Calculate changes
                                current_price = latest['Close']
                                previous_close = df['Close'].iloc[0]
                                change = current_price - previous_close
                                change_percent = (change / previous_close * 100) if previous_close else 0
                                
                                # Get ticker info for volume
                                ticker = yf.Ticker(symbol)
                                info = ticker.info
                                
                                price_data = {
                                    'symbol': symbol,
                                    'price': current_price,
                                    'change': change,
                                    'change_percent': change_percent,
                                    'volume': info.get('regularMarketVolume', 0),
                                    'timestamp': current_time
                                }
                                results.append(price_data)
                    except Exception as e:
                        logger.error(f"Error processing data for {symbol}: {e}")
                        continue
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                return df
                
            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                # Fallback to individual symbol processing if batch download fails
                results = []
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        current_price = info.get('regularMarketPrice', 0)
                        previous_close = info.get('previousClose', 0)
                        
                        change = current_price - previous_close
                        change_percent = (change / previous_close * 100) if previous_close else 0
                        
                        price_data = {
                            'symbol': symbol,
                            'price': current_price,
                            'change': change,
                            'change_percent': change_percent,
                            'volume': info.get('regularMarketVolume', 0),
                            'timestamp': current_time
                        }
                        results.append(price_data)
                    except Exception as e:
                        logger.error(f"Error getting data for {symbol}: {e}")
                        continue
                
                return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error in get_latest_prices: {e}")
            return pd.DataFrame(columns=['symbol', 'price', 'change', 'change_percent', 'volume', 'timestamp'])
    
    def get_historical_stock_prices(self, symbol: str, start_date: Optional[Union[str, datetime]] = None,
                                  end_date: Optional[Union[str, datetime]] = None,
                                  interval: str = '1d') -> pd.DataFrame:
        """
        Get historical stock prices using yfinance.
        
        Args:
            symbol: Stock symbol
            start_date: Optional start date
            end_date: Optional end date
            interval: Time interval for data points. Options:
                - '1m': 1 minute (only available for last 7 days)
                - '2m': 2 minutes (only available for last 7 days)
                - '5m': 5 minutes (only available for last 7 days)
                - '15m': 15 minutes (only available for last 7 days)
                - '30m': 30 minutes (only available for last 7 days)
                - '60m': 60 minutes (only available for last 7 days)
                - '90m': 90 minutes (only available for last 7 days)
                - '1h': 1 hour (only available for last 730 days)
                - '1d': 1 day (default)
                - '5d': 5 days
                - '1wk': 1 week
                - '1mo': 1 month
                - '3mo': 3 months
            
        Returns:
            DataFrame with historical stock prices containing:
            - symbol: Stock symbol
            - price: Current price
            - change: Price change
            - change_percent: Percentage change
            - volume: Trading volume
            - timestamp: Timestamp of the price (Eastern Time)
        """
        try:
            # Set default dates if not provided
            if not end_date:
                # end_date = datetime.now(self.eastern_tz)
                end_date = datetime.now()
 
            if not start_date:
                # Adjust default history period based on interval
                if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
                    start_date = end_date - timedelta(days=1)  # 7 days for minute data
                elif interval == '1h':
                    start_date = end_date - timedelta(days=7)  # 60 days for hourly data
                else:
                    start_date = end_date - timedelta(days=30)  # 30 days for daily and larger intervals
            
            # Convert dates to string format if they are datetime objects
            # if isinstance(start_date, datetime):
            #     start_date = start_date.strftime('%Y-%m-%d')
            # if isinstance(end_date, datetime):
            #     end_date = end_date.strftime('%Y-%m-%d')
            logger.info(f"now start_date is {start_date} and end_date is {end_date}!!!!!!!!!")
            # Get ticker data
            ticker = yf.Ticker(symbol)
            
            # Get historical data with specified interval
            hist = ticker.history(start=start_date, end=end_date, interval=interval)
   
            if hist.empty:
                return pd.DataFrame(columns=['symbol', 'price', 'change', 'change_percent', 'volume', 'timestamp'])
            
            # Calculate changes
            hist['change'] = hist['Close'].diff()
            hist['change_percent'] = hist['Close'].pct_change() * 100
            
            # Convert index to Eastern Time
            # hist.index = hist.index.tz_localize('UTC').tz_convert('US/Eastern')
            
            # Create DataFrame with required columns
            df = pd.DataFrame({
                'symbol': symbol,
                'price': hist['Close'],
                'change': hist['change'],
                'change_percent': hist['change_percent'],
                'volume': hist['Volume'],
                'timestamp': hist.index
            })
            
            # Reset index to make timestamp a column
            df = df.reset_index(drop=True)
            
            # Fill NaN values
            df['change'] = df['change'].fillna(0)
            df['change_percent'] = df['change_percent'].fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in get_historical_stock_prices for {symbol}: {e}")
            return pd.DataFrame(columns=['symbol', 'price', 'change', 'change_percent', 'volume', 'timestamp'])
    
    def get_realtime_prices(self, symbols: List[str], refresh_interval: int = 2) -> pd.DataFrame:
        """
        Get real-time prices from the database with specified refresh interval.
        
        Args:
            symbols: List of stock symbols
            refresh_interval: Refresh interval in seconds (default: 2)
            
        Returns:
            DataFrame with latest prices containing:
            - symbol: Stock symbol
            - price: Current price
            - change: Price change
            - change_percent: Percentage change
            - volume: Trading volume
            - timestamp: Timestamp of the price (Eastern Time)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query to get latest prices for each symbol
            placeholders = ', '.join(['?'] * len(symbols))
            query = f"""
            SELECT p.*
            FROM stock_prices p
            INNER JOIN (
                SELECT symbol, MAX(timestamp) as max_timestamp
                FROM stock_prices
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
            ) m ON p.symbol = m.symbol AND p.timestamp = m.max_timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=symbols)
            conn.close()
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error in get_realtime_prices: {e}")
            return pd.DataFrame(columns=['symbol', 'price', 'change', 'change_percent', 'volume', 'timestamp']) 