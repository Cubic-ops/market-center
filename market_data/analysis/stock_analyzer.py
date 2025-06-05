"""
Real-time stock analysis and streaming calculations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
import pytz
import sqlite3
from ..storage.database_manager import DatabaseManager
import time

logger = logging.getLogger(__name__)

class StockAnalyzer:
    """Analyzer for real-time stock data streaming and calculations."""
    
    def __init__(self, db_path: str = "finance.db"):
        """
        Initialize the stock analyzer.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_manager = DatabaseManager(db_path)
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.market_open_time = datetime.now(self.eastern_tz).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
    
    def is_market_open(self) -> bool:
        """
        Check if the US stock market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        now = datetime.now(self.eastern_tz)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            return False
        
        # Check if current time is between 9:30 AM and 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_last_trading_day(self) -> datetime:
        """
        Get the last trading day.
        
        Returns:
            datetime: Last trading day
        """
        now = datetime.now(self.eastern_tz)
        
        # If it's weekend, go back to Friday
        if now.weekday() >= 5:
            days_to_subtract = now.weekday() - 4
            return now - timedelta(days=days_to_subtract)
        
        # If it's before market open, go back to previous day
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now < market_open:
            return now - timedelta(days=1)
        
        return now
    
    def get_closing_prices(self) -> pd.DataFrame:
        """
        Get closing prices for the last trading day.
        
        Returns:
            DataFrame with closing prices
        """
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            
            # Get the last trading day
            # last_trading_day = self.get_last_trading_day()
            # last_trading_date = last_trading_day.strftime('%Y-%m-%d')
            trading_date = datetime.now(self.eastern_tz).strftime('%Y-%m-%d')
            
            # Get closing prices for the last trading day
            query = """
            SELECT symbol, price, change_percent, timestamp
            FROM stock_prices
            WHERE date(timestamp) = ?
            AND time(timestamp) >= '16:00:00'
            ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(trading_date,))
            conn.close()
            
            if df.empty:
                logger.warning(f"No closing prices found for {trading_date}")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Get the latest price for each symbol (closing price)
            closing_prices = df.groupby('symbol').first().reset_index()
            
            return closing_prices
            
        except Exception as e:
            logger.error(f"Error getting closing prices: {e}")
            return pd.DataFrame()
    
    def get_today_prices(self) -> pd.DataFrame:
        """
        Get all stock prices from today's market open.
        If market is closed, returns closing prices from last trading day.
        
        Returns:
            DataFrame with price data
        """
        # if not self.is_market_open():
        #     return self.get_closing_prices()
            
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            
            # Get today's prices
            query = """
            SELECT symbol, price, change_percent, timestamp
            FROM stock_prices
            WHERE date(timestamp) = date('now', 'localtime')
            ORDER BY timestamp DESC
            """
            
            # Read the data with proper timestamp parsing
            df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
            
            # Ensure timestamp is in the correct timezone
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
            
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # Get the latest price for each symbol
            latest_prices = df.groupby('symbol').first().reset_index()
            
            return latest_prices
            
        except Exception as e:
            logger.error(f"Error getting today's prices: {e}")
            return pd.DataFrame()
    
    def get_top_movers(self, n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get top gainers and losers.
        If market is closed, uses closing prices from last trading day.
        
        Args:
            n: Number of stocks to return for each category
            
        Returns:
            Tuple of (top_gainers, top_losers) DataFrames
        """
        try:
            # Get prices (either today's or last trading day's closing prices)
            prices_df = self.get_today_prices()
            
            if prices_df.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Sort by change_percent
            sorted_df = prices_df.sort_values('change_percent', ascending=False)
            
            # Get top gainers and losers
            top_gainers = sorted_df.head(n)
            top_losers = sorted_df.tail(n)
            
            # Add rank column
            top_gainers['rank'] = range(1, n + 1)
            top_losers['rank'] = range(1, n + 1)
            
            return top_gainers, top_losers
            
        except Exception as e:
            logger.error(f"Error getting top movers: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_current_top_movers(self) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
        """
        Get current top movers data.
        Returns top gainers and losers for current market session or last trading day.
        
        Returns:
            Tuple of (top_gainers, top_losers, is_market_open)
        """
        try:
            is_market_open = self.is_market_open()
            top_gainers, top_losers = self.get_top_movers()
            
            if not top_gainers.empty and not top_losers.empty:
                logger.info(f"Retrieved top movers data (Market {'Open' if is_market_open else 'Closed'})")
            
            return top_gainers, top_losers, is_market_open
            
        except Exception as e:
            logger.error(f"Error getting current top movers: {e}")
            return pd.DataFrame(), pd.DataFrame(), False

    def stream_top_movers(self, interval: int = 60):
        """
        Stream top movers data continuously.
        Shows closing prices when market is closed.
        
        Args:
            interval: Update interval in seconds
        """
        while True:
            try:
                top_gainers, top_losers, is_market_open = self.get_current_top_movers()
                
                if not top_gainers.empty and not top_losers.empty:
                    # Log top gainers
                    logger.info("\n=== Top Gainers (Current Session) ===")
                    for _, row in top_gainers.iterrows():
                        logger.info(f"{row['symbol']}: {row['change_percent']:.2f}% (${row['price']:.2f})")
                    
                    # Log top losers
                    logger.info("\n=== Top Losers (Current Session) ===")
                    for _, row in top_losers.iterrows():
                        logger.info(f"{row['symbol']}: {row['change_percent']:.2f}% (${row['price']:.2f})")
                    
                    logger.info(f"\nLast updated: {datetime.now(self.eastern_tz).strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"Market status: {'Open' if is_market_open else 'Closed'}")
                
                # Wait for next update
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in stream_top_movers: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    # def get_market_summary(self) -> Dict[str, Any]:
    #     """
    #     Get market summary statistics.
    #     Uses closing prices when market is closed.
        
    #     Returns:
    #         Dictionary containing market summary data
    #     """
    #     try:
    #         prices_df = self.get_today_prices()
            
    #         if prices_df.empty:
    #             return {
    #                 'total_stocks': 0,
    #                 'advancing': 0,
    #                 'declining': 0,
    #                 'unchanged': 0,
    #                 'avg_change': 0.0,
    #                 'market_status': 'Closed' if not self.is_market_open() else 'Open'
    #             }
            
    #         # Calculate statistics
    #         total_stocks = len(prices_df)
    #         advancing = len(prices_df[prices_df['change_percent'] > 0])
    #         declining = len(prices_df[prices_df['change_percent'] < 0])
    #         unchanged = len(prices_df[prices_df['change_percent'] == 0])
    #         avg_change = prices_df['change_percent'].mean()
            
    #         return {
    #             'total_stocks': total_stocks,
    #             'advancing': advancing,
    #             'declining': declining,
    #             'unchanged': unchanged,
    #             'avg_change': avg_change,
    #             'market_status': 'Closed' if not self.is_market_open() else 'Open'
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error getting market summary: {e}")
    #         return {
    #             'total_stocks': 0,
    #             'advancing': 0,
    #             'declining': 0,
    #             'unchanged': 0,
    #             'avg_change': 0.0,
    #             'market_status': 'Error'
    #         }

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the analyzer
    analyzer = StockAnalyzer()
    analyzer.stream_top_movers() 