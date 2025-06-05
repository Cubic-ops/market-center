"""
Database manager for financial data storage and retrieval.
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manage database operations for financial data."""
    
    def __init__(self, db_path: str):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create database if it doesn't exist
        self._create_database()
    
    def _create_database(self) -> None:
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create stock prices table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            change REAL,
            change_percent REAL,
            volume INTEGER,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
        ''')
        
        # Create OHLCV table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER,
            period TEXT,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, period, timestamp)
        )
        ''')
        
        # Create technical indicators table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            indicator_type TEXT NOT NULL,
            parameter TEXT,
            value REAL NOT NULL,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, indicator_type, parameter, timestamp)
        )
        ''')
        
        # Create news articles table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT,
            source TEXT,
            url TEXT,
            published_date TEXT NOT NULL,
            sentiment_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(url)
        )
        ''')
        
        # Create news-symbol relation table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_id INTEGER,
            symbol TEXT NOT NULL,
            relevance_score REAL,
            FOREIGN KEY (news_id) REFERENCES news_articles(id),
            UNIQUE(news_id, symbol)
        )
        ''')
        
        # Create anomalies table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            anomaly_type TEXT NOT NULL,
            description TEXT,
            severity REAL,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, anomaly_type, timestamp)
        )
        ''')
        
        # Create model predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            model_name TEXT NOT NULL,
            target TEXT NOT NULL,
            prediction_value REAL NOT NULL,
            confidence REAL,
            prediction_date TEXT NOT NULL,
            target_date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, model_name, target, prediction_date, target_date)
        )
        ''')
        
        # Create level 2 market data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS level2_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            bid_price_1 REAL,
            bid_volume_1 INTEGER,
            bid_price_2 REAL,
            bid_volume_2 INTEGER,
            bid_price_3 REAL,
            bid_volume_3 INTEGER,
            bid_price_4 REAL,
            bid_volume_4 INTEGER,
            bid_price_5 REAL,
            bid_volume_5 INTEGER,
            ask_price_1 REAL,
            ask_volume_1 INTEGER,
            ask_price_2 REAL,
            ask_volume_2 INTEGER,
            ask_price_3 REAL,
            ask_volume_3 INTEGER,
            ask_price_4 REAL,
            ask_volume_4 INTEGER,
            ask_price_5 REAL,
            ask_volume_5 INTEGER,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
        ''')
        
        # Create index for level2_data
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_level2_data_symbol ON level2_data(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_level2_data_timestamp ON level2_data(timestamp)')
        
        # Create indices for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_prices_timestamp ON stock_prices(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol ON technical_indicators(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_symbols_symbol ON news_symbols(symbol)')
        
        conn.commit()
        conn.close()
        
        logger.info("Database tables created successfully")
    
    def insert_stock_prices(self, prices: List[Dict[str, Any]]) -> int:
        """
        Insert stock prices into the database.
        
        Args:
            prices: List of stock price dictionaries
            
        Returns:
            Number of inserted records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted_count = 0
        
        for price in prices:
            try:
                # Ensure timestamp is in ISO format
                if isinstance(price['timestamp'], datetime):
                    timestamp = price['timestamp'].isoformat()
                else:
                    timestamp = price['timestamp']
                
                cursor.execute('''
                INSERT OR REPLACE INTO stock_prices 
                (symbol, price, change, change_percent, volume, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    price['symbol'],
                    price['price'],
                    price.get('change', 0),
                    price.get('change_percent', 0),
                    price.get('volume', 0),
                    timestamp
                ))
                
                inserted_count += 1
                
            except Exception as e:
                logger.error(f"Error inserting stock price for {price.get('symbol')}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Inserted {inserted_count} stock price records")
        return inserted_count
    
    def insert_ohlcv_data(self, ohlcv_data: List[Dict[str, Any]]) -> int:
        """
        Insert OHLCV data into the database.
        
        Args:
            ohlcv_data: List of OHLCV dictionaries
            
        Returns:
            Number of inserted records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted_count = 0
        
        for ohlcv in ohlcv_data:
            try:
                # Ensure timestamp is in ISO format
                if isinstance(ohlcv['timestamp'], datetime):
                    timestamp = ohlcv['timestamp'].isoformat()
                else:
                    timestamp = ohlcv['timestamp']
                
                cursor.execute('''
                INSERT OR REPLACE INTO ohlcv 
                (symbol, open, high, low, close, volume, period, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ohlcv['symbol'],
                    ohlcv['open'],
                    ohlcv['high'],
                    ohlcv['low'],
                    ohlcv['close'],
                    ohlcv.get('volume', 0),
                    ohlcv.get('period', '1d'),
                    timestamp
                ))
                
                inserted_count += 1
                
            except Exception as e:
                logger.error(f"Error inserting OHLCV data for {ohlcv.get('symbol')}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Inserted {inserted_count} OHLCV records")
        return inserted_count
    
    def insert_technical_indicators(self, indicators: List[Dict[str, Any]]) -> int:
        """
        Insert technical indicators into the database.
        
        Args:
            indicators: List of technical indicator dictionaries
            
        Returns:
            Number of inserted records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted_count = 0
        
        for indicator in indicators:
            try:
                # Ensure timestamp is in ISO format
                if isinstance(indicator['timestamp'], datetime):
                    timestamp = indicator['timestamp'].isoformat()
                else:
                    timestamp = indicator['timestamp']
                
                cursor.execute('''
                INSERT OR REPLACE INTO technical_indicators 
                (symbol, indicator_type, parameter, value, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    indicator['symbol'],
                    indicator['indicator_type'],
                    indicator.get('parameter', ''),
                    indicator['value'],
                    timestamp
                ))
                
                inserted_count += 1
                
            except Exception as e:
                logger.error(f"Error inserting technical indicator for {indicator.get('symbol')}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Inserted {inserted_count} technical indicator records")
        return inserted_count
    
    def insert_news_article(self, article: Dict[str, Any], symbols: List[Dict[str, Any]] = None) -> Optional[int]:
        """
        Insert news article into the database.
        
        Args:
            article: News article dictionary
            symbols: List of dictionaries with symbol and relevance score
            
        Returns:
            ID of inserted article or None if error
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        
        try:
            # Ensure published_date is in ISO format
            if isinstance(article['published_date'], datetime):
                published_date = article['published_date'].isoformat()
            else:
                published_date = article['published_date']
            
            # Insert article
            cursor.execute('''
            INSERT OR REPLACE INTO news_articles 
            (title, content, source, url, published_date, sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                article['title'],
                article.get('content', ''),
                article.get('source', ''),
                article.get('url', ''),
                published_date,
                article.get('sentiment_score', None)
            ))
            
            # Get article ID
            article_id = cursor.lastrowid
            
            # Insert symbols if provided
            if symbols and article_id:
                for symbol_info in symbols:
                    cursor.execute('''
                    INSERT OR REPLACE INTO news_symbols 
                    (news_id, symbol, relevance_score)
                    VALUES (?, ?, ?)
                    ''', (
                        article_id,
                        symbol_info['symbol'],
                        symbol_info.get('relevance_score', 1.0)
                    ))
            
            conn.commit()
            
            logger.info(f"Inserted news article: {article['title']}")
            return article_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting news article: {e}")
            return None
        
        finally:
            conn.close()
    
    def insert_anomaly(self, anomaly: Dict[str, Any]) -> Optional[int]:
        """
        Insert market anomaly into the database.
        
        Args:
            anomaly: Anomaly dictionary
            
        Returns:
            ID of inserted anomaly or None if error
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Ensure timestamp is in ISO format
            if isinstance(anomaly['timestamp'], datetime):
                timestamp = anomaly['timestamp'].isoformat()
            else:
                timestamp = anomaly['timestamp']
            
            # Insert anomaly
            cursor.execute('''
            INSERT OR REPLACE INTO anomalies 
            (symbol, anomaly_type, description, severity, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                anomaly['symbol'],
                anomaly['anomaly_type'],
                anomaly.get('description', ''),
                anomaly.get('severity', 1.0),
                timestamp
            ))
            
            # Get anomaly ID
            anomaly_id = cursor.lastrowid
            
            conn.commit()
            
            logger.info(f"Inserted anomaly for {anomaly['symbol']}: {anomaly['anomaly_type']}")
            return anomaly_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting anomaly: {e}")
            return None
        
        finally:
            conn.close()
    
    def insert_model_prediction(self, prediction: Dict[str, Any]) -> Optional[int]:
        """
        Insert model prediction into the database.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            ID of inserted prediction or None if error
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Ensure dates are in ISO format
            if isinstance(prediction['prediction_date'], datetime):
                prediction_date = prediction['prediction_date'].isoformat()
            else:
                prediction_date = prediction['prediction_date']
            
            if isinstance(prediction['target_date'], datetime):
                target_date = prediction['target_date'].isoformat()
            else:
                target_date = prediction['target_date']
            
            # Insert prediction
            cursor.execute('''
            INSERT OR REPLACE INTO model_predictions 
            (symbol, model_name, target, prediction_value, confidence, prediction_date, target_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction['symbol'],
                prediction['model_name'],
                prediction['target'],
                prediction['prediction_value'],
                prediction.get('confidence', None),
                prediction_date,
                target_date
            ))
            
            # Get prediction ID
            prediction_id = cursor.lastrowid
            
            conn.commit()
            
            logger.info(f"Inserted prediction for {prediction['symbol']}: {prediction['target']}")
            return prediction_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting prediction: {e}")
            return None
        
        finally:
            conn.close()
    
    def get_latest_prices(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get latest stock prices.
        
        Args:
            symbols: Optional list of symbols to filter
            
        Returns:
            DataFrame with latest prices
        """
        conn = sqlite3.connect(self.db_path)
        
        if symbols:
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
        else:
            query = """
            SELECT p.*
            FROM stock_prices p
            INNER JOIN (
                SELECT symbol, MAX(timestamp) as max_timestamp
                FROM stock_prices
                GROUP BY symbol
            ) m ON p.symbol = m.symbol AND p.timestamp = m.max_timestamp
            """
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='raise')
        
        return df
    
    def get_historical_prices(self, symbol: str, start_date: Optional[Union[str, datetime]] = None, 
                              end_date: Optional[Union[str, datetime]] = None, period: str = None) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Stock symbol
            start_date: Optional start date
            end_date: Optional end date
            period: Optional period filter
            
        Returns:
            DataFrame with historical prices
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = "SELECT * FROM ohlcv WHERE symbol = ?"
        params = [symbol]
        
        if period:
            query += " AND period = ?"
            params.append(period)
        
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.isoformat()
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp ASC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_historical_stock_prices(self, symbol: str, start_date: Optional[Union[str, datetime]] = None,
                                  end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get historical stock prices from the stock_prices table.
        
        Args:
            symbol: Stock symbol
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with historical stock prices containing:
            - symbol: Stock symbol
            - price: Current price
            - change: Price change
            - change_percent: Percentage change
            - volume: Trading volume
            - timestamp: Timestamp of the price
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = "SELECT * FROM stock_prices WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.isoformat()
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp ASC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_technical_indicators(self, symbol: str, indicator_type: Optional[str] = None, 
                                 start_date: Optional[Union[str, datetime]] = None,
                                 end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get technical indicators.
        
        Args:
            symbol: Stock symbol
            indicator_type: Optional indicator type filter
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with technical indicators
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = "SELECT * FROM technical_indicators WHERE symbol = ?"
        params = [symbol]
        
        if indicator_type:
            query += " AND indicator_type = ?"
            params.append(indicator_type)
        
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.isoformat()
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp ASC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_news_for_symbol(self, symbol: str, limit: int = 100, 
                            start_date: Optional[Union[str, datetime]] = None,
                            end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get news articles for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of articles to return
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with news articles
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = """
        SELECT a.*, s.relevance_score
        FROM news_articles a
        INNER JOIN news_symbols s ON a.id = s.news_id
        WHERE s.symbol = ?
        """
        params = [symbol]
        
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.isoformat()
            query += " AND a.published_date >= ?"
            params.append(start_date)
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
            query += " AND a.published_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY a.published_date DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert published_date to datetime
        if 'published_date' in df.columns:
            df['published_date'] = pd.to_datetime(df['published_date'])
        
        return df
    
    def get_anomalies(self, symbol: Optional[str] = None, anomaly_type: Optional[str] = None,
                     start_date: Optional[Union[str, datetime]] = None,
                     end_date: Optional[Union[str, datetime]] = None,
                     min_severity: Optional[float] = None) -> pd.DataFrame:
        """
        Get market anomalies.
        
        Args:
            symbol: Optional stock symbol filter
            anomaly_type: Optional anomaly type filter
            start_date: Optional start date
            end_date: Optional end date
            min_severity: Optional minimum severity filter
            
        Returns:
            DataFrame with anomalies
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = "SELECT * FROM anomalies WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if anomaly_type:
            query += " AND anomaly_type = ?"
            params.append(anomaly_type)
        
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.isoformat()
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        if min_severity is not None:
            query += " AND severity >= ?"
            params.append(min_severity)
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_model_predictions(self, symbol: str, model_name: Optional[str] = None,
                             target: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """
        Get model predictions.
        
        Args:
            symbol: Stock symbol
            model_name: Optional model name filter
            target: Optional prediction target filter
            limit: Maximum number of predictions to return
            
        Returns:
            DataFrame with predictions
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = "SELECT * FROM model_predictions WHERE symbol = ?"
        params = [symbol]
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        if target:
            query += " AND target = ?"
            params.append(target)
        
        query += " ORDER BY prediction_date DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert dates to datetime
        if 'prediction_date' in df.columns:
            df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        
        if 'target_date' in df.columns:
            df['target_date'] = pd.to_datetime(df['target_date'])
        
        return df
    
    def run_custom_query(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Run custom SQL query.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            DataFrame with query results
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
                
            return df
            
        except Exception as e:
            logger.error(f"Error running custom query: {e}")
            return pd.DataFrame()
            
        finally:
            conn.close()
    
    def optimize_database(self) -> bool:
        """
        Optimize database performance.
        
        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Run VACUUM to rebuild the database file
            cursor.execute("VACUUM")
            
            # Run ANALYZE to update statistics
            cursor.execute("ANALYZE")
            
            conn.commit()
            logger.info("Database optimization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False
            
        finally:
            conn.close()
    
    def backup_database(self, backup_path: Optional[str] = None) -> Optional[str]:
        """
        Backup database to file.
        
        Args:
            backup_path: Optional path for backup file
            
        Returns:
            Path to backup file or None if error
        """
        if not backup_path:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_name = os.path.basename(self.db_path)
            db_dir = os.path.dirname(self.db_path)
            backup_path = os.path.join(db_dir, f"{db_name}.{timestamp}.bak")
        
        try:
            # Create connection to source database
            source = sqlite3.connect(self.db_path)
            
            # Create connection to backup database
            backup = sqlite3.connect(backup_path)
            
            # Copy data
            source.backup(backup)
            
            # Close connections
            backup.close()
            source.close()
            
            logger.info(f"Database backup created at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return None
    
    def get_level2_data(self, symbol: str) -> pd.DataFrame:
        """
        Get the latest level 2 market data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with level 2 market data containing:
            - symbol: Stock symbol
            - bid_price_1 to bid_price_5: Bid prices for 5 levels
            - bid_volume_1 to bid_volume_5: Bid volumes for 5 levels
            - ask_price_1 to ask_price_5: Ask prices for 5 levels
            - ask_volume_1 to ask_volume_5: Ask volumes for 5 levels
            - timestamp: Timestamp of the data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT *
        FROM level2_data
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        df = pd.read_sql_query(query, conn, params=[symbol])
        conn.close()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

    def insert_level2_data(self, data: Dict[str, Any]) -> bool:
        """
        Insert level 2 market data into the database.
        
        Args:
            data: Dictionary containing level 2 market data
            
        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Ensure timestamp is in ISO format
            if isinstance(data['timestamp'], datetime):
                timestamp = data['timestamp'].isoformat()
            else:
                timestamp = data['timestamp']
            
            cursor.execute('''
            INSERT OR REPLACE INTO level2_data 
            (symbol, bid_price_1, bid_volume_1, bid_price_2, bid_volume_2,
             bid_price_3, bid_volume_3, bid_price_4, bid_volume_4,
             bid_price_5, bid_volume_5, ask_price_1, ask_volume_1,
             ask_price_2, ask_volume_2, ask_price_3, ask_volume_3,
             ask_price_4, ask_volume_4, ask_price_5, ask_volume_5,
             timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['symbol'],
                data.get('bid_price_1', 0),
                data.get('bid_volume_1', 0),
                data.get('bid_price_2', 0),
                data.get('bid_volume_2', 0),
                data.get('bid_price_3', 0),
                data.get('bid_volume_3', 0),
                data.get('bid_price_4', 0),
                data.get('bid_volume_4', 0),
                data.get('bid_price_5', 0),
                data.get('bid_volume_5', 0),
                data.get('ask_price_1', 0),
                data.get('ask_volume_1', 0),
                data.get('ask_price_2', 0),
                data.get('ask_volume_2', 0),
                data.get('ask_price_3', 0),
                data.get('ask_volume_3', 0),
                data.get('ask_price_4', 0),
                data.get('ask_volume_4', 0),
                data.get('ask_price_5', 0),
                data.get('ask_volume_5', 0),
                timestamp
            ))
            
            conn.commit()
            logger.info(f"Inserted level 2 data for {data['symbol']}")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting level 2 data: {e}")
            return False
            
        finally:
            conn.close()
 