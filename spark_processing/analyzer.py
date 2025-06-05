"""
Financial data analyzer using Spark.
"""
import os
import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, max, min, count, lit
from pyspark.sql.window import Window
import pyspark.sql.functions as F

class FinancialDataAnalyzer:
    """Analyzes financial data using Spark."""
    
    def __init__(self, db_path):
        """Initialize the analyzer with the database path."""
        self.db_path = db_path
        
        # Create Spark session with additional configurations
        self.spark = SparkSession.builder \
            .appName("FinancialDataAnalysis") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.host", "0.0.0.0") \
            .config("spark.driver.bindAddress", "0.0.0.0") \
            .config("spark.ui.enabled", "true") \
            .config("spark.ui.port", "4040") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .master("local[*]") \
            .getOrCreate()
        
        # Set log level
        self.spark.sparkContext.setLogLevel("WARN")
        
        print("Financial Data Analyzer initialized with Spark", self.spark.version)
    
    def load_price_data(self):
        """Load price data from the database into a Spark DataFrame."""
        # First load with pandas (easier with SQLite)
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM stock_prices"
        pdf = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert to Spark DataFrame
        sdf = self.spark.createDataFrame(pdf)
        
        # Add timestamp as proper timestamp type
        sdf = sdf.withColumn("timestamp", F.to_timestamp("timestamp"))
        
        # Cache the DataFrame
        sdf.cache()
        
        print(f"Loaded {sdf.count()} price records")
        return sdf
    
    def load_ohlcv_data(self):
        """Load OHLCV data from the database into a Spark DataFrame."""
        # First load with pandas (easier with SQLite)
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM ohlcv"
        pdf = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert to Spark DataFrame
        sdf = self.spark.createDataFrame(pdf)
        
        # Add timestamp as proper timestamp type
        sdf = sdf.withColumn("timestamp", F.to_timestamp("timestamp"))
        
        # Cache the DataFrame
        sdf.cache()
        
        print(f"Loaded {sdf.count()} OHLCV records")
        return sdf
    
    def calculate_moving_averages(self, ohlcv_df, window_sizes=[5, 10, 20]):
        """
        Calculate moving averages for each symbol.
        
        Args:
            ohlcv_df: Spark DataFrame with OHLCV data
            window_sizes: List of window sizes for moving averages
            
        Returns:
            Spark DataFrame with moving averages
        """
        # Ensure data is sorted by symbol and timestamp
        df = ohlcv_df.orderBy("symbol", "timestamp")
        
        # Create a window spec for each symbol ordered by timestamp
        window_spec = Window.partitionBy("symbol").orderBy("timestamp")
        
        # Calculate moving averages for different window sizes
        result_df = df
        
        for window_size in window_sizes:
            col_name = f"sma_{window_size}"
            result_df = result_df.withColumn(
                col_name,
                F.avg("close").over(window_spec.rowsBetween(-(window_size-1), 0))
            )
        
        return result_df
    
    def calculate_relative_strength(self, ohlcv_df, period=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            ohlcv_df: Spark DataFrame with OHLCV data
            period: RSI period (default: 14)
            
        Returns:
            Spark DataFrame with RSI values
        """
        # Ensure data is sorted by symbol and timestamp
        df = ohlcv_df.orderBy("symbol", "timestamp")
        
        # Create a window spec for each symbol ordered by timestamp
        window_spec = Window.partitionBy("symbol").orderBy("timestamp")
        
        # Calculate price changes
        df = df.withColumn("prev_close", F.lag("close", 1).over(window_spec))
        df = df.withColumn("price_change", col("close") - col("prev_close"))
        
        # Calculate gains (positive changes) and losses (negative changes)
        df = df.withColumn("gain", F.when(col("price_change") > 0, col("price_change")).otherwise(0))
        df = df.withColumn("loss", F.when(col("price_change") < 0, -col("price_change")).otherwise(0))
        
        # Calculate average gains and losses
        df = df.withColumn(
            "avg_gain",
            F.avg("gain").over(window_spec.rowsBetween(-(period-1), 0))
        )
        df = df.withColumn(
            "avg_loss",
            F.avg("loss").over(window_spec.rowsBetween(-(period-1), 0))
        )
        
        # Calculate RS and RSI
        df = df.withColumn(
            "rs", 
            F.when(col("avg_loss") != 0, col("avg_gain") / col("avg_loss")).otherwise(lit(100))
        )
        df = df.withColumn(
            "rsi",
            F.when(col("avg_loss") == 0, 100).otherwise(100 - (100 / (1 + col("rs"))))
        )
        
        return df
    
    def calculate_volatility(self, ohlcv_df, window_size=20):
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            ohlcv_df: Spark DataFrame with OHLCV data
            window_size: Window size for volatility calculation
            
        Returns:
            Spark DataFrame with volatility values
        """
        # Ensure data is sorted by symbol and timestamp
        df = ohlcv_df.orderBy("symbol", "timestamp")
        
        # Create a window spec for each symbol ordered by timestamp
        window_spec = Window.partitionBy("symbol").orderBy("timestamp")
        
        # Calculate returns
        df = df.withColumn("prev_close", F.lag("close", 1).over(window_spec))
        df = df.withColumn("return", (col("close") / col("prev_close")) - 1)
        
        # Calculate volatility (standard deviation of returns)
        df = df.withColumn(
            "volatility",
            F.stddev("return").over(window_spec.rowsBetween(-(window_size-1), 0))
        )
        
        return df
    
    def get_market_summary(self, price_df):
        """
        Generate a market summary from price data.
        
        Args:
            price_df: Spark DataFrame with price data
            
        Returns:
            Pandas DataFrame with market summary
        """
        # Get the latest price for each symbol
        latest_prices = price_df.groupBy("symbol").agg(
            F.max("timestamp").alias("latest_timestamp")
        )
        
        latest_data = price_df.join(
            latest_prices,
            (price_df.symbol == latest_prices.symbol) & 
            (price_df.timestamp == latest_prices.latest_timestamp)
        ).select(price_df["*"])
        
        # Calculate market statistics
        summary = latest_data.groupBy().agg(
            F.avg("price").alias("average_price"),
            F.avg("change_percent").alias("average_change_percent"),
            F.sum("volume").alias("total_volume"),
            F.count("symbol").alias("num_symbols"),
            F.sum(F.when(col("change") > 0, 1).otherwise(0)).alias("gainers"),
            F.sum(F.when(col("change") < 0, 1).otherwise(0)).alias("losers")
        )
        
        # Convert to pandas for easier handling
        return summary.toPandas()
    
    def get_top_movers(self, price_df, n=5):
        """
        Get top gaining and losing stocks.
        
        Args:
            price_df: Spark DataFrame with price data
            n: Number of top/bottom stocks to return
            
        Returns:
            Tuple of (top_gainers, top_losers) as Pandas DataFrames
        """
        # Get the latest price for each symbol
        latest_prices = price_df.groupBy("symbol").agg(
            F.max("timestamp").alias("latest_timestamp")
        )
        
        latest_data = price_df.join(
            latest_prices,
            (price_df.symbol == latest_prices.symbol) & 
            (price_df.timestamp == latest_prices.latest_timestamp)
        ).select(price_df["*"])
        
        # Get top gainers
        top_gainers = latest_data.orderBy(col("change_percent").desc()).limit(n)
        
        # Get top losers
        top_losers = latest_data.orderBy(col("change_percent").asc()).limit(n)
        
        return top_gainers.toPandas(), top_losers.toPandas()
    
    def analyze_market_data(self):
        """Perform comprehensive market data analysis."""
        # Load data
        price_df = self.load_price_data()
        ohlcv_df = self.load_ohlcv_data()
        
        # Calculate technical indicators
        ma_df = self.calculate_moving_averages(ohlcv_df)
        rsi_df = self.calculate_relative_strength(ohlcv_df)
        vol_df = self.calculate_volatility(ohlcv_df)
        
        # Get market summary
        market_summary = self.get_market_summary(price_df)
        
        # Get top movers
        top_gainers, top_losers = self.get_top_movers(price_df)
        
        # Print results
        print("\n=== Market Summary ===")
        print(f"Average Price: ${market_summary.iloc[0]['average_price']:.2f}")
        print(f"Average Change: {market_summary.iloc[0]['average_change_percent']:.2f}%")
        print(f"Total Volume: {market_summary.iloc[0]['total_volume']:,}")
        print(f"Gainers: {market_summary.iloc[0]['gainers']}")
        print(f"Losers: {market_summary.iloc[0]['losers']}")
        
        print("\n=== Top Gainers ===")
        for _, row in top_gainers.iterrows():
            print(f"{row['symbol']}: ${row['price']:.2f} ({row['change_percent']:.2f}%)")
        
        print("\n=== Top Losers ===")
        for _, row in top_losers.iterrows():
            print(f"{row['symbol']}: ${row['price']:.2f} ({row['change_percent']:.2f}%)")
        
        print("\n=== Technical Indicators (Sample) ===")
        # Show a sample of the technical indicators
        sample_ma = ma_df.filter(col("symbol") == "AAPL").orderBy(col("timestamp").desc()).limit(3)
        sample_ma.show()
        
        sample_rsi = rsi_df.filter(col("symbol") == "AAPL").orderBy(col("timestamp").desc()).limit(3)
        sample_rsi.select("symbol", "timestamp", "close", "rsi").show()
        
        sample_vol = vol_df.filter(col("symbol") == "AAPL").orderBy(col("timestamp").desc()).limit(3)
        sample_vol.select("symbol", "timestamp", "close", "volatility").show()
        
        return {
            "price_df": price_df,
            "ohlcv_df": ohlcv_df,
            "ma_df": ma_df,
            "rsi_df": rsi_df,
            "vol_df": vol_df,
            "market_summary": market_summary,
            "top_gainers": top_gainers,
            "top_losers": top_losers
        }
    
    def stop(self):
        """Stop the Spark session."""
        self.spark.stop()

if __name__ == "__main__":
    # Use the test database if it exists
    db_path = "test_finance.db"
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found. Please run the producer test first.")
        exit(1)
    
    analyzer = FinancialDataAnalyzer(db_path)
    try:
        analyzer.analyze_market_data()
    finally:
        analyzer.stop()
