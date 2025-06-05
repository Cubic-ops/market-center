"""
Test the market data producer functionality.
"""
import os
import sqlite3
from market_data.data.producer import MarketDataProducer

def test_fetch_prices():
    """Test fetching and saving real-time prices."""
    # Use in-memory database for testing
    producer = MarketDataProducer(":memory:")
    
    # Fetch price data
    prices = producer.fetch_realtime_prices()
    
    print(f"Fetched {len(prices)} price records:")
    for price in prices:
        print(f"{price.symbol}: ${price.price:.2f} ({price.change_percent:.2f}%) - Volume: {price.volume}")
    
    # Save price data
    producer.save_price_data(prices)
    
    return len(prices) > 0

def test_fetch_ohlcv():
    """Test fetching and saving OHLCV data."""
    # Use in-memory database for testing
    producer = MarketDataProducer(":memory:")
    
    # Fetch OHLCV data
    ohlcv_data = producer.fetch_ohlcv_data(period="1d", interval="1h")
    
    print(f"Fetched {len(ohlcv_data)} OHLCV records")
    
    # Group by symbol
    by_symbol = {}
    for ohlcv in ohlcv_data:
        if ohlcv.symbol not in by_symbol:
            by_symbol[ohlcv.symbol] = []
        by_symbol[ohlcv.symbol].append(ohlcv)
    
    # Print sample for each symbol
    for symbol, data in by_symbol.items():
        print(f"{symbol}: {len(data)} records")
        if data:
            sample = data[0]
            print(f"  Sample: O=${sample.open:.2f}, H=${sample.high:.2f}, L=${sample.low:.2f}, C=${sample.close:.2f}, V={sample.volume}")
    
    # Save OHLCV data
    producer.save_ohlcv_data(ohlcv_data)
    
    return len(ohlcv_data) > 0

def verify_database(db_path):
    """Verify data was saved to the database."""
    if db_path == ":memory:":
        print("Skipping database verification for in-memory database")
        return True
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check price data
    cursor.execute("SELECT COUNT(*) FROM stock_prices")
    price_count = cursor.fetchone()[0]
    print(f"Database contains {price_count} price records")
    
    # Check OHLCV data
    cursor.execute("SELECT COUNT(*) FROM ohlcv")
    ohlcv_count = cursor.fetchone()[0]
    print(f"Database contains {ohlcv_count} OHLCV records")
    
    # Show some sample data
    if price_count > 0:
        cursor.execute("SELECT symbol, price, change_percent FROM stock_prices LIMIT 5")
        print("\nSample price data:")
        for row in cursor.fetchall():
            print(f"{row[0]}: ${row[1]:.2f} ({row[2]:.2f}%)")
    
    if ohlcv_count > 0:
        cursor.execute("SELECT symbol, open, high, low, close FROM ohlcv LIMIT 5")
        print("\nSample OHLCV data:")
        for row in cursor.fetchall():
            print(f"{row[0]}: O=${row[1]:.2f}, H=${row[2]:.2f}, L=${row[3]:.2f}, C=${row[4]:.2f}")
    
    conn.close()
    
    return price_count > 0 and ohlcv_count > 0

if __name__ == "__main__":
    print("Testing market data producer...")
    
    # Test with in-memory database
    print("\n=== Testing with in-memory database ===")
    price_test_passed = test_fetch_prices()
    print(f"\nPrice fetch test {'PASSED' if price_test_passed else 'FAILED'}")
    
    ohlcv_test_passed = test_fetch_ohlcv()
    print(f"\nOHLCV fetch test {'PASSED' if ohlcv_test_passed else 'FAILED'}")
    
    # Test with file database
    print("\n=== Testing with file database ===")
    producer = MarketDataProducer("test_finance.db")
    
    # Fetch and save price data
    prices = producer.fetch_realtime_prices()
    producer.save_price_data(prices)
    
    # Fetch and save OHLCV data
    ohlcv_data = producer.fetch_ohlcv_data()
    producer.save_ohlcv_data(ohlcv_data)
    
    # Verify database
    db_test_passed = verify_database("test_finance.db")
    print(f"\nDatabase test {'PASSED' if db_test_passed else 'FAILED'}")
    
    # Overall result
    if price_test_passed and ohlcv_test_passed and db_test_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
