"""
Test the financial data analyzer.
"""
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_data.data.analyzer import FinancialDataAnalyzer

def test_analyzer():
    """Test the financial data analyzer."""
    # Use the test database
    db_path = "test_finance.db"
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found. Please run the producer test first.")
        return False
    
    analyzer = FinancialDataAnalyzer(db_path)
    results = analyzer.analyze_market_data()
    
    # Check if we got results
    if not results:
        print("No results returned from analyzer")
        return False
    
    # Check if we have market summary
    if "market_summary" not in results:
        print("No market summary in results")
        return False
    
    # Check if we have technical indicators
    if "ma_df" not in results or "rsi_df" not in results or "vol_df" not in results:
        print("Missing technical indicators in results")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing financial data analyzer...")
    success = test_analyzer()
    print(f"\nAnalyzer test {'PASSED' if success else 'FAILED'}")
