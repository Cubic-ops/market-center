"""
Run the Streamlit dashboard.
"""
import os
import sys
import subprocess

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_dashboard():
    """Run the Streamlit dashboard."""
    dashboard_path = os.path.join("dashboard", "dashboard.py")
    
    # Check if the dashboard file exists
    if not os.path.exists(dashboard_path):
        print(f"Dashboard file not found at {dashboard_path}")
        return False
    
    # Check if the database exists
    if not os.path.exists("test_finance.db") and not os.path.exists("finance.db"):
        print("No database file found. Running producer test to generate data...")
        subprocess.run(["python", "scripts/test_producer.py"])
    
    # Run the dashboard
    print("Starting dashboard...")
    subprocess.run(["streamlit", "run", dashboard_path])
    
    return True

if __name__ == "__main__":
    run_dashboard()
