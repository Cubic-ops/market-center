"""
Deployment script for financial data platform.
"""
import os
import subprocess
import argparse

def setup_environment():
    """Set up the environment for the application."""
    print("Setting up environment...")
    subprocess.run(["pip", "install", "-e", "."])

def run_data_producer(continuous=False):
    """Run the market data producer."""
    print("Running market data producer...")
    if continuous:
        # Run in continuous mode
        subprocess.run(["python", "-m", "market_data.data.producer"])
    else:
        # Run once for testing
        subprocess.run(["python", "scripts/test_producer.py"])

def run_data_analyzer():
    """Run the data analyzer."""
    print("Running data analyzer...")
    subprocess.run(["python", "scripts/test_analyzer.py"])

def run_dashboard():
    """Run the Streamlit dashboard."""
    print("Running dashboard...")
    subprocess.run(["streamlit", "run", "dashboard/advanced_dashboard.py"])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial Data Platform Deployment")
    parser.add_argument("--setup", action="store_true", help="Set up the environment")
    parser.add_argument("--producer", action="store_true", help="Run the market data producer")
    parser.add_argument("--continuous", action="store_true", help="Run the producer in continuous mode")
    parser.add_argument("--analyzer", action="store_true", help="Run the data analyzer")
    parser.add_argument("--dashboard", action="store_true", help="Run the dashboard")
    parser.add_argument("--all", action="store_true", help="Run all components")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.setup or args.all:
        setup_environment()
    
    if args.producer or args.all:
        run_data_producer(args.continuous)
    
    if args.analyzer or args.all:
        run_data_analyzer()
    
    if args.dashboard or args.all:
        run_dashboard()

if __name__ == "__main__":
    main()
