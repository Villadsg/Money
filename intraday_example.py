#!/usr/bin/env python3
"""
Example script demonstrating how to use onlyant.py with intraday data
"""

import subprocess
import sys

def run_analysis(ticker, use_intraday=False, intraday_timespan='minute', multiplier=15, days=90):
    """Run the enhanced onlyant.py analysis"""
    
    cmd = [sys.executable, 'onlyant.py', ticker, '--days', str(days)]
    
    if use_intraday:
        cmd.extend([
            '--use-intraday',
            '--intraday-timespan', intraday_timespan,
            '--intraday-multiplier', str(multiplier)
        ])
    
    print(f"Running analysis for {ticker}...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running analysis: {e}")
        return False

def main():
    """Run examples showing different configurations"""
    
    # Example 1: Traditional daily data analysis
    print("Example 1: Traditional daily data analysis")
    run_analysis('AAPL', use_intraday=False, days=180)
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: Enhanced analysis with 15-minute intraday data
    print("Example 2: Enhanced analysis with 15-minute intraday data")
    run_analysis('AAPL', use_intraday=True, intraday_timespan='minute', multiplier=15, days=60)
    
    print("\n" + "="*80 + "\n")
    
    # Example 3: Enhanced analysis with 1-hour intraday data
    print("Example 3: Enhanced analysis with 1-hour intraday data")
    run_analysis('TSLA', use_intraday=True, intraday_timespan='hour', multiplier=1, days=120)

if __name__ == "__main__":
    main()