#!/usr/bin/env python3
"""
Example script demonstrating the usage of funny_duration.py with different configurations
"""

import subprocess
import sys
import pandas as pd

def run_duration_analysis(ticker, days=180, min_events=5, check_interval=20):
    """Run the duration analysis script"""
    
    cmd = [
        sys.executable, 'funny_duration.py', ticker,
        '--days', str(days),
        '--min-events', str(min_events),
        '--check-interval', str(check_interval)
    ]
    
    print(f"Running duration analysis for {ticker}...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running analysis: {e}")
        return False

def analyze_duration_results(ticker):
    """Analyze the results from duration tracking"""
    
    try:
        # Look for the most recent results file
        import glob
        import os
        files = glob.glob(f"{ticker}_duration_analysis_*.csv")
        if not files:
            print(f"No duration analysis results found for {ticker}")
            return
        
        latest_file = max(files, key=os.path.getctime)
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        
        # Filter to earnings events
        earnings_events = df[df['is_earnings_date'] == True]
        
        if earnings_events.empty:
            print("No earnings events found in the results")
            return
        
        print(f"\n=== DURATION ANALYSIS SUMMARY FOR {ticker} ===")
        print(f"Total earnings events analyzed: {len(earnings_events)}")
        
        # Summary statistics
        avg_duration = earnings_events['duration_hours'].mean()
        max_duration = earnings_events['duration_hours'].max()
        min_duration = earnings_events['duration_hours'].min()
        
        print(f"Average pattern duration: {avg_duration:.2f} hours")
        print(f"Longest pattern duration: {max_duration:.2f} hours")
        print(f"Shortest pattern duration: {min_duration:.2f} hours")
        
        # Pattern change analysis
        total_changes = earnings_events['pattern_changes'].astype(str).astype(int).sum()
        avg_changes = total_changes / len(earnings_events)
        print(f"Average pattern changes per event: {avg_changes:.1f}")
        
        # Classification distribution
        print(f"\nPrimary pattern distribution:")
        class_counts = earnings_events['earnings_classification'].value_counts()
        for pattern, count in class_counts.items():
            percentage = (count / len(earnings_events)) * 100
            print(f"  {pattern}: {count} events ({percentage:.1f}%)")
        
        # Final vs Primary classification comparison
        print(f"\nPattern persistence analysis:")
        same_pattern = (earnings_events['earnings_classification'] == earnings_events['final_classification']).sum()
        changed_pattern = len(earnings_events) - same_pattern
        
        print(f"  Patterns that remained consistent: {same_pattern} ({same_pattern/len(earnings_events)*100:.1f}%)")
        print(f"  Patterns that changed during the day: {changed_pattern} ({changed_pattern/len(earnings_events)*100:.1f}%)")
        
        # Detailed event breakdown
        print(f"\n=== INDIVIDUAL EVENT DETAILS ===")
        for date, event in earnings_events.iterrows():
            print(f"\n{date.strftime('%Y-%m-%d')}:")
            print(f"  Primary pattern: {event['earnings_classification']}")
            print(f"  Final pattern: {event['final_classification']}")
            print(f"  Duration: {event['duration_hours']:.2f} hours ({event['duration_minutes']:.0f} minutes)")
            print(f"  Pattern changes: {event['pattern_changes']}")
            print(f"  Max strength: {event['event_strength']:.2f}%")
            
            pattern_changed = event['earnings_classification'] != event['final_classification']
            if pattern_changed:
                print(f"  >>> PATTERN EVOLVED FROM {event['earnings_classification'].upper()} TO {event['final_classification'].upper()}")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")

def compare_intervals():
    """Compare different check intervals for the same stock"""
    
    ticker = "AAPL"
    intervals = [15, 20, 30]
    
    print(f"=== COMPARING CHECK INTERVALS FOR {ticker} ===")
    
    results = {}
    for interval in intervals:
        print(f"\nTesting {interval}-minute intervals...")
        success = run_duration_analysis(ticker, days=120, min_events=3, check_interval=interval)
        if success:
            results[interval] = f"{ticker}_duration_analysis_*.csv"
    
    print(f"\n=== INTERVAL COMPARISON SUMMARY ===")
    print("Different check intervals can affect:")
    print("1. Detected pattern duration (finer intervals = more precise)")
    print("2. Number of pattern changes detected")
    print("3. Final classification accuracy")
    print("4. Computational time (more frequent checks = slower)")

def main():
    """Run demonstration examples"""
    
    print("=== FUNNY_DURATION.PY DEMONSTRATION ===")
    print("This script demonstrates event duration tracking capabilities\n")
    
    # Example 1: Basic duration analysis
    print("Example 1: Basic duration analysis with 20-minute intervals")
    run_duration_analysis("TSLA", days=180, min_events=4, check_interval=20)
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: Fine-grained analysis
    print("Example 2: Fine-grained analysis with 15-minute intervals")
    run_duration_analysis("MSFT", days=120, min_events=3, check_interval=15)
    
    print("\n" + "="*80 + "\n")
    
    # Example 3: Coarse analysis
    print("Example 3: Coarse analysis with 30-minute intervals")
    run_duration_analysis("GOOGL", days=200, min_events=5, check_interval=30)
    
    print("\n" + "="*80 + "\n")
    
    # Analyze results
    print("Example 4: Analyzing duration results")
    analyze_duration_results("NVDA")  # Analyze the NVDA results we generated earlier
    
    print("\n=== KEY INSIGHTS FROM DURATION ANALYSIS ===")
    print("1. Event Type Persistence: How long does the initial reaction pattern last?")
    print("2. Pattern Evolution: Do anticipated events become surprising or vice versa?")
    print("3. Optimal Timing: When is the best time to act on earnings reactions?")
    print("4. Market Efficiency: How quickly do prices reflect new information?")
    print("5. Volatility Windows: Identify the most volatile periods after earnings")

if __name__ == "__main__":
    main()