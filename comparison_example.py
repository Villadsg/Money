#!/usr/bin/env python3
"""
Example script showing how different time windows affect earnings reaction classification
"""

import pandas as pd
import matplotlib.pyplot as plt

def compare_classifications():
    """Compare earnings classifications across different time windows"""
    
    # Load the different analysis results
    try:
        traditional = pd.read_csv('NVDA_analysis_20250625.csv', index_col=0, parse_dates=True)
        min_30 = pd.read_csv('NVDA_analysis_30min_20250625.csv', index_col=0, parse_dates=True)
        min_60 = pd.read_csv('NVDA_analysis_60min_20250625.csv', index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the funny_intraday.py script first to generate the comparison data.")
        return
    
    # Filter to earnings events only
    traditional_earnings = traditional[traditional['is_earnings_date'] == True]
    min_30_earnings = min_30[min_30['is_earnings_date'] == True]
    min_60_earnings = min_60[min_60['is_earnings_date'] == True]
    
    print("=== EARNINGS CLASSIFICATION COMPARISON ===")
    print("\nTraditional (Close Price):")
    print(traditional_earnings['earnings_classification'].value_counts())
    
    print("\n30 Minutes After Open:")
    print(min_30_earnings['earnings_classification'].value_counts())
    
    print("\n60 Minutes After Open:")
    print(min_60_earnings['earnings_classification'].value_counts())
    
    print("\n=== DETAILED COMPARISON ===")
    for date in traditional_earnings.index:
        trad_class = traditional_earnings.loc[date, 'earnings_classification']
        min30_class = min_30_earnings.loc[date, 'earnings_classification']
        min60_class = min_60_earnings.loc[date, 'earnings_classification']
        
        trad_strength = traditional_earnings.loc[date, 'event_strength']
        min30_strength = min_30_earnings.loc[date, 'event_strength']
        min60_strength = min_60_earnings.loc[date, 'event_strength']
        
        print(f"\nDate: {date.strftime('%Y-%m-%d')}")
        print(f"  Traditional:   {trad_class:20} (Strength: {trad_strength:5.2f}%)")
        print(f"  +30 minutes:   {min30_class:20} (Strength: {min30_strength:5.2f}%)")
        print(f"  +60 minutes:   {min60_class:20} (Strength: {min60_strength:5.2f}%)")
        
        # Highlight if classifications differ
        if not (trad_class == min30_class == min60_class):
            print(f"  >>> CLASSIFICATION CHANGED OVER TIME <<<")

def plot_comparison():
    """Create a visual comparison of the different time windows"""
    
    try:
        traditional = pd.read_csv('NVDA_analysis_20250625.csv', index_col=0, parse_dates=True)
        min_30 = pd.read_csv('NVDA_analysis_30min_20250625.csv', index_col=0, parse_dates=True)
        min_60 = pd.read_csv('NVDA_analysis_60min_20250625.csv', index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Filter to earnings events
    earnings_dates = traditional[traditional['is_earnings_date'] == True].index
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Event Strength Comparison
    trad_strengths = [traditional.loc[date, 'event_strength'] for date in earnings_dates]
    min30_strengths = [min_30.loc[date, 'event_strength'] for date in earnings_dates]
    min60_strengths = [min_60.loc[date, 'event_strength'] for date in earnings_dates]
    
    x = range(len(earnings_dates))
    width = 0.25
    
    axes[0].bar([i - width for i in x], trad_strengths, width, label='Traditional (Close)', alpha=0.7)
    axes[0].bar(x, min30_strengths, width, label='+30 Minutes', alpha=0.7)
    axes[0].bar([i + width for i in x], min60_strengths, width, label='+60 Minutes', alpha=0.7)
    
    axes[0].set_title('Event Strength Comparison Across Time Windows')
    axes[0].set_ylabel('Event Strength (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([d.strftime('%m-%d') for d in earnings_dates], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Classification Changes
    # Create a heatmap showing how classifications change
    class_map = {
        'negative_anticipated': 1,
        'positive_anticipated': 2,
        'surprising_negative': 3,
        'surprising_positive': 4
    }
    
    trad_classes = [class_map.get(traditional.loc[date, 'earnings_classification'], 0) for date in earnings_dates]
    min30_classes = [class_map.get(min_30.loc[date, 'earnings_classification'], 0) for date in earnings_dates]
    min60_classes = [class_map.get(min_60.loc[date, 'earnings_classification'], 0) for date in earnings_dates]
    
    classification_data = [trad_classes, min30_classes, min60_classes]
    
    im = axes[1].imshow(classification_data, aspect='auto', cmap='viridis')
    axes[1].set_title('Classification Changes Across Time Windows')
    axes[1].set_ylabel('Time Window')
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(['Traditional', '+30min', '+60min'])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.strftime('%m-%d') for d in earnings_dates], rotation=45)
    
    # Add color bar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels(['Neg Anticipated', 'Pos Anticipated', 'Surprising Neg', 'Surprising Pos'])
    
    plt.tight_layout()
    plt.savefig('data/earnings_classification_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to data/earnings_classification_comparison.png")
    plt.close()

def main():
    """Run the comparison analysis"""
    print("Analyzing earnings reaction classifications across different time windows...\n")
    
    compare_classifications()
    plot_comparison()
    
    print("\n=== KEY INSIGHTS ===")
    print("1. Time window affects classification: Early reactions may reverse later")
    print("2. Event strength varies: Some stocks show peak volatility at different times")
    print("3. 30-60 minutes often captures institutional reaction vs retail emotion")
    print("4. Traditional closing price may include full-day noise vs pure reaction")

if __name__ == "__main__":
    main()