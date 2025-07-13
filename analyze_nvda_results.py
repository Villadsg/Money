import pandas as pd

# Load the duration analysis results
df = pd.read_csv('NVDA_duration_analysis_20250625.csv', index_col=0, parse_dates=True)
earnings_events = df[df['is_earnings_date'] == True]

print('=== NVDA DURATION ANALYSIS INSIGHTS ===')
print('Total earnings events:', len(earnings_events))

for date, event in earnings_events.iterrows():
    print()
    print(date.strftime('%Y-%m-%d'), ':')
    print('  Primary:', event['earnings_classification'])
    print('  Final:', event['final_classification'])
    print('  Duration: {:.2f}h ({:.0f}min)'.format(event['duration_hours'], event['duration_minutes']))
    print('  Changes:', event['pattern_changes'])
    print('  Strength: {:.2f}%'.format(event['event_strength']))
    
    if event['earnings_classification'] != event['final_classification']:
        print('  >>> PATTERN EVOLVED!')

print('\n=== SUMMARY STATISTICS ===')
avg_duration = earnings_events['duration_hours'].mean()
print('Average duration: {:.2f} hours'.format(avg_duration))

total_changes = earnings_events['pattern_changes'].astype(float).astype(int).sum()
avg_changes = total_changes / len(earnings_events)
print('Average pattern changes: {:.1f}'.format(avg_changes))

pattern_evolved = (earnings_events['earnings_classification'] != earnings_events['final_classification']).sum()
print('Events that evolved: {}/{} ({:.1f}%)'.format(
    pattern_evolved, len(earnings_events), pattern_evolved/len(earnings_events)*100))