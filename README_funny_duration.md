# Funny Duration Analysis - Enhanced Earnings Reaction Tracker

## Overview

`funny_duration.py` is an enhanced version of `funny.py` that tracks how long anticipation and surprise patterns last throughout the trading day. Instead of just classifying events based on closing prices, it monitors price movements every N minutes to determine pattern persistence and evolution.

## Key Features

### 1. **Duration Tracking**
- Monitors earnings reaction patterns every 20 minutes (configurable)
- Tracks how long the initial pattern (anticipation/surprise) persists
- Records when patterns change during the trading day

### 2. **Pattern Evolution Analysis**
- **Primary Classification**: The longest-sustained pattern during the day
- **Final Classification**: The pattern at market close
- **Pattern Changes**: Number of times the classification changed during the day

### 3. **Enhanced Metrics**
- **Duration Hours/Minutes**: How long the primary pattern lasted
- **Event Strength Timeline**: Maximum volatility reached during the event
- **Pattern Consistency**: Whether the pattern remained stable or evolved

## Usage

```bash
# Basic usage with 20-minute intervals
python3 funny_duration.py NVDA --days 180 --min-events 5 --check-interval 20

# Fine-grained analysis with 15-minute intervals
python3 funny_duration.py AAPL --days 365 --min-events 10 --check-interval 15

# Coarse analysis with 30-minute intervals  
python3 funny_duration.py TSLA --days 120 --min-events 3 --check-interval 30
```

### Command Line Arguments

- `ticker`: Stock symbol to analyze
- `--days`: Number of days to analyze (default: 365)
- `--min-events`: Number of earnings events to identify (default: 15)
- `--check-interval`: Minutes between pattern checks (default: 20)
- `--benchmark`: Benchmark ticker (default: SPY)

## Key Insights from NVDA Analysis

### Event Summary
- **3 major earnings events** analyzed over 180 days
- **Average pattern duration**: 2.67 hours
- **Pattern evolution rate**: 66.7% of events changed classification during the day

### Individual Events

#### 2025-01-27 (Major Earnings Event)
- **Gap**: -12.49% (negative gap)
- **Primary Pattern**: Negative Anticipated (2.5 hours)
- **Evolution**: Changed to Surprising Negative
- **Pattern Changes**: 5 times during the day
- **Insight**: Initial recovery attempt failed, became full selloff

#### 2025-01-28 (Recovery Day)
- **Gap**: +2.86% (positive gap)  
- **Primary Pattern**: Surprising Positive (2.5 hours)
- **Evolution**: Changed to Positive Anticipated
- **Pattern Changes**: 3 times during the day
- **Insight**: Strong opening momentum eventually stalled

#### 2025-02-27 (Stable Event)
- **Gap**: +2.83% (positive gap)
- **Primary Pattern**: Positive Anticipated (3.0 hours)
- **Evolution**: Remained Positive Anticipated
- **Pattern Changes**: Only 2 times during the day
- **Insight**: Most stable pattern, but still ended poorly

## Pattern Definitions

### During Trading Day Classification
1. **Negative Anticipated**: Stock gaps down but recovers intraday
2. **Surprising Negative**: Stock gaps down and continues falling
3. **Positive Anticipated**: Stock gaps up but declines intraday  
4. **Surprising Positive**: Stock gaps up and continues rising

### Key Differences from Traditional Analysis

| Aspect | Traditional (funny.py) | Duration Enhanced (funny_duration.py) |
|--------|----------------------|--------------------------------------|
| **Time Scope** | Open vs Close only | Every 20 minutes throughout day |
| **Pattern Detection** | Single classification | Primary + evolution tracking |
| **Duration Info** | None | Hours/minutes of pattern persistence |
| **Pattern Changes** | Not tracked | Counts transitions during day |
| **Market Timing** | End-of-day only | Real-time pattern shifts |

## Practical Applications

### 1. **Trading Strategy Optimization**
- Identify optimal exit times for earnings plays
- Understand when patterns typically reverse
- Set stop-losses based on historical pattern durations

### 2. **Risk Management**
- Anticipate pattern changes before they happen
- Understand typical volatility windows
- Plan position sizing based on pattern stability

### 3. **Market Efficiency Analysis**
- Measure how quickly markets process earnings information
- Identify stocks with unstable vs stable reaction patterns
- Study institutional vs retail reaction timing

### 4. **Algorithmic Trading Signals**
- Use pattern duration as a confidence metric
- Trigger alerts when patterns change unexpectedly
- Backtest strategies based on pattern persistence

## Technical Implementation

### Data Sources
- **Daily Data**: Polygon.io API for daily OHLCV
- **Intraday Data**: 1-minute bars for precise timing
- **Market Hours**: 9:30 AM - 4:00 PM ET analysis window

### Pattern Detection Algorithm
1. Identify earnings events using Volume × Gap metric
2. Determine initial gap direction (positive/negative)
3. Check price movement every N minutes
4. Classify current pattern based on gap + movement
5. Track pattern changes and calculate durations
6. Determine primary (longest) pattern vs final pattern

### Performance Metrics
- **Pattern Persistence**: How long patterns last on average
- **Evolution Rate**: Percentage of events that change classification
- **Stability Score**: Number of pattern changes as instability measure
- **Strength Timeline**: Maximum volatility reached during event window

## Future Enhancements

1. **Volume Analysis**: Track volume patterns during events
2. **Sector Comparison**: Compare pattern durations across sectors
3. **Market Cap Effects**: Analyze how company size affects pattern stability
4. **Options Activity**: Incorporate options flow for pattern prediction
5. **News Sentiment**: Correlate news sentiment with pattern evolution

## Files Generated

- `{TICKER}_duration_analysis_{DATE}.csv`: Complete analysis results
- `data/{TICKER}_duration_analysis.png`: Visualization with 6 plots including duration bars
- Duration tracking includes intraday price timeline and pattern change timestamps

This enhanced analysis provides unprecedented insight into the temporal dynamics of earnings reactions, enabling more sophisticated trading strategies and market understanding.