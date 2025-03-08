# Stock Analyzer

A powerful tool for analyzing stocks against market benchmarks and making buy/sell suggestions based on residual analysis.

## Overview

This application analyzes how a stock performs relative to a market benchmark (default: MSCI World ETF) using linear regression. It then provides buy/sell signals based on the residuals from this regression, which represent the difference between the actual stock price and what would be expected based on the market's performance.

The key insight: When residuals are negative (stock price is below the regression line), it may indicate an undervalued stock and a potential buying opportunity. Conversely, positive residuals may indicate an overvalued stock and a potential selling opportunity.

## Features

- **Stock vs Market Analysis**: Compare any stock against various market benchmarks
- **Flexible Timeframes**: Analyze data over different periods (1 month to 5 years)
- **Residual Analysis**: Identify potential buy/sell opportunities based on regression residuals
- **Market Segment Analysis**: Analyze multiple tickers as a market segment and detect correlation changes
- **Correlation-Based Signals**: Identify buy/sell opportunities when a stock's correlation with its segment changes
- **Money Flow Analysis**: View weekly money flow (volume × price) alongside price movements
- **Interactive Visualizations**: Explore data through interactive charts
- **Historical Signals**: See how buy/sell signals would have performed over time
- **Command Line Interface**: Run quick analyses from the terminal
- **Web Application**: User-friendly interface built with Streamlit

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stock-analyzer.git
   cd stock-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web Application

Run the Streamlit web application:

```
streamlit run app.py
```

Then open your browser and navigate to the URL displayed in the terminal (usually http://localhost:8501).

### Command Line Interface

Analyze a stock from the command line:

```
python stock_analyzer.py AAPL
```

With custom parameters:

```
python stock_analyzer.py AAPL --market SPY --timeframe 2y --threshold -0.5
```

Available options:
- `--market`: Market benchmark ticker (default: URTH - MSCI World ETF)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--timeframe`: Analysis timeframe (1m, 3m, 6m, 1y, 2y, 5y, max)
- `--threshold`: Residual threshold for buy/sell signals
- `--lookback`: Lookback period in days for signal context
- `--no-plots`: Disable plot generation

## How It Works

1. **Data Collection**: Historical price data is fetched for both the stock and the market benchmark.
2. **Linear Regression**: A regression model is built to predict the stock price based on the market price.
3. **Residual Analysis**: The differences between actual and predicted prices (residuals) are analyzed.
4. **Signal Generation**: Buy/sell signals are generated based on residual values and their recent trends.

### Interpretation

- **Negative residuals**: Stock may be undervalued relative to the market (potential buy)
- **Positive residuals**: Stock may be overvalued relative to the market (potential sell)
- **Signal strength**: Determined by how far the residual is from its recent average (z-score)

### Market Segment Analysis

The application can analyze multiple tickers as a market segment to identify correlation-based signals:

1. **Correlation Analysis**: Calculate correlations between residuals of stocks in the segment
2. **Correlation Changes**: Detect when a stock's correlation with the segment changes significantly
3. **Signal Generation**: Generate buy/sell signals based on correlation changes and residual values

#### Correlation-Based Signal Interpretation

- **Decreased correlation + negative residual**: Strong buy signal (stock hasn't risen yet but is diverging from segment)
- **Decreased correlation + positive residual**: Sell signal (only this stock has risen while diverging from segment)
- **Increased correlation + positive residual**: Strong sell signal (stock has risen and is now more correlated with segment)
- **Increased correlation + negative residual**: Buy signal (stock hasn't risen but is becoming more correlated with segment)

## Example

Analyzing Apple (AAPL) against the S&P 500 (SPY) over a 2-year period:

```
python stock_analyzer.py AAPL --market SPY --timeframe 2y
```

This will:
1. Fetch historical data for AAPL and SPY
2. Perform regression analysis
3. Calculate residuals and generate buy/sell signals
4. Create visualizations of the analysis
5. Print a summary of the results

## Disclaimer

This tool is for informational purposes only and does not constitute investment advice. Always do your own research and consider consulting a financial advisor before making investment decisions.

## License

MIT License
