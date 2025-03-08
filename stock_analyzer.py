#!/usr/bin/env python3
"""
Stock Analyzer - A tool for analyzing stocks against market benchmarks
and making buy/sell suggestions based on residual analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import date, timedelta
import argparse
import os
from typing import List, Dict, Union, Tuple


class StockAnalyzer:
    """Stock analysis tool that compares individual stocks against market benchmarks."""
    
    def __init__(self, ticker_symbol, market_symbol="URTH", 
                 start_date=None, end_date=None, timeframe="1y",
                 segment_tickers=None):
        """
        Initialize the StockAnalyzer with stock and market data.
        
        Args:
            ticker_symbol (str): Symbol of the stock to analyze
            market_symbol (str): Symbol of the market benchmark (default: URTH - MSCI World ETF)
            start_date (str): Start date for analysis in YYYY-MM-DD format
            end_date (str): End date for analysis in YYYY-MM-DD format (default: today)
            timeframe (str): Predefined timeframe if start_date is not provided
                             Options: '1m', '3m', '6m', '1y', '2y', '5y', 'max'
            segment_tickers (list): List of additional ticker symbols that constitute a market segment
        """
        self.ticker_symbol = ticker_symbol
        self.market_symbol = market_symbol
        self.segment_tickers = segment_tickers or []
        self.end_date = end_date if end_date else date.today()
        
        # Calculate start date based on timeframe if not explicitly provided
        if start_date:
            self.start_date = start_date
        else:
            if timeframe == "1m":
                self.start_date = (self.end_date - timedelta(days=30)).strftime("%Y-%m-%d")
            elif timeframe == "3m":
                self.start_date = (self.end_date - timedelta(days=90)).strftime("%Y-%m-%d")
            elif timeframe == "6m":
                self.start_date = (self.end_date - timedelta(days=180)).strftime("%Y-%m-%d")
            elif timeframe == "1y":
                self.start_date = (self.end_date - timedelta(days=365)).strftime("%Y-%m-%d")
            elif timeframe == "2y":
                self.start_date = (self.end_date - timedelta(days=730)).strftime("%Y-%m-%d")
            elif timeframe == "5y":
                self.start_date = (self.end_date - timedelta(days=1825)).strftime("%Y-%m-%d")
            elif timeframe == "max":
                self.start_date = "2000-01-01"  # Far back enough for most stocks
            else:
                self.start_date = (self.end_date - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Fetch data
        self.fetch_data()
        self.model = None
        self.aligned_data = None
        self.weekly_money_flow = None
        self.segment_data = None
        self.segment_residuals = None
        self.segment_models = {}
        self.correlation_matrix = None
        self.historical_correlations = None
        
    def fetch_data(self):
        """Fetch historical stock and market data."""
        # Get the historical stock data
        stock_data = yf.Ticker(self.ticker_symbol)
        self.historical_data = stock_data.history(start=self.start_date)
        
        # Get the historical market benchmark data
        market_data = yf.Ticker(self.market_symbol)
        self.historical_market_data = market_data.history(start=self.start_date)
        
        # Extract 'Close' prices and 'Volume'
        self.stock_close = self.historical_data['Close']
        self.market_close = self.historical_market_data['Close']
        self.stock_volume = self.historical_data['Volume']
        
        # Compute the money flow (Volume × Price)
        self.money_flow = self.stock_close * self.stock_volume
        
        # Convert index to date-only format
        self.stock_close.index = self.stock_close.index.date
        self.market_close.index = self.market_close.index.date
        self.money_flow.index = self.money_flow.index.date
        
        # Fetch segment tickers data if provided
        if self.segment_tickers:
            self.segment_data = {}
            for ticker in self.segment_tickers:
                try:
                    ticker_data = yf.Ticker(ticker)
                    hist_data = ticker_data.history(start=self.start_date)
                    close_prices = hist_data['Close']
                    close_prices.index = close_prices.index.date
                    self.segment_data[ticker] = close_prices
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {str(e)}")
                    # Continue with other tickers even if one fails
        
    def prepare_data(self):
        """Align and prepare data for analysis."""
        # Align data
        self.aligned_data = pd.concat([self.stock_close, self.market_close, self.money_flow], 
                                      axis=1, join='inner')
        self.aligned_data.columns = [self.ticker_symbol, 'Market', 'MoneyFlow']
        self.aligned_data.index = pd.to_datetime(self.aligned_data.index)
        
        # Prepare segment data if available
        if self.segment_tickers and self.segment_data:
            # Create a DataFrame with all segment tickers plus the main ticker
            all_tickers = self.segment_tickers.copy()
            if self.ticker_symbol not in all_tickers:
                all_tickers.append(self.ticker_symbol)
            
            # Collect all ticker data
            ticker_data_dict = {ticker: self.segment_data.get(ticker, pd.Series()) for ticker in self.segment_tickers}
            ticker_data_dict[self.ticker_symbol] = self.stock_close
            
            # Align all ticker data
            segment_df = pd.DataFrame(ticker_data_dict)
            segment_df.index = pd.to_datetime(segment_df.index)
            
            # Ensure we have the market data aligned with segment data
            market_series = pd.Series(self.market_close, index=pd.to_datetime(self.market_close.index))
            segment_df['Market'] = market_series
            
            # Drop any rows with missing values
            self.segment_data_aligned = segment_df.dropna()
        
        # Resample money flow to weekly
        self.weekly_money_flow = self.aligned_data['MoneyFlow'].resample('W', label='right').max()
        
    def run_regression(self):
        """Perform linear regression analysis."""
        if self.aligned_data is None:
            self.prepare_data()
            
        if self.aligned_data.shape[0] > 0:
            # Perform linear regression for the main ticker
            X = sm.add_constant(self.aligned_data['Market'])  # Add constant for intercept
            y = self.aligned_data[self.ticker_symbol]
            self.model = sm.OLS(y, X).fit()
            
            # Calculate residuals
            self.aligned_data['Residuals'] = self.model.resid
            
            # Calculate weekly total volume
            self.weekly_volume = self.aligned_data['MoneyFlow'].resample('W', label='right').sum()
            
            # Run regression for segment tickers if available
            if hasattr(self, 'segment_data_aligned') and self.segment_data_aligned is not None:
                self.segment_residuals = pd.DataFrame(index=self.segment_data_aligned.index)
                
                # Run regression for each ticker in the segment
                for ticker in self.segment_data_aligned.columns:
                    if ticker != 'Market':  # Skip the market column
                        X_segment = sm.add_constant(self.segment_data_aligned['Market'])
                        y_segment = self.segment_data_aligned[ticker]
                        try:
                            model = sm.OLS(y_segment, X_segment).fit()
                            self.segment_models[ticker] = model
                            self.segment_residuals[ticker] = model.resid
                        except Exception as e:
                            print(f"Error in regression for {ticker}: {str(e)}")
                
                # Calculate correlation matrix of residuals
                if not self.segment_residuals.empty:
                    self.correlation_matrix = self.segment_residuals.corr()
                    
                    # Calculate rolling correlations (30-day window by default)
                    self.calculate_rolling_correlations()
            
            return True
        else:
            print("No aligned data available for regression analysis.")
            return False
            
    def calculate_rolling_correlations(self, window=30):
        """Calculate rolling correlations between residuals of segment tickers."""
        if self.segment_residuals is None or self.segment_residuals.empty:
            return
            
        # Initialize a dictionary to store historical correlations for each ticker
        self.historical_correlations = {}
        
        # For each ticker, calculate its average correlation with all other tickers
        for ticker in self.segment_residuals.columns:
            # Calculate rolling correlation with each other ticker
            rolling_corrs = []
            for other_ticker in self.segment_residuals.columns:
                if ticker != other_ticker:
                    # Calculate rolling correlation between this ticker and other ticker
                    rolling_corr = self.segment_residuals[ticker].rolling(window=window).corr(
                        self.segment_residuals[other_ticker])
                    rolling_corrs.append(rolling_corr)
            
            # Average the correlations if we have any
            if rolling_corrs:
                avg_rolling_corr = pd.concat(rolling_corrs, axis=1).mean(axis=1)
                self.historical_correlations[ticker] = avg_rolling_corr
            
    def get_segment_correlation_signals(self, lookback_period=30, correlation_threshold=0.3):
        """
        Generate buy/sell signals based on correlation changes in segment residuals.
        
        Args:
            lookback_period (int): Number of days to look back for context
            correlation_threshold (float): Threshold for correlation change to generate signals
            
        Returns:
            dict: Dictionary containing correlation signals for each ticker
        """
        if not hasattr(self, 'historical_correlations') or not self.historical_correlations:
            return {}
            
        signals = {}
        
        for ticker, corr_series in self.historical_correlations.items():
            if len(corr_series) < lookback_period + 5:  # Need enough data
                continue
                
            # Get recent correlation values
            recent_corrs = corr_series.dropna().iloc[-lookback_period:]
            if len(recent_corrs) < 5:  # Need at least some recent data
                continue
                
            # Calculate average correlation over lookback period
            avg_corr = recent_corrs.iloc[:-5].mean()  # Average excluding most recent 5 days
            latest_corr = recent_corrs.iloc[-1]  # Most recent correlation
            
            # Calculate correlation change
            corr_change = latest_corr - avg_corr
            
            # Get the latest price and residual
            latest_price = self.segment_data_aligned[ticker].iloc[-1] if ticker in self.segment_data_aligned else None
            latest_residual = self.segment_residuals[ticker].iloc[-1] if ticker in self.segment_residuals else None
            
            if latest_price is None or latest_residual is None:
                continue
                
            # Determine signal based on correlation change and residual
            signal = "HOLD"
            reason = "No significant correlation change detected"
            
            if corr_change < -correlation_threshold:  # Correlation decreased significantly
                if latest_residual < 0:  # Price is below expected (hasn't risen yet)
                    signal = "STRONG BUY"
                    reason = f"Correlation decreased ({corr_change:.2f}) and price is below expected level"
                else:  # Price is above expected
                    signal = "SELL"
                    reason = f"Correlation decreased ({corr_change:.2f}) but price is above expected level"
            elif corr_change > correlation_threshold:  # Correlation increased significantly
                if latest_residual > 0:  # Price is above expected
                    signal = "STRONG SELL"
                    reason = f"Correlation increased ({corr_change:.2f}) and price is above expected level"
                else:  # Price is below expected
                    signal = "BUY"
                    reason = f"Correlation increased ({corr_change:.2f}) but price is below expected level"
            
            # Store signal information
            signals[ticker] = {
                "signal": signal,
                "reason": reason,
                "latest_correlation": latest_corr,
                "avg_correlation": avg_corr,
                "correlation_change": corr_change,
                "latest_price": latest_price,
                "latest_residual": latest_residual
            }
            
        return signals
    
    def get_buy_sell_signal(self, residual_threshold=0.0, lookback_period=30):
        """
        Generate buy/sell signals based on residual analysis.
        
        Args:
            residual_threshold (float): Threshold for residual to generate signals
            lookback_period (int): Number of days to look back for context
            
        Returns:
            dict: Dictionary containing signal information
        """
        if self.aligned_data is None or 'Residuals' not in self.aligned_data.columns:
            if not self.run_regression():
                return {"signal": "UNKNOWN", "reason": "Insufficient data for analysis"}
        
        # Get the latest residual
        latest_residual = self.aligned_data['Residuals'].iloc[-1]
        
        # Get historical context (recent residuals)
        recent_residuals = self.aligned_data['Residuals'].iloc[-lookback_period:]
        avg_residual = recent_residuals.mean()
        std_residual = recent_residuals.std()
        
        # Z-score of latest residual compared to recent history
        z_score = (latest_residual - avg_residual) / std_residual if std_residual > 0 else 0
        
        # Determine signal
        if latest_residual < residual_threshold:
            if z_score < -1.0:  # Significantly below recent average
                signal = "STRONG BUY"
            else:
                signal = "BUY"
            reason = f"Residual ({latest_residual:.2f}) is below threshold ({residual_threshold:.2f})"
        elif latest_residual > abs(residual_threshold):
            if z_score > 1.0:  # Significantly above recent average
                signal = "STRONG SELL"
            else:
                signal = "SELL"
            reason = f"Residual ({latest_residual:.2f}) is above threshold ({abs(residual_threshold):.2f})"
        else:
            signal = "HOLD"
            reason = f"Residual ({latest_residual:.2f}) is within normal range"
        
        # Additional context
        context = {
            "latest_date": self.aligned_data.index[-1].strftime("%Y-%m-%d"),
            "latest_price": self.aligned_data[self.ticker_symbol].iloc[-1],
            "latest_residual": latest_residual,
            "avg_residual": avg_residual,
            "std_residual": std_residual,
            "z_score": z_score,
            "lookback_period": lookback_period
        }
        
        return {
            "signal": signal,
            "reason": reason,
            "context": context
        }
    
    def plot_price_and_volume(self):
        """Plot stock price movement with weekly volume."""
        if self.aligned_data is None or self.weekly_volume is None:
            if not self.run_regression():
                return
                
        fig, ax1 = plt.subplots(figsize=(12, 4))
        
        # Line plot for stock price
        ax1.plot(self.aligned_data.index, self.aligned_data[self.ticker_symbol], 
                 label=f'{self.ticker_symbol} Price', color='tab:green')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)', color='tab:green')
        ax1.tick_params(axis='y', labelcolor='tab:green')
        ax1.set_title(f'Stock Price Movement & Weekly Volume of {self.ticker_symbol}')
        ax1.grid(True)
        
        # Secondary y-axis for weekly total volume
        ax2 = ax1.twinx()
        ax2.bar(self.weekly_volume.index, self.weekly_volume, 
                label='Weekly Volume', color='tab:gray', alpha=0.5, width=5)
        ax2.set_ylabel('Weekly Volume', color='tab:gray')
        ax2.tick_params(axis='y', labelcolor='tab:gray')
        
        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Create directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{self.ticker_symbol}_price_volume.png')
        plt.close()
        
    def plot_residuals_and_money_flow(self):
        """Plot residuals and weekly money flow."""
        if self.aligned_data is None or 'Residuals' not in self.aligned_data.columns:
            if not self.run_regression():
                return
                
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Line plot for daily residuals
        ax1.plot(self.aligned_data.index, self.aligned_data['Residuals'], 
                 label='Residuals', color='tab:blue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Residuals', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_title(f'Residual Price Movements & Weekly Money Flow for {self.ticker_symbol}')
        
        # Add horizontal line at y=0
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Secondary y-axis for weekly money flow
        ax2 = ax1.twinx()
        ax2.bar(self.weekly_money_flow.index, self.weekly_money_flow, 
                label='Weekly Money Flow (Volume × Price)', color='tab:gray', alpha=0.5, width=5)
        ax2.set_ylabel('Weekly Money Flow ($)', color='tab:gray')
        ax2.tick_params(axis='y', labelcolor='tab:gray')
        
        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Grid and layout
        ax1.grid(True)
        fig.tight_layout()
        
        # Create directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{self.ticker_symbol}_residuals.png')
        plt.close()
        
    def print_summary(self):
        """Print summary of the regression analysis and latest stock information."""
        if self.model is None:
            if not self.run_regression():
                return
                
        print(f"\n{'='*80}")
        print(f"Stock Analysis: {self.ticker_symbol} vs {self.market_symbol}")
        print(f"{'='*80}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"\nRegression Summary:")
        print(self.model.summary())
        
        # Get latest information
        latest_date = self.aligned_data.index[-1]
        latest_price = self.aligned_data[self.ticker_symbol].iloc[-1]
        latest_residual = self.aligned_data['Residuals'].iloc[-1]
        
        print(f"\nLatest Data:")
        print(f"Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"Price: ${latest_price:.2f}")
        print(f"Residual: {latest_residual:.4f}")
        
        # Get buy/sell signal
        signal_info = self.get_buy_sell_signal()
        print(f"\nSignal: {signal_info['signal']}")
        print(f"Reason: {signal_info['reason']}")
        print(f"\nContext:")
        for key, value in signal_info['context'].items():
            if key != 'latest_date':  # Already printed above
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\nInterpretation:")
        print("- Negative residuals suggest the stock is undervalued relative to the market")
        print("- Positive residuals suggest the stock is overvalued relative to the market")
        print("- The strength of the signal depends on how far the residual is from its recent average")
        print(f"{'='*80}")


def main():
    """Main function to run the stock analyzer from command line."""
    parser = argparse.ArgumentParser(description='Stock Analyzer Tool')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--market', type=str, default='URTH', 
                        help='Market benchmark ticker (default: URTH - MSCI World ETF)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='1y',
                        choices=['1m', '3m', '6m', '1y', '2y', '5y', 'max'],
                        help='Analysis timeframe (default: 1y)')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Residual threshold for buy/sell signals (default: 0.0)')
    parser.add_argument('--lookback', type=int, default=30,
                        help='Lookback period in days for signal context (default: 30)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = StockAnalyzer(
        ticker_symbol=args.ticker,
        market_symbol=args.market,
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe
    )
    
    # Run analysis
    analyzer.prepare_data()
    analyzer.run_regression()
    
    # Generate plots
    if not args.no_plots:
        analyzer.plot_price_and_volume()
        analyzer.plot_residuals_and_money_flow()
    
    # Print summary
    analyzer.print_summary()
    
    # Get buy/sell signal with custom parameters
    signal_info = analyzer.get_buy_sell_signal(
        residual_threshold=args.threshold,
        lookback_period=args.lookback
    )
    
    return signal_info['signal']


if __name__ == "__main__":
    main()
