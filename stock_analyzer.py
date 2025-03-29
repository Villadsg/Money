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

# Try to import RL components
try:
    from rl_agent import train_rl_agent, evaluate_rl_agent
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


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
        
        # RL agent components
        self.rl_agent = None
        self.rl_training_history = None
        self.rl_evaluation_results = None
        
    def fetch_data(self):
        """Fetch historical stock and market data."""
        try:
            # Get the historical stock data
            print(f"DEBUG: Fetching data for {self.ticker_symbol} from {self.start_date}")
            stock_data = yf.Ticker(self.ticker_symbol)
            self.historical_data = stock_data.history(start=self.start_date)
            
            if self.historical_data.empty:
                print(f"ERROR: No data available for {self.ticker_symbol}")
                return False
                
            # Get the historical market benchmark data
            market_data = yf.Ticker(self.market_symbol)
            self.historical_market_data = market_data.history(start=self.start_date)
            
            if self.historical_market_data.empty:
                print(f"ERROR: No data available for market symbol {self.market_symbol}")
                return False
                
            # Extract 'Close' prices and 'Volume'
            self.stock_close = self.historical_data['Close']
            self.market_close = self.historical_market_data['Close']
            self.stock_volume = self.historical_data['Volume']
            
            # Compute the money flow (Volume × Price)
            self.money_flow = self.stock_close * self.stock_volume
            
            # Convert index to datetime format for consistent handling
            self.stock_close.index = pd.to_datetime(self.stock_close.index.date)
            self.market_close.index = pd.to_datetime(self.market_close.index.date)
            self.money_flow.index = pd.to_datetime(self.money_flow.index.date)
            
            # Fetch segment tickers data if provided
            if self.segment_tickers and len(self.segment_tickers) > 0:
                print(f"DEBUG: Fetching data for {len(self.segment_tickers)} segment tickers")
                self.segment_data = {}
                successful_tickers = 0
                
                for ticker in self.segment_tickers:
                    try:
                        ticker_data = yf.Ticker(ticker)
                        hist_data = ticker_data.history(start=self.start_date)
                        
                        if hist_data.empty:
                            print(f"WARNING: No data available for segment ticker {ticker}")
                            continue
                            
                        close_prices = hist_data['Close']
                        close_prices.index = pd.to_datetime(close_prices.index.date)
                        self.segment_data[ticker] = close_prices
                        successful_tickers += 1
                    except Exception as e:
                        print(f"ERROR: Failed to fetch data for {ticker}: {str(e)}")
                
                print(f"DEBUG: Successfully fetched data for {successful_tickers} out of {len(self.segment_tickers)} segment tickers")
            else:
                print("DEBUG: No segment tickers specified")
                
            return True
        except Exception as e:
            print(f"ERROR: Failed to fetch data: {str(e)}")
            return False
        
    def prepare_data(self):
        """Align and prepare data for analysis."""
        # Align data for main ticker
        self.aligned_data = pd.concat([self.stock_close, self.market_close, self.money_flow], 
                                      axis=1, join='inner')
        self.aligned_data.columns = [self.ticker_symbol, 'Market', 'MoneyFlow']
        self.aligned_data.index = pd.to_datetime(self.aligned_data.index)
        
        # Initialize empty segment data containers to avoid None issues
        self.segment_data_aligned = pd.DataFrame()
        self.segment_residuals = pd.DataFrame()
        self.segment_models = {}
        self.historical_correlations = {}
        
        # Prepare segment data if available
        if self.segment_tickers and len(self.segment_tickers) > 0:
            print(f"DEBUG: Preparing segment data for {len(self.segment_tickers)} tickers")
            
            # If segment_data is empty or doesn't exist, try to fetch it
            if not hasattr(self, 'segment_data') or not self.segment_data:
                print("DEBUG: segment_data not found, attempting to fetch it now")
                self.segment_data = {}
                for ticker in self.segment_tickers:
                    try:
                        ticker_data = yf.Ticker(ticker)
                        hist_data = ticker_data.history(start=self.start_date)
                        close_prices = hist_data['Close']
                        close_prices.index = pd.to_datetime(close_prices.index.date)
                        self.segment_data[ticker] = close_prices
                        print(f"DEBUG: Successfully fetched {ticker} data, {len(close_prices)} rows")
                    except Exception as e:
                        print(f"ERROR: Failed to fetch {ticker} data: {str(e)}")
            
            # Proceed only if we have valid segment data
            if self.segment_data and len(self.segment_data) > 0:
                print(f"DEBUG: Building segment dataframe with {len(self.segment_data)} tickers")
                
                # Create a DataFrame with all segment tickers
                ticker_data_dict = {}
                for ticker in self.segment_tickers:
                    if ticker in self.segment_data and not self.segment_data[ticker].empty:
                        ticker_data_dict[ticker] = self.segment_data[ticker]
                
                # Add main ticker if it's not already included
                if self.ticker_symbol not in ticker_data_dict:
                    ticker_data_dict[self.ticker_symbol] = self.stock_close
                
                # Proceed only if we have at least one valid ticker
                if ticker_data_dict:
                    # Create and align dataframe
                    segment_df = pd.DataFrame(ticker_data_dict)
                    segment_df.index = pd.to_datetime(segment_df.index)
                    
                    # Add market data
                    market_series = pd.Series(self.market_close, name='Market')
                    market_series.index = pd.to_datetime(market_series.index)
                    segment_df['Market'] = market_series
                    
                    # Handle missing values more carefully - use a threshold
                    # Only drop rows where ALL ticker data is missing (keep rows where at least some data exists)
                    segment_df = segment_df.dropna(how='all')
                    
                    # For remaining NaN values, use forward-fill then backward-fill
                    segment_df = segment_df.ffill().bfill()
                    
                    # Store the aligned data
                    self.segment_data_aligned = segment_df
                    print(f"DEBUG: Segment data prepared successfully, shape: {segment_df.shape}")
                else:
                    print("DEBUG: No valid ticker data available for segment analysis")
            else:
                print("DEBUG: No segment data available for processing")
        else:
            print("DEBUG: No segment tickers provided")
        
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
            if hasattr(self, 'segment_data_aligned') and self.segment_data_aligned is not None and len(self.segment_tickers) > 0:
                print(f"DEBUG: Running regression for segment tickers, data shape: {self.segment_data_aligned.shape}")
                
                if self.segment_data_aligned.shape[0] < 30:
                    print("DEBUG: Warning: Limited data points for segment analysis")
                
                # Initialize segment_residuals with the primary ticker's data first
                self.segment_residuals = pd.DataFrame(index=self.segment_data_aligned.index)
                
                # Make sure to include the main ticker in segment analysis if not already present
                segment_columns = list(self.segment_data_aligned.columns)
                if self.ticker_symbol not in segment_columns and self.ticker_symbol in self.aligned_data.columns:
                    # Check if dates are aligned and copy the main ticker data to segment_data_aligned
                    common_dates = self.aligned_data.index.intersection(self.segment_data_aligned.index)
                    if len(common_dates) > 0:
                        self.segment_data_aligned[self.ticker_symbol] = self.aligned_data.loc[common_dates, self.ticker_symbol]
                
                # Run regression for each ticker in the segment
                tickers_processed = []
                for ticker in self.segment_data_aligned.columns:
                    if ticker != 'Market':  # Skip the market column
                        X_segment = sm.add_constant(self.segment_data_aligned['Market'])
                        y_segment = self.segment_data_aligned[ticker]
                        
                        # Skip tickers with insufficient data
                        if y_segment.isna().sum() > len(y_segment) * 0.3:  # If more than 30% NaN
                            print(f"DEBUG: Skipping {ticker} - too many NaN values")
                            continue
                            
                        try:
                            model = sm.OLS(y_segment, X_segment).fit()
                            self.segment_models[ticker] = model
                            self.segment_residuals[ticker] = model.resid
                            tickers_processed.append(ticker)
                            print(f"DEBUG: Regression successful for {ticker}, residuals shape: {self.segment_residuals[ticker].shape}")
                        except Exception as e:
                            print(f"ERROR: Regression failed for {ticker}: {str(e)}")
                
                print(f"DEBUG: Successfully processed {len(tickers_processed)} segment tickers")
                
                # Calculate correlation matrix of residuals
                if not self.segment_residuals.empty and len(self.segment_residuals.columns) >= 2:
                    print(f"DEBUG: Calculating correlation matrix for {len(self.segment_residuals.columns)} tickers")
                    self.correlation_matrix = self.segment_residuals.corr()
                    
                    # Calculate rolling correlations (adjust window size if needed)
                    window_size = min(30, self.segment_residuals.shape[0] // 3)
                    if window_size < 5:
                        window_size = 5
                    print(f"DEBUG: Using window size of {window_size} for rolling correlations")
                    self.calculate_rolling_correlations(window=window_size)
                else:
                    print("DEBUG: segment_residuals is empty or has fewer than 2 tickers after regressions")
                    # Initialize empty historical_correlations to avoid errors
                    self.historical_correlations = {}
            else:
                print("DEBUG: No segment data available for analysis")
                # Initialize empty containers to avoid NoneType errors
                self.segment_residuals = pd.DataFrame()
                self.historical_correlations = {}
            
            return True
        else:
            print("ERROR: No aligned data available for regression analysis.")
            return False
            
    def calculate_rolling_correlations(self, window=30):
        """Calculate rolling correlations between residuals of segment tickers."""
        if self.segment_residuals is None or self.segment_residuals.empty:
            print("DEBUG: Cannot calculate rolling correlations - segment_residuals is None or empty")
            return
        
        # Ensure we have at least 2 tickers to calculate correlations
        if len(self.segment_residuals.columns) < 2:
            print(f"DEBUG: Need at least 2 tickers for correlation, but only have {len(self.segment_residuals.columns)}")
            return
            
        print(f"DEBUG: Calculating rolling correlations for {len(self.segment_residuals.columns)} tickers with window={window}")
        
        # Make sure window size isn't larger than the data we have
        if window >= len(self.segment_residuals):
            window = max(5, len(self.segment_residuals) // 2)  # Use half the data points or at least 5
            print(f"DEBUG: Adjusted window size to {window} due to limited data")
            
        # Initialize a dictionary to store historical correlations for each ticker
        self.historical_correlations = {}
        
        # For each ticker, calculate its average correlation with all other tickers
        for ticker in self.segment_residuals.columns:
            # Calculate rolling correlation with each other ticker
            rolling_corrs = []
            
            # Handle NaN values in the residuals
            ticker_data = self.segment_residuals[ticker].fillna(method='ffill').fillna(method='bfill')
            
            for other_ticker in self.segment_residuals.columns:
                if ticker != other_ticker:
                    # Calculate rolling correlation between this ticker and other ticker
                    try:
                        # Handle NaN values in other ticker's data as well
                        other_ticker_data = self.segment_residuals[other_ticker].fillna(method='ffill').fillna(method='bfill')
                        
                        # Only calculate if both series have sufficient non-NaN values
                        combined_df = pd.DataFrame({
                            'ticker': ticker_data,
                            'other': other_ticker_data
                        }).dropna()
                        
                        if len(combined_df) > window:  # Ensure we have enough data after dropping NaNs
                            # Calculate rolling correlation using the cleaned data
                            rolling_corr = combined_df['ticker'].rolling(window=window).corr(combined_df['other'])
                            rolling_corrs.append(rolling_corr)
                            print(f"DEBUG: Calculated rolling correlation between {ticker} and {other_ticker}")
                        else:
                            print(f"DEBUG: Not enough clean data for correlation between {ticker} and {other_ticker}")
                    except Exception as e:
                        print(f"ERROR: Failed to calculate rolling correlation for {ticker} and {other_ticker}: {str(e)}")
            
            # Average the correlations if we have any
            if rolling_corrs and len(rolling_corrs) > 0:
                try:
                    # Create a DataFrame from all rolling correlations and handle missing values
                    corr_df = pd.concat(rolling_corrs, axis=1)
                    
                    # Average correlations across all other tickers, ignoring NaN values
                    avg_rolling_corr = corr_df.mean(axis=1, skipna=True)
                    
                    # Fill any remaining NaN values with the nearest valid observation
                    avg_rolling_corr = avg_rolling_corr.fillna(method='ffill').fillna(method='bfill')
                    
                    # Store the cleaned average correlation
                    self.historical_correlations[ticker] = avg_rolling_corr
                    print(f"DEBUG: Added {len(avg_rolling_corr)} historical correlations for {ticker}")
                except Exception as e:
                    print(f"ERROR: Failed to average correlations for {ticker}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            else:
                print(f"DEBUG: No valid rolling correlations calculated for {ticker}")
            
    def analyze_residual_divergence(self, lookback_period=30, divergence_threshold=1.5):
        """
        Analyze how each stock's residuals diverge from the segment average pattern.
        Identifies stocks that suddenly move differently than their segment peers.
        
        Args:
            lookback_period (int): Number of days to look back for context
            divergence_threshold (float): Z-score threshold to consider a divergence significant
            
        Returns:
            dict: Dictionary containing divergence signals for each ticker
        """
        if not hasattr(self, 'segment_residuals') or self.segment_residuals is None or self.segment_residuals.empty:
            print("DEBUG: Cannot analyze residual divergence - segment_residuals is None or empty")
            # Try to fix the issue by running regression again if possible
            if hasattr(self, 'segment_data_aligned') and self.segment_data_aligned is not None and 'Market' in self.segment_data_aligned.columns:
                print("DEBUG: Attempting to regenerate segment residuals")
                
                # Create an empty DataFrame for segment residuals
                self.segment_residuals = pd.DataFrame(index=self.segment_data_aligned.index)
                
                # Process each ticker
                tickers_processed = []
                for ticker in self.segment_data_aligned.columns:
                    if ticker != 'Market':
                        X_segment = sm.add_constant(self.segment_data_aligned['Market'])
                        y_segment = self.segment_data_aligned[ticker]
                        
                        # Skip tickers with insufficient data
                        if y_segment.isna().sum() > len(y_segment) * 0.3:  # If more than 30% NaN
                            continue
                            
                        try:
                            model = sm.OLS(y_segment, X_segment).fit()
                            self.segment_residuals[ticker] = model.resid
                            tickers_processed.append(ticker)
                        except Exception:
                            pass
                
                if len(tickers_processed) < 2:
                    print(f"DEBUG: Still not enough valid tickers for divergence analysis: {len(tickers_processed)}")
                    return {}
            else:
                return {}
        
        # Ensure we have enough tickers for a meaningful segment average
        if len(self.segment_residuals.columns) < 2:
            print(f"DEBUG: Need at least 2 tickers for divergence analysis, but only have {len(self.segment_residuals.columns)}")
            return {}
            
        print(f"DEBUG: Analyzing residual divergence for {len(self.segment_residuals.columns)} tickers")
        
        # Calculate segment average residual pattern
        # First handle NaN values by forward/backward filling
        filled_residuals = self.segment_residuals.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate the median residual across all tickers (more robust to outliers than mean)
        segment_median_residual = filled_residuals.median(axis=1)
        
        # Initialize dictionary to store results
        divergence_signals = {}
        
        # Analyze each ticker
        for ticker in filled_residuals.columns:
            # Get the ticker's residuals
            ticker_residuals = filled_residuals[ticker]
            
            # Calculate how much this ticker deviates from the segment median
            deviation_from_segment = ticker_residuals - segment_median_residual
            
            # Recent deviation (absolute values since we care about magnitude of difference, not direction)
            recent_deviations = deviation_from_segment.abs()
            
            # Use rolling window to calculate the average recent deviation
            deviation_z_scores = pd.Series(index=recent_deviations.index)
            
            # Calculate z-scores of recent deviations using rolling windows
            window_size = min(lookback_period, len(recent_deviations) - 5)  # Ensure we have enough data
            if window_size < 5:
                window_size = 5
                
            for i in range(window_size, len(recent_deviations)):
                # Get historical window excluding most recent points
                historical_window = recent_deviations.iloc[i-window_size:i-5]
                
                # Get the most recent deviation
                recent_value = recent_deviations.iloc[i]
                
                # Calculate z-score
                if not historical_window.empty:
                    window_mean = historical_window.mean()
                    window_std = historical_window.std()
                    
                    if window_std > 0:
                        z_score = (recent_value - window_mean) / window_std
                        deviation_z_scores.iloc[i] = z_score
            
            # Get the most recent z-score
            if not deviation_z_scores.dropna().empty:
                latest_z_score = deviation_z_scores.dropna().iloc[-1]
                latest_date = deviation_z_scores.dropna().index[-1]
                latest_residual = ticker_residuals.loc[latest_date]
                latest_deviation = deviation_from_segment.loc[latest_date]
                
                # Get latest price if available
                latest_price = None
                try:
                    if hasattr(self, 'segment_data_aligned') and ticker in self.segment_data_aligned.columns:
                        latest_price = self.segment_data_aligned[ticker].iloc[-1]
                except Exception as e:
                    print(f"DEBUG: Could not get latest price for {ticker}: {e}")
                
                # Determine signal based on divergence z-score and residual
                signal = "HOLD"
                reason = "No significant divergence from segment pattern"
                
                if latest_z_score > divergence_threshold:
                    if latest_residual < 0:
                        signal = "BUY"
                        reason = f"Diverging significantly from segment (z={latest_z_score:.2f}) with price below trend"
                    else:
                        signal = "SELL"
                        reason = f"Diverging significantly from segment (z={latest_z_score:.2f}) with price above trend"
                        
                    # If divergence is very strong, strengthen the signal
                    if latest_z_score > 2 * divergence_threshold:
                        signal = "STRONG " + signal
                
                # Store the signal and related information
                divergence_signals[ticker] = {
                    "signal": signal,
                    "reason": reason,
                    "latest_z_score": latest_z_score,
                    "latest_residual": latest_residual,
                    "latest_deviation": latest_deviation,
                    "latest_price": latest_price,
                    "latest_date": latest_date
                }
            else:
                print(f"DEBUG: No valid z-scores calculated for {ticker}")
        
        print(f"DEBUG: Generated {len(divergence_signals)} divergence signals")
        return divergence_signals
    
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
            print("DEBUG: No historical_correlations available for signal generation")
            return {}
        
        # If historical_correlations is empty, try calculating them again with more lenient parameters
        if len(self.historical_correlations) == 0 and hasattr(self, 'segment_residuals') and not self.segment_residuals.empty:
            print("DEBUG: Attempting to recalculate correlations with adjusted parameters")
            window_size = min(10, max(5, self.segment_residuals.shape[0] // 5))  # Use an even smaller window
            self.calculate_rolling_correlations(window=window_size)
            
            # If still empty, return empty results
            if len(self.historical_correlations) == 0:
                print("DEBUG: Still couldn't generate correlations with adjusted parameters")
                return {}
            
        signals = {}
        print(f"DEBUG: Checking correlation signals for {len(self.historical_correlations)} tickers")
        
        # Adjust lookback period if we don't have enough data
        min_data_length = 0
        for ticker, corr_series in self.historical_correlations.items():
            non_na_length = corr_series.dropna().shape[0]
            if non_na_length > min_data_length:
                min_data_length = non_na_length
        
        # Adjust lookback period if needed
        adjusted_lookback = min(lookback_period, max(5, min_data_length-5))
        if adjusted_lookback < lookback_period:
            print(f"DEBUG: Adjusted lookback period from {lookback_period} to {adjusted_lookback} due to limited data")
        lookback_period = adjusted_lookback
        
        for ticker, corr_series in self.historical_correlations.items():
            print(f"DEBUG: Processing {ticker}, correlation series length: {len(corr_series)}, non-NA values: {corr_series.dropna().shape[0]}")
            
            # Get recent correlation values (handle case of limited data)
            recent_corrs = corr_series.dropna()
            
            if len(recent_corrs) < 5:  # Need at least some recent data
                print(f"DEBUG: Not enough recent data for {ticker}, only {len(recent_corrs)} points")
                continue
                
            # Calculate average correlation over lookback period (handle small datasets)
            split_point = max(1, len(recent_corrs) - 5)  # Keep at least last 5 days separate
            avg_corr = recent_corrs.iloc[:split_point].mean()  # Average excluding most recent 5 days
            latest_corr = recent_corrs.iloc[-1]  # Most recent correlation
            
            # Calculate correlation change
            corr_change = latest_corr - avg_corr
            print(f"DEBUG: {ticker} - avg_corr: {avg_corr:.4f}, latest_corr: {latest_corr:.4f}, change: {corr_change:.4f}")
            
            # Get the latest price and residual
            latest_price = None
            latest_residual = None
            
            try:
                if ticker in self.segment_data_aligned.columns:
                    latest_price = self.segment_data_aligned[ticker].iloc[-1]
                    print(f"DEBUG: {ticker} latest price: {latest_price:.2f}")
                else:
                    print(f"DEBUG: {ticker} not found in segment_data_aligned")
                    
                if ticker in self.segment_residuals.columns:
                    latest_residual = self.segment_residuals[ticker].iloc[-1]
                    print(f"DEBUG: {ticker} latest residual: {latest_residual:.4f}")
                else:
                    print(f"DEBUG: {ticker} not found in segment_residuals")
            except Exception as e:
                print(f"ERROR: Could not get latest price/residual for {ticker}: {str(e)}")
                continue
            
            if latest_price is None or latest_residual is None:
                print(f"DEBUG: Missing price or residual for {ticker}")
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
            print(f"DEBUG: Generated {signal} signal for {ticker}")
            
        print(f"DEBUG: Generated {len(signals)} correlation signals in total")
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


    def train_rl_model(self, episodes=50, batch_size=32):
        """Train a reinforcement learning agent for stock trading.
        
        Args:
            episodes (int): Number of training episodes
            batch_size (int): Batch size for experience replay
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if not RL_AVAILABLE:
            print("ERROR: RL functionality is not available. Please install the required dependencies.")
            return False
            
        try:
            print(f"Training RL agent for {self.ticker_symbol} with {episodes} episodes...")
            
            # Ensure we have the necessary data
            if self.stock_close is None or self.market_close is None:
                print("ERROR: Stock or market data is missing. Cannot train RL agent.")
                return False
                
            # Train the agent
            self.rl_agent, self.rl_training_history = train_rl_agent(
                self.stock_close, 
                self.market_close,
                episodes=episodes,
                batch_size=batch_size
            )
            
            # Evaluate the trained agent
            self.rl_evaluation_results = evaluate_rl_agent(
                self.rl_agent,
                self.stock_close,
                self.market_close
            )
            
            print(f"RL agent training completed. Total return: {self.rl_evaluation_results['total_return']:.2f}%")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to train RL agent: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_rl_trading_signal(self):
        """Get the latest trading signal from the trained RL agent.
        
        Returns:
            dict: Dictionary containing signal information
        """
        if not RL_AVAILABLE:
            return {
                'signal': 'UNAVAILABLE',
                'reason': 'RL functionality is not available',
                'context': {}
            }
            
        if self.rl_agent is None or self.rl_evaluation_results is None:
            return {
                'signal': 'UNAVAILABLE',
                'reason': 'RL agent has not been trained yet',
                'context': {}
            }
            
        try:
            # Get the latest decision from the evaluation results
            decisions = self.rl_evaluation_results['decisions']
            latest_decision = decisions.iloc[-1]
            
            # Map action to signal
            action_map = {
                'buy': 'BUY',
                'sell': 'SELL',
                'hold': 'HOLD'
            }
            
            signal = action_map.get(latest_decision['action'], 'UNKNOWN')
            
            # Compare with buy & hold strategy
            rl_return = self.rl_evaluation_results['total_return']
            buy_hold_return = self.rl_evaluation_results['buy_hold_return']
            
            if rl_return > buy_hold_return:
                confidence = "high"
                reason = f"RL agent outperformed buy & hold strategy by {rl_return - buy_hold_return:.2f}%"
            else:
                confidence = "moderate"
                reason = f"RL agent's strategy based on historical price patterns and market benchmark"
            
            # Add strength to signal based on portfolio performance
            if rl_return > buy_hold_return * 1.5:  # Significantly better
                signal = f"STRONG {signal}"
            
            return {
                'signal': signal,
                'reason': reason,
                'confidence': confidence,
                'context': {
                    'latest_date': latest_decision['date'],
                    'latest_price': latest_decision['price'],
                    'portfolio_value': latest_decision['portfolio_value'],
                    'stock_owned': latest_decision['stock_owned'],
                    'cash_balance': latest_decision['cash_balance'],
                    'total_return': rl_return,
                    'buy_hold_return': buy_hold_return,
                    'sharpe_ratio': self.rl_evaluation_results['sharpe_ratio']
                }
            }
        except Exception as e:
            print(f"ERROR: Failed to get RL trading signal: {str(e)}")
            return {
                'signal': 'ERROR',
                'reason': f'Failed to get RL trading signal: {str(e)}',
                'context': {}
            }
    
    def plot_rl_performance(self):
        """Plot the RL agent's performance compared to buy & hold strategy."""
        if not RL_AVAILABLE or self.rl_evaluation_results is None:
            print("ERROR: RL evaluation results not available")
            return
            
        try:
            # Get decisions and portfolio values
            decisions = self.rl_evaluation_results['decisions']
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot portfolio value over time
            ax1.plot(decisions['date'], decisions['portfolio_value'], 'b-', label='RL Agent Portfolio')
            
            # Calculate and plot buy & hold strategy
            initial_price = self.stock_close.iloc[0]
            buy_hold_values = self.stock_close / initial_price * decisions['portfolio_value'].iloc[0]
            ax1.plot(self.stock_close.index, buy_hold_values, 'g--', label='Buy & Hold Strategy')
            
            # Mark buy/sell actions on the chart
            buys = decisions[decisions['action'] == 'buy']
            sells = decisions[decisions['action'] == 'sell']
            
            ax1.scatter(buys['date'], buys['portfolio_value'], color='green', marker='^', s=100, label='Buy')
            ax1.scatter(sells['date'], sells['portfolio_value'], color='red', marker='v', s=100, label='Sell')
            
            ax1.set_title(f'RL Agent Performance for {self.ticker_symbol}')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot stock ownership over time
            ax2.plot(decisions['date'], decisions['stock_owned'], 'r-', label='Stock Owned')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Units Owned')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Create directory for plots if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/{self.ticker_symbol}_rl_performance.png')
            plt.close()
            
        except Exception as e:
            print(f"ERROR: Failed to plot RL performance: {str(e)}")


def main():
    """Main function to run the stock analyzer from command line."""
    parser = argparse.ArgumentParser(description='Stock Analyzer Tool')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--market', type=str, default='QQQ', 
                        help='Market benchmark ticker (default: QQQ - NASDAQ ETF)')
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
    parser.add_argument('--use-rl', action='store_true',
                        help='Use reinforcement learning for trading signals')
    parser.add_argument('--rl-episodes', type=int, default=50,
                        help='Number of episodes for RL training (default: 50)')
    
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
    
    # Train and evaluate RL agent if requested
    if args.use_rl and RL_AVAILABLE:
        print("\nTraining reinforcement learning agent...")
        if analyzer.train_rl_model(episodes=args.rl_episodes):
            rl_signal = analyzer.get_rl_trading_signal()
            print(f"\nRL Agent Signal: {rl_signal['signal']}")
            print(f"Reason: {rl_signal['reason']}")
            print(f"Confidence: {rl_signal['confidence']}")
            
            if 'context' in rl_signal and rl_signal['context']:
                context = rl_signal['context']
                print(f"\nRL Performance:")
                print(f"Total Return: {context.get('total_return', 'N/A'):.2f}%")
                print(f"Buy & Hold Return: {context.get('buy_hold_return', 'N/A'):.2f}%")
                print(f"Sharpe Ratio: {context.get('sharpe_ratio', 'N/A'):.4f}")
            
            # Plot RL performance
            if not args.no_plots:
                analyzer.plot_rl_performance()
        else:
            print("Failed to train RL agent. Using only residual analysis.")
    
    return signal_info['signal']


if __name__ == "__main__":
    main()
