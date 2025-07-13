import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
import glob
warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"

    def _get_csv_filename(self, ticker, start_date, end_date):
        """Generate a standardized CSV filename."""
        return f"data/{ticker}_{start_date}_to_{end_date}.csv"
        
    def save_to_csv(self, df, ticker, start_date, end_date):
        """Save stock data to a CSV file"""
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Delete any existing CSV files for this ticker
        existing_files = glob.glob(f"data/{ticker}_*.csv")
        for file in existing_files:
            try:
                os.remove(file)
                print(f"Removed existing file: {file}")
            except OSError as e:
                print(f"Error removing {file}: {e}")
        
        # Create filename with ticker and date range
        filename = self._get_csv_filename(ticker, start_date, end_date)
        df.to_csv(filename)
        print(f"Data saved to {filename}")
        return filename
    
    def load_from_csv(self, ticker, start_date, end_date):
        """Load stock data from a CSV file if available"""
        filename = self._get_csv_filename(ticker, start_date, end_date)
        if os.path.exists(filename):
            print(f"Loading {ticker} data from {filename}")
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            return df
        return None
        
    def fetch_stock_data(self, ticker, start_date, end_date):
        """Fetch stock data from CSV if available, otherwise from Polygon API"""
        # Try to load from CSV first
        df = self.load_from_csv(ticker, start_date, end_date)
        if df is not None:
            return df
            
        # If CSV not available, fetch from API
        print(f"Fetching {ticker} data from Polygon API...")
        url = f"{self.base_url}/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={'c': 'close', 'o': 'open', 'h': 'high', 'l': 'low', 'v': 'volume'})
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].set_index('date')
                    df = df.sort_index()
                    
                    # Save to CSV for future use
                    self.save_to_csv(df, ticker, start_date, end_date)
                    
                    return df
                else:
                    raise Exception(f"No data returned for {ticker}")
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Error fetching data from API: {str(e)}")
            print("Checking for older data files...")
            
            # Try to find any CSV file for this ticker
            # Note: This glob pattern is intentionally different from _get_csv_filename
            # as it searches for any date range for the given ticker.
            csv_files = glob.glob(f"data/{ticker}_*.csv")
            if csv_files:
                # Use the most recent file
                latest_file = max(csv_files, key=os.path.getctime)
                print(f"Using most recent data file: {latest_file}")
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                return df
            else:
                raise Exception(f"No data available for {ticker}")
    
    def calculate_returns(self, df):
        """Calculate daily returns"""
        df['return'] = df['close'].pct_change()
        return df
    
    def filter_market_movements(self, stock_df, market_df):
        """Filter out movements explained by market index using linear regression"""
        # Align dates
        combined = pd.merge(stock_df, market_df, left_index=True, right_index=True, suffixes=('_stock', '_market'))
        combined = combined.dropna()
        
        # Calculate returns
        combined['stock_return'] = combined['close_stock'].pct_change()
        combined['market_return'] = combined['close_market'].pct_change()
        combined = combined.dropna()
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined['market_return'], combined['stock_return']
        )
        
        # Calculate residuals (stock-specific movements)
        combined['predicted_return'] = slope * combined['market_return'] + intercept
        combined['residual_return'] = combined['stock_return'] - combined['predicted_return']
        
        # Add residuals back to original stock dataframe
        stock_filtered = stock_df.copy()
        stock_filtered['market_return'] = combined['market_return']
        stock_filtered['stock_return'] = combined['stock_return']
        stock_filtered['residual_return'] = combined['residual_return']
        
        print(f"Market correlation (R²): {r_value**2:.3f}")
        print(f"Beta coefficient: {slope:.3f}")
        
        return stock_filtered
    
    def identify_earnings_dates(self, df, threshold_percentile=95, target_dates=15, use_residuals=True):
        """Identify potential earnings dates using volume * gap, with option to use residual returns"""
        df = df.copy()
        
        if use_residuals and 'residual_return' in df.columns:
            # We need to convert residual returns (which are daily returns) to residual prices
            # Calculate cumulative residual price levels, starting at a base of 100.
            # The first residual_return is typically NaN; fillna(0) handles this for cumprod,
            # effectively treating the first day's growth factor as 1 (0% change).
            if not df.empty:
                df['residual_price'] = 100.0 * (1 + df['residual_return'].fillna(0)).cumprod()
            else:
                df['residual_price'] = pd.Series(dtype=float) # Handle empty DataFrame case

            
            # Calculate the residual gap (change between previous day's residual price and current day's)
            # Since we don't have open/close for residuals, we use the daily shift
            df['residual_gap_pct'] = abs(df['residual_price'].pct_change() * 100)
            
            # Calculate the combined metric: volume * residual gap
            df['volume_gap_product'] = df['volume'] * df['residual_gap_pct']
            
            gap_type = "residual returns"
        else:
            # Use the original price gap calculation if residuals not available or not requested
            df['price_gap_pct'] = abs((df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100)
            df['volume_gap_product'] = df['volume'] * df['price_gap_pct']
            gap_type = "price gap"
        
        # Initialize earnings_classification column with 'none' for all rows
        df['earnings_classification'] = 'none'
        
        # Handle edge cases
        if target_dates <= 0:
            print("Target dates must be greater than 0, defaulting to 1")
            target_dates = 1
        elif target_dates > len(df) - 1:
            print(f"Target dates {target_dates} exceeds available data points, using maximum available: {len(df) - 1}")
            target_dates = len(df) - 1
        
        # Use binary search to find the exact threshold that gives us the target number of dates
        # Sort the volume_gap_product values
        sorted_products = sorted(df['volume_gap_product'].dropna().values, reverse=True)
        
        # If we have fewer data points than target_dates, use all available data
        if len(sorted_products) <= target_dates:
            threshold = 0
        else:
            # Get the threshold value at the target_dates position (0-indexed)
            threshold = sorted_products[target_dates - 1]
        
        # Mark potential earnings dates
        df['is_earnings_date'] = df['volume_gap_product'] >= threshold
        
        # Count identified dates
        dates_found = df['is_earnings_date'].sum()
        
        print(f"Using {gap_type} for gap calculation")
        print(f"Volume * Gap threshold: {threshold:.2f}")
        print(f"Identified {dates_found} potential earnings dates using volume * gap formula")
        
        # For debugging, print the top dates sorted by the combined metric
        top_dates = df.sort_values('volume_gap_product', ascending=False).head(10)
        print("\nTop 10 dates by volume * gap:")
        for date, row in top_dates.iterrows():
            print(f"{date.strftime('%Y-%m-%d')}: Volume={row['volume']:,.0f}, Gap={row['price_gap_pct'] if 'price_gap_pct' in row else row['residual_gap_pct']:.2f}%, Product={row['volume_gap_product']:,.0f}")
        
        return df
    
    def classify_earnings_reactions(self, df):
        """Classify earnings reactions based on price movements"""
        df = df.copy()
        earnings_dates = df[df['is_earnings_date']].copy()
        
        if len(earnings_dates) == 0:
            print("No earnings dates found to classify")
            return df
        
        classifications = []
        strengths = []
        
        for date, row in earnings_dates.iterrows():
            # Get previous day's close
            try:
                prev_date = df.index[df.index < date][-1]
                prev_close = df.loc[prev_date, 'close']
            except (IndexError, KeyError):
                classifications.append('unknown')
                continue
            
            current_open = row['open']
            current_close = row['close']
            
            # Determine if gap is negative (prev_close > current_open)
            gap_negative = prev_close > current_open
            
            # Determine if intraday movement is positive (current_close > current_open)
            intraday_positive = current_close > current_open
            
            # Classification logic based on your specification
            if gap_negative:
                if intraday_positive:
                    # Gap down but recovered during day = negative anticipated
                    classification = 'negative_anticipated'
                else:
                    # Gap down and continued down = surprising negative
                    classification = 'surprising_negative'
            else:  # gap_positive or no gap
                if intraday_positive:
                    # Gap up and continued up = surprising positive
                    classification = 'surprising_positive'
                else:
                    # Gap up but declined during day = positive anticipated
                    classification = 'positive_anticipated'
            
            # Calculate strength as the difference between high and low on event day
            strength = row['high'] - row['low']
            
            classifications.append(classification)
            strengths.append(strength)
        
        # Add classifications and strengths to the dataframe
        df['earnings_classification'] = 'none'
        df['event_strength'] = 0.0  # Initialize strength column
        earnings_indices = df[df['is_earnings_date']].index
        for i, (classification, strength) in enumerate(zip(classifications, strengths)):
            df.loc[earnings_indices[i], 'earnings_classification'] = classification
            df.loc[earnings_indices[i], 'event_strength'] = strength
        
        return df
    
    def analyze_earnings_statistics(self, df):
        """Analyze and display earnings classification statistics"""
        earnings_data = df[df['is_earnings_date']].copy()
        
        if len(earnings_data) == 0:
            print("No earnings data to analyze")
            return
        
        print("\n=== EARNINGS CLASSIFICATION SUMMARY ===")
        classification_counts = earnings_data['earnings_classification'].value_counts()
        print(classification_counts)
        
        print("\n=== DETAILED EARNINGS EVENTS ===")
        for date, row in earnings_data.iterrows():
            prev_date = df.index[df.index < date][-1] if len(df.index[df.index < date]) > 0 else None
            if prev_date is not None:
                prev_close = df.loc[prev_date, 'close']
                gap = ((row['open'] - prev_close) / prev_close) * 100
                intraday = ((row['close'] - row['open']) / row['open']) * 100
                total_change = ((row['close'] - prev_close) / prev_close) * 100
                
                print(f"\nDate: {date.strftime('%Y-%m-%d')}")
                print(f"Classification: {row['earnings_classification']}")
                print(f"Previous Close: ${prev_close:.2f}")
                print(f"Open: ${row['open']:.2f} (Gap: {gap:+.2f}%)")
                print(f"Close: ${row['close']:.2f} (Intraday: {intraday:+.2f}%)")
                print(f"Total Change: {total_change:+.2f}%")
                print(f"Event Strength: {row['event_strength']:.2f}%")
                print(f"Volume: {row['volume']:,}")
                print(f"Volume * Gap Product: {row['volume_gap_product']:,.0f}")
    
    def plot_analysis(self, df):
        """Create visualization of the analysis"""
        try:
            fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
            
            # Get ticker name from the first plot title
            ticker = df.index.name if df.index.name else 'Stock'
            
            # Plot 1: Stock price with earnings dates
            axes[0].plot(df.index, df['close'], label=f'{ticker} Close Price', alpha=0.7)
            earnings_dates = df[df['is_earnings_date']]
            
            if not earnings_dates.empty:
                colors = {'negative_anticipated': 'orange', 'surprising_negative': 'red',
                         'positive_anticipated': 'lightgreen', 'surprising_positive': 'darkgreen',
                         'none': 'blue'}
                
                # Check if earnings_classification column exists and has valid values
                if 'earnings_classification' in earnings_dates.columns:
                    for classification in colors.keys():
                        mask = earnings_dates['earnings_classification'] == classification
                        if mask.any():
                            axes[0].scatter(earnings_dates[mask].index, earnings_dates[mask]['close'],
                                          color=colors[classification], label=classification, s=100, alpha=0.8)
                else:
                    # If no classification, just mark all earnings dates
                    axes[0].scatter(earnings_dates.index, earnings_dates['close'],
                                  color='blue', label='Earnings Date', s=100, alpha=0.8)
            
            axes[0].set_title(f'{ticker} Stock Price with Earnings Events')
            axes[0].set_ylabel('Price ($)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Volume with earnings dates
            axes[1].bar(df.index, df['volume'], alpha=0.5, color='blue', width=1)
            if not earnings_dates.empty:
                axes[1].bar(earnings_dates.index, earnings_dates['volume'], 
                           color='red', alpha=0.8, width=1, label='Earnings Dates')
            
            axes[1].set_title('Trading Volume with Identified Earnings Dates')
            axes[1].set_ylabel('Volume')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Residual returns (market-filtered movements)
            if 'residual_return' in df.columns:
                axes[2].plot(df.index, df['residual_return'] * 100, alpha=0.7, color='purple', label='Residual Return')
                
                if not earnings_dates.empty and 'residual_return' in earnings_dates.columns:
                    axes[2].scatter(earnings_dates.index, earnings_dates['residual_return'] * 100,
                                   color='red', s=50, alpha=0.8, label='Earnings Dates')
                    
                axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[2].set_title('Market-Filtered Stock Movements (Residual Returns)')
                axes[2].set_ylabel('Residual Return (%)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            else:
                axes[2].text(0.5, 0.5, 'No residual return data available', 
                             horizontalalignment='center', verticalalignment='center',
                             transform=axes[2].transAxes)
            
            # Plot 4: Volume * Gap Product
            if 'volume_gap_product' in df.columns:
                # Use log scale for better visualization
                log_product = np.log10(df['volume_gap_product'].replace(0, 1))
                axes[3].plot(df.index, log_product, alpha=0.5, color='blue', label='Log10(Volume * Gap)')
                
                if not earnings_dates.empty:
                    log_product_earnings = np.log10(earnings_dates['volume_gap_product'].replace(0, 1))
                    axes[3].scatter(earnings_dates.index, log_product_earnings,
                                  color='red', s=50, alpha=0.8, label='Earnings Dates')
                
                # Add threshold line
                if 'is_earnings_date' in df.columns and df['is_earnings_date'].any():
                    threshold = np.min(np.log10(earnings_dates['volume_gap_product']))
                    axes[3].axhline(y=threshold, color='green', linestyle='--', 
                                   alpha=0.8, label=f'Threshold (Log10)')
                
                axes[3].set_title('Volume * Gap Product (Log10 Scale)')
                axes[3].set_ylabel('Log10(Volume * Gap)')
                axes[3].set_xlabel('Date')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
            else:
                axes[3].text(0.5, 0.5, 'No Volume * Gap Product data available', 
                             horizontalalignment='center', verticalalignment='center',
                             transform=axes[3].transAxes)
            
            # Plot 5: Event Strength (price difference between open and close on event days)
            if 'event_strength' in df.columns:
                # Create a new dataframe with only earnings dates for better visualization
                strength_data = df[df['is_earnings_date']].copy()
                
                if not strength_data.empty:
                    # Create bar chart of strength values
                    colors = {'negative_anticipated': 'orange', 'surprising_negative': 'red',
                             'positive_anticipated': 'lightgreen', 'surprising_positive': 'darkgreen',
                             'none': 'blue'}
                    
                    # Get colors based on classification
                    bar_colors = [colors.get(c, 'blue') for c in strength_data['earnings_classification']]
                    
                    # Plot bars
                    bars = axes[4].bar(strength_data.index, strength_data['event_strength'], 
                                      alpha=0.7, color=bar_colors, width=5)
                    
                    # Add classification labels above bars
                    for i, (date, row) in enumerate(strength_data.iterrows()):
                        axes[4].text(date, row['event_strength'] + 0.5, 
                                    row['earnings_classification'].replace('_', '\n'),
                                    ha='center', va='bottom', rotation=90, fontsize=8)
                    
                    # Add average strength line
                    avg_strength = strength_data['event_strength'].mean()
                    axes[4].axhline(y=avg_strength, color='black', linestyle='--', 
                                   alpha=0.8, label=f'Avg Strength: {avg_strength:.2f}%')
                    
                    axes[4].set_title('Event Strength (Open-Close Price Difference on Event Days)')
                    axes[4].set_ylabel('Strength (%)')
                    axes[4].set_xlabel('Date')
                    axes[4].legend()
                    axes[4].grid(True, alpha=0.3)
                else:
                    axes[4].text(0.5, 0.5, 'No event strength data available', 
                                horizontalalignment='center', verticalalignment='center',
                                transform=axes[4].transAxes)
            else:
                axes[4].text(0.5, 0.5, 'No event strength data available', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[4].transAxes)
            
            plt.tight_layout()
            plt.savefig(f'data/{ticker}_analysis.png')  # Save figure to file instead of showing
            print(f"Analysis plot saved to data/{ticker}_analysis.png")
            plt.close()
        except Exception as e:
            print(f"Error in plot_analysis: {str(e)}")

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Stock Earnings Date Analysis")
    parser.add_argument("ticker", nargs="?", default="NVDA", help="Stock ticker symbol to analyze")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker symbol (default: SPY)")
    parser.add_argument("--days", type=int, default=1700, help="Number of days to analyze (default: 365)")
    parser.add_argument("--min-events", type=int, default=15, help="Exact number of events to identify (default: 15)")
    args = parser.parse_args()
    
    # Configuration
    POLYGON_API_KEY = "24EpiGqrNzm3s6rjkfxaP7eVX6PSaubu"
    STOCK_TICKER = args.ticker
    WORLD_INDEX_TICKER = args.benchmark  # Using SPY as world index proxy by default
    
    # Date range (default: last year)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    
    print("=== STOCK ANALYSIS ===")
    print(f"Analyzing {STOCK_TICKER} vs {WORLD_INDEX_TICKER}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Initialize analyzer
    analyzer = StockAnalyzer(POLYGON_API_KEY)
    
    try:
        # Step 1: Load stock data
        print("\n1. Loading stock data...")
        stock_data = analyzer.fetch_stock_data(STOCK_TICKER, start_date, end_date)
        stock_data.index.name = STOCK_TICKER  # Set the ticker as index name for reference
        world_data = analyzer.fetch_stock_data(WORLD_INDEX_TICKER, start_date, end_date)
        
        print(f"{STOCK_TICKER} data: {len(stock_data)} days")
        print(f"World index data: {len(world_data)} days")
        
        # Step 2: Filter out market movements
        print("\n2. Filtering out market movements...")
        stock_filtered = analyzer.filter_market_movements(stock_data, world_data)
        
        # Step 3: Identify potential earnings dates
        print("\n3. Identifying potential earnings dates...")
        # Use the formula: volume * residual gap
        stock_with_earnings = analyzer.identify_earnings_dates(
            stock_filtered,
            threshold_percentile=95,  # Starting percentile (not used anymore but kept for compatibility)
            target_dates=args.min_events,  # Use the command-line argument to control exact number of events
            use_residuals=True       # Use residual returns for gap calculation
        )
        
        # Step 4: Classify earnings reactions
        print("\n4. Classifying earnings reactions...")
        stock_with_classifications = analyzer.classify_earnings_reactions(stock_with_earnings)
        
        # Analyze results
        analyzer.analyze_earnings_statistics(stock_with_classifications)
        
        # Step 5: Visualize the analysis
        print("\n5. Generating visualizations...")
        analyzer.plot_analysis(stock_with_classifications)
        
        # Save results to CSV in data folder
        output_file = f"data/{STOCK_TICKER}_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Delete any existing analysis files for this ticker
        existing_analysis_files = glob.glob(f"data/{STOCK_TICKER}_analysis_*.csv")
        for file in existing_analysis_files:
            try:
                os.remove(file)
                print(f"Removed existing analysis file: {file}")
            except OSError as e:
                print(f"Error removing {file}: {e}")
        
        stock_with_classifications.to_csv(output_file)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your API key and internet connection.")

if __name__ == "__main__":
    main()