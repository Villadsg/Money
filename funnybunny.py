import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        
    def save_to_csv(self, df, ticker, start_date, end_date):
        """Save stock data to a CSV file"""
        # Create data directory if it doesn't exist
        import os
        os.makedirs('data', exist_ok=True)
        
        # Create filename with ticker and date range
        filename = f"data/{ticker}_{start_date}_to_{end_date}.csv"
        df.to_csv(filename)
        print(f"Data saved to {filename}")
        return filename
    
    def load_from_csv(self, ticker, start_date, end_date):
        """Load stock data from a CSV file if available"""
        filename = f"data/{ticker}_{start_date}_to_{end_date}.csv"
        import os
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
            import glob
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
    
    def identify_earnings_dates(self, df, threshold_percentile=95, min_dates=15, use_residuals=True):
        """Identify potential earnings dates using volume * gap, with option to use residual returns"""
        df = df.copy()
        
        if use_residuals and 'residual_return' in df.columns:
            # We need to convert residual returns (which are daily returns) to residual prices
            # First, create a column for residual price level (starting at 100)
            df['residual_price'] = 100.0
            
            # Calculate cumulative residual price levels
            for i in range(1, len(df)):
                prev_price = df['residual_price'].iloc[i-1]
                daily_return = df['residual_return'].iloc[i]
                df['residual_price'].iloc[i] = prev_price * (1 + daily_return)
            
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
        
        # Set threshold based on percentile of the combined metric
        threshold = np.percentile(df['volume_gap_product'].dropna(), threshold_percentile)
        
        # Mark potential earnings dates
        df['is_earnings_date'] = df['volume_gap_product'] > threshold
        
        # Initialize earnings_classification column with 'none' for all rows
        df['earnings_classification'] = 'none'
        
        # Count identified dates
        dates_found = df['is_earnings_date'].sum()
        
        # If we found too few dates, lower the threshold until we find at least min_dates
        if dates_found < min_dates and threshold_percentile > 50:
            print(f"Found only {dates_found} dates, lowering threshold to find at least {min_dates}...")
            # Recursively call with a lower threshold
            return self.identify_earnings_dates(df, threshold_percentile-5, min_dates, use_residuals)
        
        print(f"Using {gap_type} for gap calculation")
        print(f"Volume * Gap threshold: {threshold:.2f}")
        print(f"Identified {dates_found} potential earnings dates using volume * gap formula")
        
        # For debugging, print the top dates sorted by the combined metric
        top_dates = df.sort_values('volume_gap_product', ascending=False).head(10)
        print("\nTop 10 dates by volume * gap:")
        for date, row in top_dates.iterrows():
            print(f"{date.strftime('%Y-%m-%d')}: Volume={row['volume']:,.0f}, Gap={row['price_gap_pct'] if 'price_gap_pct' in row else row['residual_gap_pct']:.2f}%, Product={row['volume_gap_product']:,.0f}")
        
        return df
    
    def classify_earnings_reactions(self, df, event_length=1):
        """Classify earnings reactions based on price movements over a configurable time period
        
        Args:
            df: DataFrame with stock data
            event_length: Number of days to measure the reaction (1 = same day only, 2 = same day + next day, etc.)
        """
        df = df.copy()
        earnings_dates = df[df['is_earnings_date']].copy()
        
        if len(earnings_dates) == 0:
            print("No earnings dates found to classify")
            return df
        
        print(f"Classifying earnings reactions over {event_length} day(s)")
        
        classifications = []
        
        for date, row in earnings_dates.iterrows():
            # Get previous day's close
            try:
                prev_date = df.index[df.index < date][-1]
                prev_close = df.loc[prev_date, 'close']
            except (IndexError, KeyError):
                classifications.append('unknown')
                continue
            
            current_open = row['open']
            
            # Find the closing price after event_length days
            try:
                # Get all dates at or after the current date
                future_dates = df.index[df.index >= date]
                
                # Make sure we have enough data for the event length
                if len(future_dates) < event_length:
                    print(f"Warning: Not enough data after {date.strftime('%Y-%m-%d')} for {event_length} day analysis")
                    classifications.append('unknown')
                    continue
                
                # Get the closing price at the end of the event period
                end_date = future_dates[event_length - 1]  # -1 because we include the event day itself
                end_close = df.loc[end_date, 'close']
                
            except (IndexError, KeyError):
                print(f"Warning: Unable to get end price for {date.strftime('%Y-%m-%d')}")
                classifications.append('unknown')
                continue
            
            # Determine if gap is negative (prev_close > current_open)
            gap_negative = prev_close > current_open
            
            # Determine if overall movement over the event period is positive
            # (end_close > current_open)
            overall_positive = end_close > current_open
            
            # Classification logic based on gap direction and overall movement
            if gap_negative:
                if overall_positive:
                    # Gap down but recovered over the event period = negative anticipated
                    classification = 'negative_anticipated'
                else:
                    # Gap down and continued down over the event period = surprising negative
                    classification = 'surprising_negative'
            else:  # gap_positive or no gap
                if overall_positive:
                    # Gap up and continued up over the event period = surprising positive
                    classification = 'surprising_positive'
                else:
                    # Gap up but declined over the event period = positive anticipated
                    classification = 'positive_anticipated'
            
            classifications.append(classification)
        
        # Add classifications to the dataframe
        df['earnings_classification'] = 'none'
        earnings_indices = df[df['is_earnings_date']].index
        for i, classification in enumerate(classifications):
            df.loc[earnings_indices[i], 'earnings_classification'] = classification
        
        return df
    
    def analyze_earnings_statistics(self, df, event_length=1):
        """Analyze and display earnings classification statistics"""
        earnings_data = df[df['is_earnings_date']].copy()
        
        if len(earnings_data) == 0:
            print("No earnings data to analyze")
            return
        
        print(f"\n=== EARNINGS CLASSIFICATION SUMMARY (Event Length: {event_length} day(s)) ===")
        classification_counts = earnings_data['earnings_classification'].value_counts()
        print(classification_counts)
        
        print(f"\n=== DETAILED EARNINGS EVENTS (Event Length: {event_length} day(s)) ===")
        for date, row in earnings_data.iterrows():
            prev_date = df.index[df.index < date][-1] if len(df.index[df.index < date]) > 0 else None
            if prev_date is not None:
                prev_close = df.loc[prev_date, 'close']
                gap = ((row['open'] - prev_close) / prev_close) * 100
                
                # Get the end date for the event period
                future_dates = df.index[df.index >= date]
                if len(future_dates) >= event_length:
                    end_date = future_dates[event_length - 1]
                    end_close = df.loc[end_date, 'close']
                    
                    # Calculate movements
                    intraday = ((row['close'] - row['open']) / row['open']) * 100
                    overall_change = ((end_close - row['open']) / row['open']) * 100
                    total_change = ((end_close - prev_close) / prev_close) * 100
                    
                    print(f"\nDate: {date.strftime('%Y-%m-%d')}")
                    if event_length > 1:
                        print(f"Event Period: {date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    print(f"Classification: {row['earnings_classification']}")
                    print(f"Previous Close: ${prev_close:.2f}")
                    print(f"Event Day Open: ${row['open']:.2f} (Gap: {gap:+.2f}%)")
                    print(f"Event Day Close: ${row['close']:.2f} (Day 1 Intraday: {intraday:+.2f}%)")
                    if event_length > 1:
                        print(f"Period End Close: ${end_close:.2f} (Overall Event Move: {overall_change:+.2f}%)")
                    print(f"Total Change: {total_change:+.2f}%")
                    print(f"Volume: {row['volume']:,}")
                    print(f"Volume * Gap Product: {row['volume_gap_product']:,.0f}")
    
    def plot_analysis(self, df):
        """Create visualization of the analysis"""
        try:
            fig, axes = plt.subplots(4, 1, figsize=(15, 16))
            
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
            
            plt.tight_layout()
            plt.savefig(f'data/{ticker}_analysis.png')  # Save figure to file instead of showing
            print(f"Analysis plot saved to data/{ticker}_analysis.png")
            plt.close()
        except Exception as e:
            print(f"Error in plot_analysis: {str(e)}")

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Stock Earnings Date Analysis with Configurable Event Length")
    parser.add_argument("ticker", nargs="?", default="NVDA", help="Stock ticker symbol to analyze")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker symbol (default: SPY)")
    parser.add_argument("--days", type=int, default=1700, help="Number of days to analyze (default: 1700)")
    parser.add_argument("--eventlength", type=int, default=1, help="Number of days to measure event reaction (default: 1)")
    args = parser.parse_args()
    
    # Configuration
    POLYGON_API_KEY = "24EpiGqrNzm3s6rjkfxaP7eVX6PSaubu"
    STOCK_TICKER = args.ticker
    WORLD_INDEX_TICKER = args.benchmark  # Using SPY as world index proxy by default
    EVENT_LENGTH = args.eventlength
    
    # Date range (default: last year)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    
    print("=== STOCK ANALYSIS ===")
    print(f"Analyzing {STOCK_TICKER} vs {WORLD_INDEX_TICKER}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Event length: {EVENT_LENGTH} day(s)")
    
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
            threshold_percentile=95,  # Use 95th percentile as threshold
            min_dates=15,            # Try to find at least 15 dates
            use_residuals=True       # Use residual returns for gap calculation
        )
        
        # Step 4: Classify earnings reactions with configurable event length
        print("\n4. Classifying earnings reactions...")
        stock_with_classifications = analyzer.classify_earnings_reactions(
            stock_with_earnings, 
            event_length=EVENT_LENGTH
        )
        
        # Analyze results
        analyzer.analyze_earnings_statistics(stock_with_classifications, event_length=EVENT_LENGTH)
        
        # Step 5: Visualize the analysis
        print("\n5. Generating visualizations...")
        analyzer.plot_analysis(stock_with_classifications)
        
        # Save results to CSV
        output_file = f"{STOCK_TICKER}_analysis_eventlen{EVENT_LENGTH}_{datetime.now().strftime('%Y%m%d')}.csv"
        stock_with_classifications.to_csv(output_file)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your API key and internet connection.")

if __name__ == "__main__":
    main()