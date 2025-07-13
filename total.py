#!/usr/bin/env python3
"""
total.py - Combine Parquet data from multiple stocks and run contextual bandits with Thompson sampling

Usage:
    python total.py AVAV EH NVO QCOM SPY --output combined_data
"""

import pandas as pd
import numpy as np
import argparse
import os
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

def find_parquet_files(ticker):
    """Find all Parquet files for a given ticker"""
    pattern = f"data/{ticker}_features_*.parquet"
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: No Parquet files found for {ticker}")
        return []
    return files

def load_stock_data(ticker):
    """Load the most recent Parquet file for a ticker"""
    files = find_parquet_files(ticker)
    if not files:
        return None
    
    # Use the most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading data for {ticker} from {latest_file}")
    
    try:
        df = pd.read_parquet(latest_file)
        return df
    except Exception as e:
        print(f"Error loading {latest_file}: {str(e)}")
        return None

def combine_stock_data(tickers):
    """Combine data from multiple stocks into a single DataFrame"""
    all_data = []
    
    for ticker in tickers:
        df = load_stock_data(ticker)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("No data found for any of the specified tickers")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    return combined_df

class ContextualThompsonBandit:
    """
    Contextual Thompson Sampling Bandit for stock trading decisions.
    Actions: 0 = do nothing, 1 = buy
    """
    def __init__(self, n_actions=2, n_contexts=10):
        # Initialize alpha and beta parameters for each action and context
        # Alpha represents success counts, Beta represents failure counts
        self.n_actions = n_actions  # 0: do nothing, 1: buy
        self.n_contexts = n_contexts  # Number of context bins (increased to 10)
        
        # For each context, we have alpha and beta parameters for each action
        self.alpha = np.ones((n_contexts, n_actions))
        self.beta = np.ones((n_contexts, n_actions))
        
        # Track rewards
        self.rewards_history = []
        self.context_history = []
        self.action_history = []
        self.date_history = []
        self.ticker_history = []
        
        # Track feature values for analysis
        self.residual_gap_history = []
        self.stock_return_history = []
        self.market_return_history = []
        self.is_earnings_history = []
        self.event_strength_history = []
    
    def get_context_bin(self, features):
        """
        Convert continuous features to discrete context bins
        Using multiple features for more nuanced context
        """
        # Extract key features
        residual_gap_pct = features['residual_gap_pct']
        residual_return = features.get('residual_return', 0)
        volume_gap_product = features.get('volume_gap_product', 0)
        stock_return = features.get('stock_return', 0)
        market_return = features.get('market_return', 0)
        is_earnings_date = features.get('is_earnings_date', False)
        event_strength = features.get('event_strength', 0)
        
        # Create a composite context based on multiple factors
        # 1. Residual gap direction and magnitude
        if residual_gap_pct <= -0.1:  # Strong negative gap
            gap_context = 0
        elif residual_gap_pct <= -0.05:
            gap_context = 1
        elif residual_gap_pct <= 0:
            gap_context = 2
        elif residual_gap_pct <= 0.05:
            gap_context = 3
        else:  # Strong positive gap
            gap_context = 4
            
        # 2. Recent momentum (stock vs market)
        if stock_return > market_return + 0.01:  # Stock outperforming market
            momentum_context = 2
        elif stock_return < market_return - 0.01:  # Stock underperforming market
            momentum_context = 0
        else:  # Stock moving with market
            momentum_context = 1
            
        # 3. Event context
        if is_earnings_date and event_strength > 0.5:
            event_context = 2  # Strong positive event
        elif is_earnings_date and event_strength < -0.5:
            event_context = 0  # Strong negative event
        elif is_earnings_date:
            event_context = 1  # Neutral event
        else:
            event_context = 3  # No event
        
        # Combine contexts into a single value (0-44)
        # Gap context (0-4) * 15 + Momentum context (0-2) * 3 + Event context (0-3)
        combined_context = gap_context * 15 + momentum_context * 3 + event_context
        
        # Map to a smaller number of bins for better learning
        # We'll use 10 context bins in total
        return combined_context % 10
    
    def choose_action(self, context_bin):
        """
        Choose action using Thompson sampling
        """
        # Sample from beta distributions for each action in this context
        samples = [beta.rvs(self.alpha[context_bin, a], self.beta[context_bin, a]) 
                  for a in range(self.n_actions)]
        
        # Choose action with highest sampled value
        return np.argmax(samples)
    
    def update(self, context_bin, action, reward):
        """
        Update bandit parameters based on observed reward
        """
        # For the chosen action and context, update alpha (success) or beta (failure)
        if reward > 0:  # Success
            self.alpha[context_bin, action] += reward
        else:  # Failure
            self.beta[context_bin, action] += abs(reward)
    
    def record_decision(self, context_bin, action, reward, date, ticker, features):
        """
        Record the decision and outcome for later analysis
        """
        self.context_history.append(context_bin)
        self.action_history.append(action)
        self.rewards_history.append(reward)
        self.date_history.append(date)
        self.ticker_history.append(ticker)
        
        # Record feature values for analysis
        self.residual_gap_history.append(features.get('residual_gap_pct', 0))
        self.stock_return_history.append(features.get('stock_return', 0))
        self.market_return_history.append(features.get('market_return', 0))
        self.is_earnings_history.append(features.get('is_earnings_date', False))
        self.event_strength_history.append(features.get('event_strength', 0))

def calculate_max_return(df, buy_date, ticker, window_days=30):
    """
    Calculate the maximum possible return within a window after buying
    """
    # Filter data for the specific ticker
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('date')
    
    # Find the buy price
    buy_row = ticker_data[ticker_data['date'] == buy_date]
    if buy_row.empty:
        return 0.0
    
    buy_price = buy_row['close'].values[0]
    
    # Calculate the end date for our window
    end_date = pd.to_datetime(buy_date) + timedelta(days=window_days)
    
    # Get all prices within the window
    window_data = ticker_data[
        (ticker_data['date'] > buy_date) & 
        (ticker_data['date'] <= end_date)
    ]
    
    if window_data.empty:
        return 0.0
    
    # Find the maximum price in the window
    max_price = window_data['close'].max()
    
    # Calculate the maximum return
    max_return = (max_price - buy_price) / buy_price
    
    return max_return

def run_contextual_bandits(df, tickers):
    """
    Run contextual bandits on the stock data
    """
    # Sort data by date
    df = df.sort_values('date')
    
    # Create a bandit for each stock
    bandits = {ticker: ContextualThompsonBandit() for ticker in tickers}
    
    # Track buy events for visualization
    buy_events = {ticker: [] for ticker in tickers}
    
    # Process data chronologically
    dates = sorted(df['date'].unique())
    
    for date in dates:
        date_df = df[df['date'] == date]
        
        for ticker in tickers:
            ticker_data = date_df[date_df['ticker'] == ticker]
            
            if ticker_data.empty:
                continue
            
            # Extract features for the current day
            features = ticker_data.iloc[0].to_dict()
            
            # Get context bin based on features
            context_bin = bandits[ticker].get_context_bin(features)
            
            # Choose action using Thompson sampling
            action = bandits[ticker].choose_action(context_bin)
            
            # If action is to buy (action=1), calculate reward
            if action == 1:  # Buy
                # Calculate the maximum return within the next month
                reward = calculate_max_return(df, date, ticker, window_days=30)
                
                # Record buy event for visualization
                buy_events[ticker].append({
                    'date': date,
                    'price': features['close'],
                    'reward': reward
                })
            else:  # Do nothing
                reward = 0.0
            
            # Update bandit parameters
            bandits[ticker].update(context_bin, action, reward)
            
            # Record decision with features
            bandits[ticker].record_decision(context_bin, action, reward, date, ticker, features)
    
    return bandits, buy_events

def visualize_results(df, bandits, buy_events, tickers):
    """
    Visualize the results of the contextual bandits
    """
    # Create a directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # 1. Plot stock prices and buy events
    plt.figure(figsize=(15, 10))
    
    for i, ticker in enumerate(tickers):
        plt.subplot(len(tickers), 1, i+1)
        
        # Get ticker data
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        
        if ticker_data.empty:
            continue
        
        # Plot stock price
        plt.plot(ticker_data['date'], ticker_data['close'], label=f'{ticker} Price')
        
        # Plot buy events
        buy_dates = [event['date'] for event in buy_events[ticker]]
        buy_prices = [event['price'] for event in buy_events[ticker]]
        buy_rewards = [event['reward'] for event in buy_events[ticker]]
        
        # Use color gradient based on reward
        if buy_dates:
            sc = plt.scatter(buy_dates, buy_prices, c=buy_rewards, 
                          cmap='RdYlGn', vmin=-0.1, vmax=0.3, s=50, label='Buy Events')
            plt.colorbar(sc, label='30-day Max Return')
        
        plt.title(f'{ticker} Price and Buy Decisions')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        
        if i == len(tickers) - 1:
            plt.xlabel('Date')
        
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/stock_buy_events.png')
    
    # 2. Plot reward distribution by context bin for each stock
    plt.figure(figsize=(15, 10))
    
    for i, ticker in enumerate(tickers):
        plt.subplot(len(tickers), 1, i+1)
        
        bandit = bandits[ticker]
        
        # Create a DataFrame for visualization
        results_df = pd.DataFrame({
            'Context': bandit.context_history,
            'Action': bandit.action_history,
            'Reward': bandit.rewards_history,
            'Date': bandit.date_history,
            'ResidualGap': bandit.residual_gap_history,
            'StockReturn': bandit.stock_return_history,
            'MarketReturn': bandit.market_return_history,
            'IsEarnings': bandit.is_earnings_history,
            'EventStrength': bandit.event_strength_history
        })
        
        # Filter for buy actions only
        buy_df = results_df[results_df['Action'] == 1]
        
        if not buy_df.empty:
            sns.boxplot(x='Context', y='Reward', data=buy_df)
            plt.title(f'{ticker} Reward Distribution by Context Bin')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            
            if i == len(tickers) - 1:
                plt.xlabel('Context Bin')
    
    plt.tight_layout()
    plt.savefig('plots/reward_by_context.png')
    
    # 3. Plot feature importance analysis
    plt.figure(figsize=(15, 15))
    
    for i, ticker in enumerate(tickers):
        bandit = bandits[ticker]
        
        # Create a DataFrame for visualization
        results_df = pd.DataFrame({
            'Context': bandit.context_history,
            'Action': bandit.action_history,
            'Reward': bandit.rewards_history,
            'ResidualGap': bandit.residual_gap_history,
            'StockReturn': bandit.stock_return_history,
            'MarketReturn': bandit.market_return_history,
            'IsEarnings': bandit.is_earnings_history,
            'EventStrength': bandit.event_strength_history
        })
        
        # Filter for buy actions only
        buy_df = results_df[results_df['Action'] == 1]
        
        if buy_df.empty:
            continue
            
        # Plot feature correlations with reward
        plt.subplot(len(tickers), 3, i*3+1)
        sns.scatterplot(x='ResidualGap', y='Reward', data=buy_df)
        plt.title(f'{ticker}: Residual Gap vs Reward')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(len(tickers), 3, i*3+2)
        sns.scatterplot(x='StockReturn', y='Reward', data=buy_df)
        plt.title(f'{ticker}: Stock Return vs Reward')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(len(tickers), 3, i*3+3)
        earnings_df = buy_df[buy_df['IsEarnings'] == True]
        if not earnings_df.empty:
            sns.scatterplot(x='EventStrength', y='Reward', data=earnings_df)
            plt.title(f'{ticker}: Event Strength vs Reward (Earnings Days)')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No earnings events with buy actions', 
                     horizontalalignment='center', verticalalignment='center')
            plt.title(f'{ticker}: No Earnings Events with Buy Actions')
    
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    
    # 4. Plot bandit learning over time
    plt.figure(figsize=(15, 10))
    
    for i, ticker in enumerate(tickers):
        plt.subplot(len(tickers), 1, i+1)
        
        bandit = bandits[ticker]
        
        # Create a DataFrame for visualization
        results_df = pd.DataFrame({
            'Context': bandit.context_history,
            'Action': bandit.action_history,
            'Reward': bandit.rewards_history,
            'Date': bandit.date_history
        })
        
        # Filter for buy actions only
        buy_df = results_df[results_df['Action'] == 1]
        
        if not buy_df.empty:
            # Calculate cumulative average reward
            buy_df = buy_df.sort_values('Date')
            buy_df['CumulativeReward'] = buy_df['Reward'].cumsum()
            buy_df['Count'] = range(1, len(buy_df) + 1)
            buy_df['CumulativeAvgReward'] = buy_df['CumulativeReward'] / buy_df['Count']
            
            plt.plot(buy_df['Date'], buy_df['CumulativeAvgReward'])
            plt.title(f'{ticker} Cumulative Average Reward Over Time')
            plt.ylabel('Cumulative Avg Reward')
            plt.grid(True, alpha=0.3)
            
            if i == len(tickers) - 1:
                plt.xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('plots/learning_over_time.png')
    
    return

def generate_summary(bandits, tickers):
    """
    Generate a summary of the bandit performance with feature importance insights
    """
    summary = []
    feature_importance = []
    context_performance = []
    
    for ticker in tickers:
        bandit = bandits[ticker]
        
        # Create a DataFrame for analysis
        results_df = pd.DataFrame({
            'Context': bandit.context_history,
            'Action': bandit.action_history,
            'Reward': bandit.rewards_history,
            'Date': bandit.date_history,
            'ResidualGap': bandit.residual_gap_history,
            'StockReturn': bandit.stock_return_history,
            'MarketReturn': bandit.market_return_history,
            'IsEarnings': bandit.is_earnings_history,
            'EventStrength': bandit.event_strength_history
        })
        
        # Filter for buy actions only
        buy_df = results_df[results_df['Action'] == 1]
        
        if buy_df.empty:
            avg_reward = 0.0
            total_buys = 0
            positive_pct = 0.0
            best_context = None
            best_context_reward = 0.0
        else:
            avg_reward = buy_df['Reward'].mean()
            total_buys = len(buy_df)
            positive_pct = (buy_df['Reward'] > 0).mean() * 100
            
            # Find best performing context
            context_rewards = buy_df.groupby('Context')['Reward'].agg(['mean', 'count'])
            context_rewards = context_rewards[context_rewards['count'] >= 5]  # At least 5 samples
            
            if not context_rewards.empty:
                best_context = context_rewards['mean'].idxmax()
                best_context_reward = context_rewards.loc[best_context, 'mean']
            else:
                best_context = None
                best_context_reward = 0.0
            
            # Calculate feature correlations with reward
            correlations = {}
            for feature in ['ResidualGap', 'StockReturn', 'MarketReturn', 'EventStrength']:
                corr = buy_df[['Reward', feature]].corr().iloc[0, 1]
                correlations[feature] = corr
            
            # Add to feature importance list
            feature_importance.append({
                'Ticker': ticker,
                'ResidualGap_Corr': correlations['ResidualGap'],
                'StockReturn_Corr': correlations['StockReturn'],
                'MarketReturn_Corr': correlations['MarketReturn'],
                'EventStrength_Corr': correlations['EventStrength']
            })
            
            # Add context performance
            for context, data in context_rewards.iterrows():
                context_performance.append({
                    'Ticker': ticker,
                    'Context': context,
                    'AvgReward': data['mean'],
                    'Count': data['count'],
                    'PositiveRate': (buy_df[buy_df['Context'] == context]['Reward'] > 0).mean() * 100
                })
        
        summary.append({
            'Ticker': ticker,
            'Total Buy Actions': total_buys,
            'Average Reward': avg_reward,
            'Positive Reward %': positive_pct,
            'Best Context': best_context,
            'Best Context Reward': best_context_reward
        })
    
    # Create summary DataFrames
    summary_df = pd.DataFrame(summary)
    feature_importance_df = pd.DataFrame(feature_importance)
    context_performance_df = pd.DataFrame(context_performance)
    
    # Save additional insights to CSV
    if not feature_importance_df.empty:
        feature_importance_df.to_csv('plots/feature_importance.csv', index=False)
    
    if not context_performance_df.empty:
        context_performance_df.to_csv('plots/context_performance.csv', index=False)
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description="Combine Parquet data and run contextual bandits")
    parser.add_argument("tickers", nargs="+", help="Stock tickers to analyze (e.g., AVAV EH NVO QCOM SPY)")
    parser.add_argument("--output", default="combined_stocks", help="Output filename (without extension)")
    args = parser.parse_args()
    
    print(f"=== ANALYZING DATA FOR {len(args.tickers)} STOCKS ===")
    print(f"Tickers: {', '.join(args.tickers)}")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Combine data from all specified tickers
    combined_data = combine_stock_data(args.tickers)
    
    if combined_data is not None:
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(combined_data['date']):
            combined_data['date'] = pd.to_datetime(combined_data['date'])
        
        # Run contextual bandits
        print("\n=== RUNNING CONTEXTUAL BANDITS WITH THOMPSON SAMPLING ===")
        bandits, buy_events = run_contextual_bandits(combined_data, args.tickers)
        
        # Visualize results
        print("\n=== GENERATING VISUALIZATIONS ===")
        visualize_results(combined_data, bandits, buy_events, args.tickers)
        
        # Generate summary
        print("\n=== BANDIT PERFORMANCE SUMMARY ===")
        summary_df = generate_summary(bandits, args.tickers)
        print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_df.to_csv('plots/bandit_summary.csv', index=False)
        print("\nSummary saved to: plots/bandit_summary.csv")
        print("Visualizations saved to: plots/stock_buy_events.png and plots/reward_by_context.png")

if __name__ == "__main__":
    main()
