"""
Test script for PyTorch-based RL agent
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from torch_rl_agent import train_rl_agent, evaluate_rl_agent

def main():
    """Main function to test the PyTorch RL agent"""
    print("Fetching sample stock data for testing...")
    
    # Fetch sample stock data
    stock_ticker = "AAPL"
    market_ticker = "^IXIC"  # NASDAQ
    
    # Get 1 year of data
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365)
    
    # Download data
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)['Close']
    market_data = yf.download(market_ticker, start=start_date, end=end_date)['Close']
    
    # Rename series for better identification
    stock_data.name = stock_ticker
    market_data.name = "NASDAQ"
    
    # Align the data (remove any missing dates)
    data = pd.concat([stock_data, market_data], axis=1).dropna()
    stock_data = data.iloc[:, 0]
    market_data = data.iloc[:, 1]
    
    print(f"Data points: {len(stock_data)}")
    print("Sample data:")
    print("Stock data first 5 points:")
    print(stock_data.head().to_frame())
    print("\nMarket data first 5 points:")
    print(market_data.head().to_frame())
    
    print("\nTraining RL agent with PyTorch...")
    # Train the agent
    agent, history = train_rl_agent(stock_data, market_data, episodes=5, batch_size=32)
    
    # Evaluate the agent
    evaluation = evaluate_rl_agent(agent, stock_data, market_data)
    
    # Plot training results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['portfolio_values'])
    plt.title('Portfolio Value')
    plt.xlabel('Episode')
    plt.ylabel('Value ($)')
    
    plt.tight_layout()
    plt.savefig('rl_training_results.png')
    print("Training results saved to 'rl_training_results.png'")
    
    # Save the trained model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'dqn_agent_{stock_ticker}.pth')
    agent.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
