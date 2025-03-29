#!/usr/bin/env python3
"""
Test script for RL agent training with CPU-only mode
"""

import pandas as pd
import numpy as np
import yfinance as yf
from rl_agent import train_rl_agent, evaluate_rl_agent

def main():
    print("Fetching sample stock data for testing...")
    # Get sample data for Apple and NASDAQ
    stock_data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")['Close']
    market_data = yf.download("^IXIC", start="2023-01-01", end="2023-12-31")['Close']
    
    # Make sure both series have the same index
    common_dates = stock_data.index.intersection(market_data.index)
    stock_data = stock_data.loc[common_dates]
    market_data = market_data.loc[common_dates]
    
    # Set names for better output
    stock_data.name = "AAPL"
    market_data.name = "NASDAQ"
    
    print(f"Data points: {len(stock_data)}")
    print("Sample data:")
    print(f"Stock data first 5 points:\n{stock_data.head()}")
    print(f"\nMarket data first 5 points:\n{market_data.head()}")
    
    # Train the agent with a small number of episodes for testing
    print("\nTraining RL agent with CPU...")
    agent, history = train_rl_agent(stock_data, market_data, episodes=5, batch_size=32)
    
    # Evaluate the agent
    print("\nEvaluating RL agent...")
    results = evaluate_rl_agent(agent, stock_data, market_data)
    
    # Print summary
    print("\nTraining summary:")
    print(f"Final portfolio value: ${results['final_portfolio_value']:.2f}")
    print(f"Total reward: {results['total_reward']:.2f}")
    print(f"Action counts: {results['action_counts']}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
