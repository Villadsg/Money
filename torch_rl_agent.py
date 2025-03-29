"""
PyTorch-based Reinforcement Learning Agent for Stock Trading
"""
import os
import random
import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, List, Any

# Set PyTorch to use CPU only if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch using device: {device}")

class StockTradingEnv:
    """Environment for stock trading using reinforcement learning."""
    
    def __init__(self, stock_data, market_data, initial_balance=10000.0):
        """
        Initialize the trading environment.
        
        Args:
            stock_data (pd.Series): Historical stock price data
            market_data (pd.Series): Historical market benchmark data
            initial_balance (float): Initial cash balance
        """
        self.stock_data = stock_data
        self.market_data = market_data
        self.initial_balance = float(initial_balance)
        
        # Calculate returns for beta calculation
        self.stock_returns = stock_data.pct_change().fillna(0)
        self.market_returns = market_data.pct_change().fillna(0)
        
        # Calculate beta (market sensitivity)
        try:
            variance = np.var(self.market_returns.values)
            covariance = np.cov(self.stock_returns.values, self.market_returns.values)[0, 1]
            self.beta = covariance / variance if variance != 0 else 0
        except:
            self.beta = 0
            
        print(f"Environment created with beta: {self.beta:.4f}")
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.cash_balance = float(self.initial_balance)
        self.stock_owned = 0.0
        
        # Store initial price for normalization
        initial_stock_price = float(self.stock_data.iloc[0].item()) if hasattr(self.stock_data.iloc[0], 'item') else float(self.stock_data.iloc[0])
        self.initial_stock_price = initial_stock_price
        
        # Current portfolio value (cash + stock holdings)
        self.current_value = float(self.cash_balance)
        
        # Trading history
        self.trade_history = []
        
        # State includes: normalized stock price, normalized market price, stock return, market return, 
        # portfolio value, cash balance, stock owned, beta
        return self._get_observation()
    
    def _get_observation(self):
        """Get the current state observation."""
        # Convert pandas Series to float if needed
        stock_price = float(self.stock_data.iloc[self.current_step].item()) if hasattr(self.stock_data.iloc[self.current_step], 'item') else float(self.stock_data.iloc[self.current_step])
        market_price = float(self.market_data.iloc[self.current_step].item()) if hasattr(self.market_data.iloc[self.current_step], 'item') else float(self.market_data.iloc[self.current_step])
        
        # Use a window of returns for better context
        window_size = 5  # Smaller window size for simplicity
        start_idx = max(0, self.current_step - window_size + 1)
        stock_return_window = self.stock_returns.iloc[start_idx:self.current_step + 1].values
        market_return_window = self.market_returns.iloc[start_idx:self.current_step + 1].values
        
        # Pad with zeros if we don't have enough history
        if len(stock_return_window) < window_size:
            stock_return_window = np.pad(stock_return_window, (window_size - len(stock_return_window), 0), 'constant')
            market_return_window = np.pad(market_return_window, (window_size - len(market_return_window), 0), 'constant')
        
        # Calculate residual (stock return - beta * market return)
        current_stock_return = float(self.stock_returns.iloc[self.current_step].item()) if self.current_step < len(self.stock_returns) else 0.0
        current_market_return = float(self.market_returns.iloc[self.current_step].item()) if self.current_step < len(self.market_returns) else 0.0
        residual = current_stock_return - (self.beta * current_market_return)
        
        # Portfolio metrics
        portfolio_value = float(self.cash_balance) + (float(self.stock_owned) * float(stock_price))
        portfolio_return = (portfolio_value / self.current_value) - 1 if self.current_value > 0 else 0
        self.current_value = portfolio_value
        
        # Create a list of features for the observation vector
        features = [
            float(stock_price) / 100,  # Normalize price
            float(market_price) / 1000,  # Normalize price
            float(residual),  # Current residual
            float(portfolio_value) / self.initial_balance,  # Normalized portfolio value
            float(self.cash_balance) / self.initial_balance,  # Normalized cash balance
            float(self.stock_owned),  # Units of stock owned
            float(self.beta)  # Stock's beta to market
        ]
        
        # Add window features as individual elements - handle numpy arrays properly
        for val in stock_return_window.flatten():
            features.append(float(val))
        for val in market_return_window.flatten():
            features.append(float(val))
            
        # Convert to numpy array
        return np.array(features, dtype=np.float32)
    
    def step(self, action):
        """
        Take a trading action in the environment.
        
        Args:
            action (int): 0 = hold, 1 = buy, 2 = sell
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Get current price
        current_price = float(self.stock_data.iloc[self.current_step].item()) if hasattr(self.stock_data.iloc[self.current_step], 'item') else float(self.stock_data.iloc[self.current_step])
        
        # Initialize reward
        reward = 0
        done = False
        
        # Execute the action
        if action == 1:  # Buy
            if self.cash_balance >= current_price:
                # Buy one share
                shares_to_buy = 1
                cost = shares_to_buy * current_price
                self.stock_owned += shares_to_buy
                self.cash_balance -= cost
                self.trade_history.append(('buy', self.current_step, current_price, shares_to_buy))
        
        elif action == 2:  # Sell
            if self.stock_owned > 0:
                # Sell one share
                shares_to_sell = 1
                profit = shares_to_sell * current_price
                self.stock_owned -= shares_to_sell
                self.cash_balance += profit
                self.trade_history.append(('sell', self.current_step, current_price, shares_to_sell))
        
        # Move to the next time step
        self.current_step += 1
        
        # Check if we've reached the end of the data
        if self.current_step >= len(self.stock_data) - 1:
            done = True
        
        # Get the next price
        next_price = float(self.stock_data.iloc[self.current_step].item()) if hasattr(self.stock_data.iloc[self.current_step], 'item') else float(self.stock_data.iloc[self.current_step])
        
        # Calculate portfolio value
        next_portfolio_value = self.cash_balance + (self.stock_owned * next_price)
        
        # Calculate reward as the change in portfolio value
        daily_profit = next_portfolio_value - self.current_value
        reward = daily_profit
        
        # Get the next state
        next_observation = self._get_observation()
        
        # Information for debugging
        info = {
            'portfolio_value': next_portfolio_value,
            'cash_balance': self.cash_balance,
            'stock_owned': self.stock_owned,
            'current_price': next_price,
            'daily_profit': daily_profit
        }
        
        return next_observation, reward, done, info


class DQNNetwork(nn.Module):
    """Deep Q-Network for stock trading."""
    
    def __init__(self, state_size, action_size):
        """Initialize the network."""
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Network agent for stock trading."""
    
    def __init__(self, state_size, action_size):
        """
        Initialize the agent with state and action dimensions.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Memory for experience replay
        self.memory = deque(maxlen=2000)
        
        # Build model
        self.model = DQNNetwork(state_size, action_size).to(device)
        self.target_model = DQNNetwork(state_size, action_size).to(device)
        self.update_target_model()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print(f"DQN Agent created with {state_size} state dimensions and {action_size} actions")
    
    def update_target_model(self):
        """Update the target model with the weights of the main model."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for replay."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose an action based on the current state."""
        if training and np.random.rand() <= self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_size)
        
        # Exploitation: choose best action based on predicted Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()  # Return action with highest Q-value
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(device)
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            
            # Calculate target Q-value
            target = reward
            if not done:
                # Q-learning formula: Q(s,a) = r + gamma * max(Q(s',a'))
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()
            
            # Update Q-value for the taken action
            with torch.no_grad():
                target_f = self.model(state_tensor).detach()
            target_f[action] = target
            
            states.append(state)
            targets.append(target_f.cpu().numpy())
        
        # Train the model
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        targets_tensor = torch.FloatTensor(np.array(targets)).to(device)
        
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(states_tensor)
        
        # Calculate loss
        loss = F.mse_loss(outputs, targets_tensor)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save the model weights."""
        torch.save(self.model.state_dict(), filepath)
    
    def load(self, filepath):
        """Load the model weights."""
        self.model.load_state_dict(torch.load(filepath))
        self.update_target_model()


def train_rl_agent(stock_data, market_data, episodes=50, batch_size=32):
    """
    Train a reinforcement learning agent for stock trading.
    
    Args:
        stock_data (pd.Series): Historical stock price data
        market_data (pd.Series): Historical market benchmark data
        episodes (int): Number of training episodes
        batch_size (int): Batch size for experience replay
        
    Returns:
        tuple: (trained_agent, training_history)
    """
    print("\n======= Starting RL Agent Training =======")
    print(f"Stock: {stock_data.name}")
    print(f"Market Benchmark: {market_data.name}")
    print(f"Data Points: {len(stock_data)}")
    print(f"Training Episodes: {episodes}")
    print(f"Batch Size: {batch_size}")
    print("==========================================")
    
    # Create environment
    env = StockTradingEnv(stock_data, market_data)
    
    # Get state and action dimensions
    state = env.reset()
    state_size = len(state)
    action_size = 3  # hold, buy, sell
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Training history
    history = {
        'episode_rewards': [],
        'portfolio_values': []
    }
    
    print("\nTraining progress:")
    
    # Train for specified number of episodes
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        total_reward = 0
        done = False
        
        # Run one episode
        while not done:
            # Choose an action
            action = agent.act(state)
            
            # Take the action
            next_state, reward, done, info = env.step(action)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # Train the agent
            agent.replay(batch_size)
        
        # Update target network every episode
        agent.update_target_model()
        
        # Record history
        history['episode_rewards'].append(total_reward)
        history['portfolio_values'].append(info['portfolio_value'])
        
        # Print progress
        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward:.2f}, "
                  f"Portfolio Value: {info['portfolio_value']:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}")
    
    print("\nTraining complete!")
    
    return agent, history


def evaluate_rl_agent(agent, stock_data, market_data):
    """
    Evaluate a trained RL agent on stock data.
    
    Args:
        agent (DQNAgent): Trained RL agent
        stock_data (pd.Series): Stock price data for evaluation
        market_data (pd.Series): Market benchmark data
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n======= Evaluating RL Agent =======")
    
    # Create environment
    env = StockTradingEnv(stock_data, market_data)
    state = env.reset()
    
    # Evaluation metrics
    total_reward = 0
    done = False
    actions_taken = {0: 0, 1: 0, 2: 0}  # hold, buy, sell
    
    # Run through the environment
    while not done:
        # Choose an action (no exploration)
        action = agent.act(state, training=False)
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        
        # Update state and metrics
        state = next_state
        total_reward += reward
        actions_taken[action] += 1
    
    # Calculate final portfolio value
    final_portfolio = info['portfolio_value']
    initial_investment = env.initial_balance
    roi = (final_portfolio / initial_investment - 1) * 100
    
    # Print evaluation results
    print(f"Initial Investment: ${initial_investment:.2f}")
    print(f"Final Portfolio Value: ${final_portfolio:.2f}")
    print(f"Return on Investment: {roi:.2f}%")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Actions Taken: Hold={actions_taken[0]}, Buy={actions_taken[1]}, Sell={actions_taken[2]}")
    
    # Return evaluation metrics
    return {
        'initial_investment': initial_investment,
        'final_portfolio': final_portfolio,
        'roi': roi,
        'total_reward': total_reward,
        'actions_taken': actions_taken,
        'trade_history': env.trade_history
    }
