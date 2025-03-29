#!/usr/bin/env python3
"""
Reinforcement Learning Agent for Stock Trading
Uses historical stock price changes and market benchmark as environment
"""

# Check if TensorFlow is available
RL_AVAILABLE = True
try:
    import numpy as np
    import pandas as pd
    import random
    from collections import deque
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    
    # Configure TensorFlow to use CPU only
    # This prevents errors when GPU libraries are not properly installed
    print("Configuring TensorFlow to use CPU only for reinforcement learning")
    tf.config.set_visible_devices([], 'GPU')
    
    # Alternative approach if you want to try using GPU but fall back to CPU:
    # try:
    #     gpus = tf.config.experimental.list_physical_devices('GPU')
    #     if gpus:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         print(f"GPU support enabled: {len(gpus)} GPUs available")
    #     else:
    #         print("No GPU found, using CPU instead")
    # except Exception as e:
    #     print(f"Error configuring GPU: {e}. Using CPU instead.")
    
    # Verify TensorFlow is using CPU
    print(f"TensorFlow devices available: {tf.config.list_physical_devices()}")
    
except ImportError:
    RL_AVAILABLE = False
    print("TensorFlow not installed. Reinforcement learning functionality will be disabled.")


class StockTradingEnv:
    """Environment for stock trading based on historical data."""
    
    def __init__(self, stock_data, market_data, initial_balance=10000.0):
        """
        Initialize the environment with stock and market data.
        
        Args:
            stock_data (pd.Series): Historical stock price data
            market_data (pd.Series): Historical market benchmark data
            initial_balance (float): Initial cash balance
        """
        # Ensure data is aligned
        common_dates = stock_data.index.intersection(market_data.index)
        self.stock_data = stock_data.loc[common_dates]
        self.market_data = market_data.loc[common_dates]
        
        # Calculate daily returns
        self.stock_returns = self.stock_data.pct_change().fillna(0)
        self.market_returns = self.market_data.pct_change().fillna(0)
        
        # Calculate stock's beta to the market
        self.beta = self._calculate_beta()
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.reset()
        
    def _calculate_beta(self):
        """Calculate the stock's beta to the market."""
        # Calculate covariance between stock and market returns
        covariance = np.cov(self.stock_returns.values, self.market_returns.values)[0, 1]
        # Calculate market variance
        market_variance = np.var(self.market_returns.values)
        # Calculate beta
        if market_variance != 0:
            return covariance / market_variance
        return 1.0  # Default to 1.0 if market_variance is 0
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.stock_owned = 1  # Start with 1 unit of stock as specified
        # Ensure current_value is a float
        initial_stock_price = float(self.stock_data.iloc[0])
        self.current_value = float(self.cash_balance) + (float(self.stock_owned) * initial_stock_price)
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
        window_size = 25  # Increased window size to match expected dimensions
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
        observation = np.array(features)
        
        return observation
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): 0 = hold, 1 = buy, 2 = sell
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Get current price and execute action
        current_price = float(self.stock_data.iloc[self.current_step])
        
        # Execute trading action
        if action == 1:  # Buy
            if self.cash_balance >= current_price:  # Can only buy if we have enough cash
                self.stock_owned += 1
                self.cash_balance -= current_price
                self.trade_history.append(('buy', self.current_step, current_price))
        elif action == 2:  # Sell
            if self.stock_owned > 0:  # Can only sell if we own stock
                self.stock_owned -= 1
                self.cash_balance += current_price
                self.trade_history.append(('sell', self.current_step, current_price))
        # Action 0 is hold, so do nothing
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.stock_data) - 1
        
        # Calculate reward: daily profit + value of holdings
        next_price = self.stock_data.iloc[self.current_step]
        daily_profit = (next_price - current_price) * self.stock_owned
        portfolio_value = self.cash_balance + (self.stock_owned * next_price)
        reward = daily_profit + (portfolio_value - self.current_value)
        
        # Get next observation
        next_observation = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': portfolio_value,
            'cash_balance': self.cash_balance,
            'stock_owned': self.stock_owned,
            'current_price': next_price,
            'daily_profit': daily_profit
        }
        
        return next_observation, reward, done, info


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
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a neural network model for deep Q learning."""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for replay."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose an action based on the current state."""
        if training and np.random.rand() <= self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_size)
        
        # Exploitation: choose best action based on predicted Q-values
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])  # Return action with highest Q-value
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Q-learning formula: Q(s,a) = r + gamma * max(Q(s',a'))
                target = reward + self.gamma * np.amax(
                    self.model.predict(np.array([next_state]), verbose=0)[0]
                )
            
            # Update Q-value for the taken action
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            
            # Train the model
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        
        # Decay epsilon for less exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights from file."""
        self.model.load_weights(name)
    
    def save(self, name):
        """Save model weights to file."""
        self.model.save_weights(name)


def train_rl_agent(stock_data, market_data, episodes=50, batch_size=32):
    """
    Train a reinforcement learning agent on historical stock data.
    
    Args:
        stock_data (pd.Series): Historical stock price data
        market_data (pd.Series): Historical market benchmark data
        episodes (int): Number of training episodes
        batch_size (int): Batch size for experience replay
    
    Returns:
        tuple: (trained_agent, training_history)
    """
    print("======= Starting RL Agent Training =======")
    print(f"Stock: {stock_data.name if hasattr(stock_data, 'name') else 'Unknown'}")
    print(f"Market Benchmark: {market_data.name if hasattr(market_data, 'name') else 'Unknown'}")
    print(f"Data Points: {len(stock_data)}")
    print(f"Training Episodes: {episodes}")
    print(f"Batch Size: {batch_size}")
    print("==========================================")
    
    # Create environment
    env = StockTradingEnv(stock_data, market_data)
    print(f"Environment created with beta: {env.beta:.4f}")
    
    # Get state and action dimensions
    state = env.reset()
    state_size = len(state)
    action_size = 3  # hold, buy, sell
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    print(f"DQN Agent created with {state_size} state dimensions and {action_size} actions")
    
    # Training history
    history = {
        'episode_rewards': [],
        'portfolio_values': [],
        'action_counts': {0: 0, 1: 0, 2: 0}  # hold, buy, sell
    }
    
    # Track best performance
    best_reward = float('-inf')
    best_portfolio_value = 0
    initial_portfolio_value = 0
    
    # Progress indicators
    print("\nTraining progress:")
    
    # Training loop
    for episode in range(episodes):
        # Reset environment for new episode
        state = env.reset()
        total_reward = 0
        done = False
        episode_actions = {0: 0, 1: 0, 2: 0}  # Count actions in this episode
        step_count = 0
        
        # Store initial portfolio value in first episode
        if episode == 0:
            initial_portfolio_value = env.current_value
        
        # Episode loop
        while not done:
            # Choose action
            action = agent.act(state)
            episode_actions[action] += 1  # Track action counts
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Learn from experience
            agent.replay(batch_size)
        
        # Update action counts in history
        for action, count in episode_actions.items():
            history['action_counts'][action] += count
        
        # Record episode results
        history['episode_rewards'].append(total_reward)
        history['portfolio_values'].append(info['portfolio_value'])
        
        # Update best performance
        if total_reward > best_reward:
            best_reward = total_reward
        if info['portfolio_value'] > best_portfolio_value:
            best_portfolio_value = info['portfolio_value']
        
        # Calculate progress percentage
        progress = (episode + 1) / episodes * 100
        
        # Progress bar characters
        bar_length = 20
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Portfolio performance relative to initial value
        portfolio_change = ((info['portfolio_value'] - initial_portfolio_value) / 
                           initial_portfolio_value * 100)
        
        # Print progress with formatted output
        print(f"[{bar}] {progress:.1f}% | Episode: {episode+1}/{episodes} | "
              f"Reward: {total_reward:.2f} | Portfolio: ${info['portfolio_value']:.2f} "
              f"({portfolio_change:+.2f}%) | Epsilon: {agent.epsilon:.4f} | "
              f"Actions: Hold: {episode_actions[0]}, Buy: {episode_actions[1]}, Sell: {episode_actions[2]}")
    
    return agent, history


def evaluate_rl_agent(agent, stock_data, market_data):
    """
    Evaluate a trained RL agent on historical stock data.
    
    Args:
        agent (DQNAgent): Trained agent
        stock_data (pd.Series): Historical stock price data
        market_data (pd.Series): Historical market benchmark data
    
    Returns:
        dict: Evaluation metrics and trading decisions
    """
    print("\n======= Evaluating RL Agent Performance =======")
    print(f"Stock: {stock_data.name if hasattr(stock_data, 'name') else 'Unknown'}")
    print(f"Market Benchmark: {market_data.name if hasattr(market_data, 'name') else 'Unknown'}")
    print(f"Data Points: {len(stock_data)}")
    print("===============================================")
    
    # Create environment
    env = StockTradingEnv(stock_data, market_data)
    
    # Reset environment
    state = env.reset()
    done = False
    
    # Trading decisions and portfolio values over time
    decisions = []
    portfolio_values = [env.current_value]
    initial_value = env.current_value
    action_counts = {0: 0, 1: 0, 2: 0}  # hold, buy, sell
    cumulative_reward = 0
    
    # Evaluation loop
    step_count = 0
    total_steps = len(stock_data) - 1  # Approximate total steps
    print("\nEvaluation progress:")
    
    while not done:
        # Choose action (no exploration during evaluation)
        action = agent.act(state, training=False)
        action_counts[action] += 1
        
        # Take action
        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward
        
        # Record decision and portfolio value
        decisions.append({
            'date': env.stock_data.index[env.current_step],
            'action': ['hold', 'buy', 'sell'][action],
            'price': info['current_price'],
            'portfolio_value': info['portfolio_value'],
            'stock_owned': info['stock_owned'],
            'cash_balance': info['cash_balance']
        })
        portfolio_values.append(info['portfolio_value'])
        
        # Progress tracking
        step_count += 1
        if step_count % max(1, total_steps // 20) == 0 or done:  # Update approximately every 5% or at the end
            progress = min(100, step_count / total_steps * 100)
            bar_length = 20
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Calculate current portfolio performance
            current_value = info['portfolio_value']
            change_pct = (current_value - initial_value) / initial_value * 100
            
            # Print progress
            print(f"[{bar}] {progress:.1f}% | Step: {step_count}/{total_steps} | "
                  f"Current value: ${current_value:.2f} ({change_pct:+.2f}%) | "
                  f"Action: {['hold', 'buy', 'sell'][action]}")
        
        # Move to next state
        state = next_state
    
    # Calculate evaluation metrics
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Calculate buy and hold return for comparison
    buy_hold_return = (stock_data.iloc[-1] - stock_data.iloc[0]) / stock_data.iloc[0] * 100
    
    # Calculate Sharpe ratio (simplified)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
    
    # Return evaluation results
    return {
        'decisions': pd.DataFrame(decisions),
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'final_portfolio_value': final_value
    }


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download data
    stock_symbol = "AAPL"
    market_symbol = "QQQ"  # NASDAQ ETF
    
    stock_data = yf.Ticker(stock_symbol).history(period="1y")['Close']
    market_data = yf.Ticker(market_symbol).history(period="1y")['Close']
    
    # Train agent
    agent, history = train_rl_agent(stock_data, market_data, episodes=10)
    
    # Evaluate agent
    results = evaluate_rl_agent(agent, stock_data, market_data)
    
    print(f"RL Agent Total Return: {results['total_return']:.2f}%")
    print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
