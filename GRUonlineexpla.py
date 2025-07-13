"""
Online Contextual Bandit Stock Trading Model

This module implements a contextual bandit approach to stock trading using Thompson sampling
for exploration/exploitation balance. The model learns to predict future stock returns
based on sequential market data and makes buy/hold decisions while continuously adapting
to realized outcomes through online learning.

Key Components:
- StockDataset: Processes stock data into sequences for training
- ContextualBandit: Neural network model using Thompson sampling
- Online Learning: Continuous model updates based on realized rewards
- Evaluation: Backtesting with Thompson sampling decisions
"""

import pandas as pd
import numpy as np
import argparse
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


class StockDataset(Dataset):
    """
    Dataset class for preparing stock data sequences for contextual bandit training.
    
    This class transforms raw stock data into sequences that can be used for training
    a contextual bandit model. Each sequence represents a window of historical data
    that the model uses as context to make trading decisions.
    
    Key Features:
    - Creates sequences with configurable lookback periods
    - Calculates forward-looking rewards for each decision point
    - Handles feature scaling and normalization
    - Supports multiple stock tickers in a single dataset
    """
    
    def __init__(self, data, scaler=None, lookback_days=5):
        """
        Initialize the stock dataset.
        
        Args:
            data (pd.DataFrame): Raw stock data with columns including ticker, date, OHLC, etc.
            scaler (StandardScaler, optional): Pre-fitted scaler for feature normalization
            lookback_days (int): Number of historical days to include in each sequence
        """
        self.lookback_days = lookback_days
        
        # Define the feature columns that will be used as model inputs
        # These represent different aspects of market behavior and stock characteristics
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',  # Basic OHLC data
            'market_return',  # Broad market performance
            'stock_return',  # Individual stock performance
            'residual_return',  # Stock return minus market return
            'residual_gap_pct',  # Gap between expected and actual returns
            'residual_price',  # Price deviation from expected
            'volume_gap_product',  # Volume-based momentum indicator
            'is_earnings_date',  # Binary indicator for earnings announcements
            'earnings_classification',  # Type/quality of earnings
            'event_strength',  # Strength of market events
            'future_strength'  # Forward-looking strength indicator
        ]
        
        # Initialize storage for all sequences and their corresponding rewards
        all_sequences = []
        all_rewards = []
        
        # Process each stock ticker separately to maintain temporal consistency
        for ticker in data['ticker'].unique():
            # Extract and sort data for this specific ticker
            ticker_data = data[data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            
            # Prepare feature matrix by converting to numeric and handling missing values
            features_df = ticker_data[feature_cols].copy()
            for col in feature_cols:
                # Convert to numeric, replacing any non-numeric values with NaN
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            # Fill any remaining NaN values with 0 (conservative approach)
            features_df = features_df.fillna(0)
            
            # Create sequences with lookback window
            # Each sequence includes historical context plus current day
            for i in range(lookback_days, len(ticker_data)):
                # Extract sequence: lookback_days of history + current day
                # Shape: (lookback_days + 1, num_features)
                sequence = features_df.iloc[i-lookback_days:i+1].values
                
                # Calculate the reward for making a buy decision at this point
                current_row = ticker_data.iloc[i]
                
                # Look forward 60 days to calculate maximum possible return
                # This represents the "perfect" outcome if we bought at this point
                future_data = data[
                    (data['ticker'] == ticker) &
                    (data['date'] > current_row['date']) &
                    (data['date'] <= current_row['date'] + pd.Timedelta(days=60))
                ]
                
                if future_data.empty:
                    # No future data available, assign neutral reward
                    reward = 0.0
                else:
                    # Calculate maximum return achievable in the next 60 days
                    max_return = (future_data['close'].max() - current_row['close']) / current_row['close']
                    reward = max_return
                
                all_sequences.append(sequence)
                all_rewards.append(reward)
        
        # Convert lists to numpy arrays for efficient processing
        self.sequences = np.array(all_sequences)  # Shape: (N, lookback_days+1, num_features)
        
        # Prepare features for scaling
        # We need to scale each feature across all time steps consistently
        original_shape = self.sequences.shape
        # Flatten to (N * timesteps, num_features) for scaling
        flattened = self.sequences.reshape(-1, original_shape[-1])
        
        # Apply feature scaling to normalize inputs
        if scaler is None:
            # First time: fit scaler on training data
            self.scaler = StandardScaler()
            flattened_scaled = self.scaler.fit_transform(flattened)
        else:
            # Validation/test time: use pre-fitted scaler
            self.scaler = scaler
            flattened_scaled = self.scaler.transform(flattened)
        
        # Reshape back to original sequence structure
        self.features = flattened_scaled.reshape(original_shape)
        self.rewards = np.array(all_rewards)
        
        # Convert to PyTorch tensors for model training
        self.features = torch.FloatTensor(self.features)
        self.rewards = torch.FloatTensor(self.rewards)
    
    def _calculate_rewards(self, data):
        """
        Legacy method for reward calculation (now handled in __init__).
        Kept for backwards compatibility.
        """
        pass
    
    def __len__(self):
        """Return the total number of sequences in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get a single sequence and its corresponding reward.
        
        Args:
            idx (int): Index of the sequence to retrieve
            
        Returns:
            tuple: (features, reward) where features is a sequence tensor
        """
        return self.features[idx], self.rewards[idx]


class ContextualBandit(pl.LightningModule):
    """
    Neural network model implementing contextual bandit with Thompson sampling.
    
    This model learns to estimate the expected return and uncertainty of buying
    a stock given its historical context. It uses Thompson sampling to balance
    exploration (trying uncertain but potentially profitable opportunities) with
    exploitation (focusing on known profitable patterns).
    
    Key Features:
    - GRU-based sequence processing for temporal patterns
    - Outputs both mean and variance for uncertainty quantification
    - Thompson sampling for exploration/exploitation balance
    - Online learning capabilities for continuous adaptation
    """
    
    def __init__(self, input_dim=15, lookback_days=5, transaction_cost_pct=0.005, learning_rate=0.001):
        """
        Initialize the contextual bandit model.
        
        Args:
            input_dim (int): Number of input features per timestep
            lookback_days (int): Number of historical days in each sequence
            transaction_cost_pct (float): Trading cost as percentage (e.g., 0.005 = 0.5%)
            learning_rate (float): Learning rate for optimization
        """
        super().__init__()
        # Save hyperparameters for PyTorch Lightning checkpointing
        self.save_hyperparameters()
        
        # Store key parameters
        self.transaction_cost_pct = transaction_cost_pct
        self.learning_rate = learning_rate
        self.lookback_days = lookback_days
        
        # Neural network architecture using GRU for sequence processing
        # GRU is chosen over LSTM for efficiency with shorter sequences
        self.gru = nn.GRU(
            input_size=input_dim,  # Features per timestep
            hidden_size=32,        # Hidden state size
            batch_first=True       # Input shape: (batch, sequence, features)
        )
        
        # Output layers to predict mean and log-variance of expected returns
        # Log-variance is used for numerical stability
        self.output_layers = nn.Sequential(
            nn.Linear(32, 64),     # First hidden layer
            nn.ReLU(),             # Non-linear activation
            nn.Dropout(0.2),       # Regularization to prevent overfitting
            nn.Linear(64, 32),     # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)       # Output: [mean, log_variance]
        )
        
        # Alternative architectures (commented out for reference):
        
        # Option 2: LSTM for sequence processing
        # self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        # self.output_layers = nn.Sequential(...)
        
        # Option 3: Flatten sequence and use feedforward network
        # sequence_input_dim = input_dim * (lookback_days + 1)
        # self.network = nn.Sequential(...)
        
        # Optimizer for online learning (initialized after Lightning training)
        self.online_optimizer = None
    
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x (torch.Tensor): Input sequences of shape (batch, sequence_length, features)
            
        Returns:
            tuple: (mean, log_variance) predictions for expected returns
        """
        # Process sequence through GRU
        # gru_out: (batch, sequence, hidden_size)
        # hidden: (1, batch, hidden_size) - final hidden state
        gru_out, hidden = self.gru(x)
        
        # Use the last output from the sequence (most recent timestep)
        # This contains information from the entire sequence due to GRU's memory
        last_output = gru_out[:, -1, :]  # Shape: (batch, hidden_size)
        
        # Pass through output layers to get predictions
        output = self.output_layers(last_output)  # Shape: (batch, 2)
        
        # Alternative processing methods (commented out):
        
        # Option 2: LSTM processing
        # lstm_out, (hidden, cell) = self.lstm(x)
        # last_output = lstm_out[:, -1, :]
        # output = self.output_layers(last_output)
        
        # Option 3: Flatten and use feedforward
        # batch_size = x.shape[0]
        # flattened = x.view(batch_size, -1)
        # output = self.network(flattened)
        
        # Return mean and log-variance separately
        return output[:, 0], output[:, 1]  # mean, log_var
    
    def predict_action(self, context):
        """
        Predict buy/hold action using Thompson sampling.
        
        Thompson sampling works by:
        1. Predicting both expected return (mean) and uncertainty (variance)
        2. Sampling from the posterior distribution
        3. Making decisions based on the sampled value
        
        This naturally balances exploration (high uncertainty → more sampling variation)
        with exploitation (low uncertainty → consistent decisions).
        
        Args:
            context (torch.Tensor): Input sequence for decision making
            
        Returns:
            tuple: (action, predicted_mean, predicted_std)
                - action: 1.0 for buy, 0.0 for hold
                - predicted_mean: Expected return estimate
                - predicted_std: Uncertainty estimate
        """
        # Set model to evaluation mode (disables dropout, etc.)
        self.eval()
        with torch.no_grad():  # No gradient computation needed for inference
            # Ensure proper batch dimension
            if len(context.shape) == 2:  # Single sequence: (seq_len, features)
                context = context.unsqueeze(0)  # Add batch dim: (1, seq_len, features)
            
            # Get predictions from the model
            mean, log_var = self.forward(context)
            
            # Convert log-variance to standard deviation
            # Using log-variance for numerical stability during training
            std = torch.exp(0.5 * log_var)
            
            # Thompson sampling: sample from the posterior distribution
            # This adds exploration by sampling from N(mean, std^2)
            sampled_reward = mean + std * torch.randn_like(std)
            
            # Make buy decision if sampled reward exceeds threshold
            # Threshold includes minimum return expectation + transaction costs
            buy_threshold = 0.5 + self.transaction_cost_pct  # 0.5% + transaction cost
            action = (sampled_reward > buy_threshold).float()
            
        # Remove batch dimension for single predictions
        return action.squeeze(), mean.squeeze(), std.squeeze()
    
    def setup_online_learning(self):
        """
        Initialize optimizer for online learning phase.
        
        This is called after the initial Lightning training is complete.
        We use a separate optimizer with a lower learning rate for online updates
        to prevent catastrophic forgetting of the initial training.
        """
        # Lower learning rate for online updates to maintain stability
        self.online_optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate * 0.1  # 10% of original learning rate
        )
    
    def online_update(self, context, action, reward):
        """
        Update model parameters based on realized trading outcomes.
        
        This implements the online learning component where the model learns
        from the actual results of its Thompson sampling decisions.
        
        Args:
            context (torch.Tensor): The context that led to the decision
            action (float): The action taken (1.0 for buy, 0.0 for hold)
            reward (torch.Tensor): The realized reward from the action
            
        Returns:
            float: Loss value for monitoring (0.0 if no update performed)
        """
        # Initialize online optimizer if not already done
        if self.online_optimizer is None:
            self.setup_online_learning()
        
        # Set model to training mode for parameter updates
        self.train()
        self.online_optimizer.zero_grad()
        
        # Ensure proper batch dimension
        if len(context.shape) == 2:  # Single sequence
            context = context.unsqueeze(0)  # Add batch dimension
        
        # Only update parameters if we actually took the buy action
        # This is because we only observe rewards for actions we took
        if action > 0.5:  # Buy action was taken
            # Get current model predictions for this context
            mean, log_var = self.forward(context)
            
            # Calculate negative log-likelihood loss
            # This updates the model to better predict the realized reward
            var = torch.exp(log_var)  # Convert log-variance to variance
            
            # Negative log-likelihood of a Gaussian distribution
            # This loss decreases when the model predicts closer to the realized reward
            loss = 0.5 * (log_var + (reward - mean) ** 2 / var).mean()
            
            # Backpropagate and update parameters
            loss.backward()
            self.online_optimizer.step()
            
            return loss.item()
        
        # No update performed for hold actions
        return 0.0
    
    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.
        
        This is used during the initial model training phase to learn
        from historical data before online deployment.
        
        Args:
            batch (tuple): (contexts, rewards) batch from DataLoader
            batch_idx (int): Batch index (unused but required by Lightning)
            
        Returns:
            torch.Tensor: Loss value for this batch
        """
        contexts, rewards = batch
        mean, log_var = self.forward(contexts)
        
        # Negative log-likelihood loss for Gaussian distribution
        # This trains the model to predict both mean and variance accurately
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + (rewards - mean) ** 2 / var).mean()
        
        # Log metrics for Lightning's progress tracking
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning.
        
        Used to monitor model performance on held-out data during training.
        
        Args:
            batch (tuple): (contexts, rewards) batch from validation DataLoader
            batch_idx (int): Batch index (unused but required by Lightning)
            
        Returns:
            torch.Tensor: Validation loss for this batch
        """
        contexts, rewards = batch
        mean, log_var = self.forward(contexts)
        
        # Same loss calculation as training
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + (rewards - mean) ** 2 / var).mean()
        
        # Log validation metrics
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer for PyTorch Lightning training phase.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer for initial training
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def load_data(tickers):
    """
    Load and combine stock data from parquet files.
    
    This function searches for the most recent feature files for each ticker
    and combines them into a single dataset for training/evaluation.
    
    Args:
        tickers (list): List of stock ticker symbols to load
        
    Returns:
        pd.DataFrame: Combined stock data sorted by date and ticker
        
    Raises:
        ValueError: If no data files are found for any ticker
    """
    all_data = []
    
    # Load data for each ticker
    for ticker in tickers:
        # Find all feature files for this ticker
        files = glob.glob(f"data/{ticker}_features_*.parquet")
        
        if files:
            # Use the most recently created file (latest data)
            latest_file = max(files, key=os.path.getctime)
            df = pd.read_parquet(latest_file)
            all_data.append(df)
            print(f"Loaded {ticker}: {len(df)} rows")
        else:
            print(f"Warning: No data files found for {ticker}")
    
    if not all_data:
        raise ValueError("No data files found for any ticker!")
    
    # Combine all ticker data
    data = pd.concat(all_data, ignore_index=True)
    
    # Ensure date column is properly formatted
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort chronologically within each ticker for proper sequence creation
    data = data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    print(f"Total combined data: {len(data)} rows")
    return data


def train_model(data, transaction_cost_pct=0.005, lookback_days=5):
    """
    Train the contextual bandit model using historical data.
    
    This function performs the initial training phase where the model learns
    from historical patterns before being deployed for online learning.
    
    Args:
        data (pd.DataFrame): Stock data for training
        transaction_cost_pct (float): Trading cost percentage
        lookback_days (int): Number of days to look back for context
        
    Returns:
        tuple: (trained_model, fitted_scaler)
    """
    # Chronological split to prevent look-ahead bias
    # This ensures the model can't learn from future information
    split_idx = int(0.8 * len(data))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    
    # Create datasets with sequence processing
    train_dataset = StockDataset(train_data, lookback_days=lookback_days)
    # Use the same scaler for validation to ensure consistent normalization
    val_dataset = StockDataset(val_data, scaler=train_dataset.scaler, lookback_days=lookback_days)
    
    # Create data loaders for batch processing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,    # Batch size for efficient GPU utilization
        shuffle=True       # Shuffle for better training dynamics
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256,    # Larger batch size for validation (no gradient computation)
        shuffle=False      # No need to shuffle validation data
    )
    
    # Determine input dimension from feature columns
    # This ensures the model architecture matches the data
    _feature_cols_list = [
        'open', 'high', 'low', 'close', 'volume', 'market_return',
        'stock_return', 'residual_return', 'residual_gap_pct', 'residual_price',
        'volume_gap_product', 'is_earnings_date', 'earnings_classification', 
        'event_strength', 'future_strength'
    ]
    input_dim = len(_feature_cols_list)
    
    print(f"Model input dimension: {input_dim} features")
    
    # Initialize the model with appropriate parameters
    model = ContextualBandit(
        input_dim=input_dim, 
        lookback_days=lookback_days,
        transaction_cost_pct=transaction_cost_pct
    )
    
    # Set up PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=50,              # Number of training epochs
        enable_progress_bar=True,   # Show training progress
        accelerator="cpu"           # Use CPU (can be changed to "gpu" if available)
    )
    
    print("Starting model training...")
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    print("Model training completed!")
    
    return model, train_dataset.scaler


def evaluate_model_online(model, data, scaler, tickers):
    """
    Evaluate model performance with online learning from Thompson sampling decisions.
    
    This function simulates real-world trading where:
    1. The model makes decisions based on Thompson sampling
    2. Decisions are recorded with their context
    3. After sufficient time passes, realized rewards are calculated
    4. The model is updated based on these realized outcomes
    
    This creates a realistic simulation of how the model would perform in production.
    
    Args:
        model: Trained contextual bandit model
        data (pd.DataFrame): Full dataset for evaluation
        scaler: Fitted feature scaler
        tickers (list): List of stock tickers being evaluated
        
    Returns:
        dict: Buy events for each ticker with decision details
    """
    # Create dataset for evaluation
    dataset = StockDataset(data, scaler=scaler, lookback_days=model.lookback_days)
    
    # Initialize tracking structures
    buy_events = {ticker: [] for ticker in tickers}  # Store all buy decisions
    online_losses = []  # Track online learning losses
    
    # Sort data chronologically for proper online learning simulation
    data_sorted = data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # Pending updates: decisions waiting for realized rewards
    # In real trading, we need to wait for time to pass to see actual outcomes
    pending_updates = []
    
    # Track buy events by unique ID for later reward updates
    buy_event_ids = {}
    
    print("Starting online evaluation...")
    
    # Iterate through each data point chronologically
    for idx, (context, true_reward) in enumerate(dataset):
        # Skip early data points where we don't have enough history
        if idx < model.lookback_days:
            continue
            
        # Get the current market data point
        row = data_sorted.iloc[idx + model.lookback_days]  # Adjust for sequence indexing
        current_date = row['date']
        
        # Process pending updates where enough time has passed to calculate realized rewards
        updates_to_process = []
        remaining_updates = []
        
        for pending in pending_updates:
            pending_date = pending['date']
            days_elapsed = (current_date - pending_date).days
            
            # Check if 60 days have passed (our reward calculation window)
            if days_elapsed >= 60:
                # Enough time has passed - calculate realized reward
                ticker = pending['ticker']
                entry_price = pending['entry_price']
                
                # Find the maximum price achieved in the 60 days after the buy decision
                # This represents the best outcome if we had perfect exit timing
                future_data = data_sorted[
                    (data_sorted['ticker'] == ticker) &
                    (data_sorted['date'] > pending_date) &
                    (data_sorted['date'] <= pending_date + pd.Timedelta(days=60))
                ]
                
                if not future_data.empty:
                    max_price = future_data['close'].max()
                    realized_reward = (max_price - entry_price) / entry_price
                else:
                    realized_reward = 0.0
                
                pending['realized_reward'] = realized_reward
                updates_to_process.append(pending)
                
                # Update the corresponding buy event record
                event_id = f"{ticker}_{pending_date}"
                if event_id in buy_event_ids:
                    ticker_events = buy_events[ticker]
                    event_idx = buy_event_ids[event_id]
                    if event_idx < len(ticker_events):
                        ticker_events[event_idx]['realized_reward'] = realized_reward
            else:
                # Not enough time has passed - keep in pending list
                remaining_updates.append(pending)
        
        # Perform online model updates with realized rewards
        for update in updates_to_process:
            # Update model parameters based on actual trading outcome
            loss = model.online_update(
                update['context'], 
                update['action'], 
                torch.tensor(update['realized_reward'])
            )
            if loss > 0:
                online_losses.append(loss)
        
        # Keep only updates that haven't matured yet
        pending_updates = remaining_updates
        
        # Make Thompson sampling decision for current timestep
        # This is where the model decides whether to buy or hold
        action, pred_mean, pred_std = model.predict_action(context)
        
        # Record buy decisions
        if action.item() > 0.5:  # Buy decision made
            # Create unique identifier for this buy event
            event_id = f"{row['ticker']}_{row['date']}"
            
            # Record the buy decision with all relevant information
            buy_event = {
                'date': row['date'],
                'price': row['close'],
                'actual_return': true_reward.item(),      # True future return (for reference)
                'predicted_return': pred_mean.item(),     # Model's prediction
                'uncertainty': pred_std.item(),           # Model's uncertainty
                'realized_reward': None,                  # Will be updated later
                'event_id': event_id
            }
            
            buy_events[row['ticker']].append(buy_event)
            buy_event_ids[event_id] = len(buy_events[row['ticker']]) - 1
            
            # Add to pending updates for future online learning
            pending_updates.append({
                'context': context,
                'action': action,
                'date': current_date,
                'ticker': row['ticker'],
                'entry_price': row['close']
            })
    
    # Print summary statistics
    total_buys = sum(len(events) for events in buy_events.values())
    print(f"Online evaluation completed!")
    print(f"Total buy decisions made: {total_buys}")
    print(f"Online learning updates performed: {len(online_losses)}")
    if online_losses:
        print(f"Average online learning loss: {np.mean(online_losses):.4f}")
    print(f"Pending updates at end: {len(pending_updates)}")
    
    return buy_events


def visualize_results(buy_events, data, tickers):
    """
    Create visualizations of trading decisions and outcomes.
    
    This function generates plots showing:
    - Stock price movements over time
    - Buy decision points colored by realized returns
    - Pending decisions (where outcomes aren't yet known)
    
    Args:
        buy_events (dict): Buy decisions for each ticker
        data (pd.DataFrame): Stock price data
        tickers (list): List of stock tickers
    """
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Create subplots for each ticker
    fig, axes = plt.subplots(len(tickers), 1, figsize=(15, 4*len(tickers)))
    if len(tickers) == 1:
        axes = [axes]  # Ensure axes is always a list
    
    for i, ticker in enumerate(tickers):
        # Plot stock price over time
        ticker_data = data[data['ticker'] == ticker].sort_values('date')
        axes[i].plot(ticker_data['date'], ticker_data['close'], 
                    label=f'{ticker} Price', alpha=0.7, linewidth=1)
        
        # Plot buy events
        events = buy_events[ticker]
        if events:
            # Separate events with realized rewards from pending events
            realized_events = [e for e in events if e['realized_reward'] is not None]
            unrealized_events = [e for e in events if e['realized_reward'] is None]
            
            # Plot events with realized rewards (colored by performance)
            if realized_events:
                dates = [e['date'] for e in realized_events]
                prices = [e['price'] for e in realized_events]
                returns = [e['realized_reward'] for e in realized_events]
                
                # Color-coded scatter plot: green for profitable, red for unprofitable
                scatter = axes[i].scatter(dates, prices, c=returns, cmap='RdYlGn', 
                                        s=100, alpha=0.8, edgecolors='black',
                                        label=f'Buy Decisions (Realized)')
                # Add colorbar to show return scale
                plt.colorbar(scatter, ax=axes[i], label='Realized 60-day Return')
            
            # Plot pending events (no realized rewards yet)
            if unrealized_events:
                dates = [e['date'] for e in unrealized_events]
                prices = [e['price'] for e in unrealized_events]
                
                axes[i].scatter(dates, prices, c='gray', 
                               s=100, alpha=0.5, edgecolors='black', marker='x',
                               label='Pending (No realized return yet)')
        
        # Formatting
        axes[i].set_title(f'{ticker} - Buy Decisions (Online Learning)', fontsize=14)
        axes[i].set_ylabel('Price ($)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Add some statistics as text
        if events:
            total_events = len(events)
            realized_events = [e for e in events if e['realized_reward'] is not None]
            if realized_events:
                avg_return = np.mean([e['realized_reward'] for e in realized_events])
                positive_rate = np.mean([e['realized_reward'] > 0 for e in realized_events])
                stats_text = f'Events: {total_events} | Avg Return: {avg_return:.1%} | Success Rate: {positive_rate:.1%}'
                axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/online_buy_decisions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Visualization saved to 'plots/online_buy_decisions.png'")


def print_summary(buy_events, tickers):
    """
    Print comprehensive performance summary for all tickers.
    
    This function analyzes the trading performance and provides key metrics
    including total returns, success rates, and per-ticker breakdowns.
    
    Args:
        buy_events (dict): Buy decisions for each ticker
        tickers (list): List of stock tickers analyzed
    """
    print("\n" + "="*60)
    print("    ONLINE CONTEXTUAL BANDIT PERFORMANCE SUMMARY")
    print("="*60)
    
    # Per-ticker analysis
    all_returns = []  # Collect all returns for overall statistics
    total_trades = 0
    
    for ticker in tickers:
        events = buy_events[ticker]
        
        if events:
            # Calculate metrics for realized returns only
            realized_events = [e for e in events if e['realized_reward'] is not None]
            
            if realized_events:
                returns = [e['realized_reward'] for e in realized_events]
                avg_return = np.mean(returns)
                positive_pct = (np.array(returns) > 0).mean() * 100
                max_return = max(returns)
                min_return = min(returns)
                total_return = sum(returns)
                realized_trades = len(realized_events)
            else:
                avg_return = positive_pct = max_return = min_return = total_return = 0
                realized_trades = 0
            
            pending_trades = len(events) - realized_trades
            total_buys = len(events)
            
            all_returns.extend([e['realized_reward'] for e in events if e['realized_reward'] is not None])
            total_trades += total_buys
        else:
            avg_return = positive_pct = total_buys = realized_trades = pending_trades = 0
            max_return = min_return = total_return = 0
        
        # Print detailed ticker information
        print(f"\n{ticker:>6}:")
        print(f"  Total Decisions:     {total_buys:>3}")
        print(f"  Realized Outcomes:   {realized_trades:>3}")
        print(f"  Pending Outcomes:    {pending_trades:>3}")
        if realized_trades > 0:
            print(f"  Average Return:      {avg_return:>6.1%}")
            print(f"  Success Rate:        {positive_pct:>5.1f}%")
            print(f"  Best Trade:          {max_return:>6.1%}")
            print(f"  Worst Trade:         {min_return:>6.1%}")
            print(f"  Cumulative Return:   {total_return:>6.1%}")
        else:
            print(f"  No realized outcomes yet")
    
    # Overall portfolio statistics
    print(f"\n{'PORTFOLIO SUMMARY':>20}")
    print("-" * 40)
    print(f"Total Trading Decisions: {total_trades:>6}")
    print(f"Decisions with Outcomes: {len(all_returns):>6}")
    
    if all_returns:
        portfolio_return = sum(all_returns)
        avg_trade_return = np.mean(all_returns)
        success_rate = (np.array(all_returns) > 0).mean() * 100
        volatility = np.std(all_returns)
        
        print(f"Total Portfolio Return:  {portfolio_return:>6.1%}")
        print(f"Average Trade Return:    {avg_trade_return:>6.1%}")
        print(f"Overall Success Rate:    {success_rate:>5.1f}%")
        print(f"Return Volatility:       {volatility:>6.1%}")
        
        # Risk-adjusted metrics
        if volatility > 0:
            sharpe_ratio = avg_trade_return / volatility
            print(f"Sharpe Ratio:            {sharpe_ratio:>6.2f}")
        
        # Additional statistics
        profitable_trades = [r for r in all_returns if r > 0]
        losing_trades = [r for r in all_returns if r < 0]
        
        if profitable_trades:
            avg_win = np.mean(profitable_trades)
            print(f"Average Winning Trade:   {avg_win:>6.1%}")
        
        if losing_trades:
            avg_loss = np.mean(losing_trades)
            print(f"Average Losing Trade:    {avg_loss:>6.1%}")
            
            if profitable_trades and losing_trades:
                win_loss_ratio = avg_win / abs(avg_loss)
                print(f"Win/Loss Ratio:          {win_loss_ratio:>6.2f}")
    else:
        print("No completed trades to analyze yet.")
    
    print("\n" + "="*60)


def main():
    """
    Main function that orchestrates the entire contextual bandit trading pipeline.
    
    This function:
    1. Parses command line arguments
    2. Loads and prepares stock data
    3. Trains the initial contextual bandit model
    4. Runs online evaluation with Thompson sampling
    5. Visualizes results and prints performance summary
    
    The pipeline simulates realistic trading conditions where the model
    learns continuously from its own decisions.
    """
    # Force CPU usage (remove CUDA to avoid GPU memory issues)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Online Contextual Bandit Stock Trading Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bandit_model.py AAPL MSFT --transaction_cost 0.005 --lookback_days 5
  python bandit_model.py TSLA NVDA --transaction_cost 0.01 --lookback_days 10
        """
    )
    
    # Required arguments
    parser.add_argument("tickers", nargs="+", 
                        help="Stock tickers to process (e.g., AAPL MSFT TSLA)")
    
    # Optional arguments with defaults
    parser.add_argument("--transaction_cost", type=float, default=0.005, 
                        help="Transaction cost as a percentage (e.g., 0.005 for 0.5%%) [default: 0.5%%]")
    parser.add_argument("--lookback_days", type=int, default=5,
                        help="Number of days to look back for context [default: 5]")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print configuration
    print("="*80)
    print("CONTEXTUAL BANDIT STOCK TRADING MODEL")
    print("="*80)
    print(f"Target Tickers:          {', '.join(args.tickers)}")
    print(f"Transaction Cost:        {args.transaction_cost:.3%}")
    print(f"Lookback Period:         {args.lookback_days} days")
    print(f"Learning Algorithm:      Thompson Sampling with Online Updates")
    print("="*80)
    
    try:
        # Step 1: Load and prepare data
        print("\n📊 STEP 1: Loading stock data...")
        data = load_data(args.tickers)
        print(f"✅ Data loaded successfully: {len(data):,} total observations")
        
        # Step 2: Train initial model
        print("\n🧠 STEP 2: Training initial contextual bandit model...")
        print("This may take several minutes depending on data size...")
        model, scaler = train_model(
            data, 
            transaction_cost_pct=args.transaction_cost, 
            lookback_days=args.lookback_days
        )
        print("✅ Initial model training completed!")
        
        # Step 3: Setup online learning capabilities
        print("\n⚙️  STEP 3: Setting up online learning...")
        model.setup_online_learning()
        print("✅ Online learning initialized with reduced learning rate")
        
        # Step 4: Run online evaluation with Thompson sampling
        print("\n📈 STEP 4: Running online evaluation with Thompson sampling...")
        print("Simulating real-world trading with continuous learning...")
        buy_events = evaluate_model_online(model, data, scaler, args.tickers)
        print("✅ Online evaluation completed!")
        
        # Step 5: Generate visualizations
        print("\n📊 STEP 5: Generating visualizations...")
        visualize_results(buy_events, data, args.tickers)
        print("✅ Visualizations created!")
        
        # Step 6: Print comprehensive summary
        print("\n📋 STEP 6: Performance Analysis...")
        print_summary(buy_events, args.tickers)
        
        print("\n🎉 Analysis complete! Check the 'plots' directory for visualizations.")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find required data files.")
        print(f"   Make sure you have data files in the format: data/{{ticker}}_features_*.parquet")
        print(f"   Details: {e}")
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print(f"   Please check your data files and try again.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Process interrupted by user.")
        
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        print(f"   Please check your input data and parameters.")


if __name__ == "__main__":
    """
    Entry point for the script.
    
    This allows the script to be run from the command line while also
    being importable as a module for use in other scripts.
    """
    main()