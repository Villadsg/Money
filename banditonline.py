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
    def __init__(self, data, scaler=None, lookback_days=5):
        self.lookback_days = lookback_days
        
        # Select features
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'market_return',
            'stock_return', 'residual_return', 'residual_gap_pct', 'residual_price',
            'volume_gap_product', 'is_earnings_date', 'earnings_classification', 
            'event_strength', 'future_strength'
        ]
        
        # Create sequences for each ticker
        all_sequences = []
        all_rewards = []
        
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            
            # Convert features to numeric
            features_df = ticker_data[feature_cols].copy()
            for col in feature_cols:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            features_df = features_df.fillna(0)
            
            # Create sequences with lookback
            for i in range(lookback_days, len(ticker_data)):
                # Get lookback_days + current day features (e.g., 6 days total for 5-day lookback)
                sequence = features_df.iloc[i-lookback_days:i+1].values  # Shape: (6, 15)
                
                # Calculate reward for this decision point
                current_row = ticker_data.iloc[i]
                future_data = data[
                    (data['ticker'] == ticker) &
                    (data['date'] > current_row['date']) &
                    (data['date'] <= current_row['date'] + pd.Timedelta(days=30))
                ]
                
                if future_data.empty:
                    reward = 0.0
                else:
                    max_return = (future_data['close'].max() - current_row['close']) / current_row['close']
                    reward = max_return
                
                all_sequences.append(sequence)
                all_rewards.append(reward)
        
        # Convert to arrays
        self.sequences = np.array(all_sequences)  # Shape: (N, lookback_days+1, num_features)
        
        # Flatten sequences for scaling (each timestep scaled independently)
        original_shape = self.sequences.shape
        flattened = self.sequences.reshape(-1, original_shape[-1])
        
        # Scale features
        if scaler is None:
            self.scaler = StandardScaler()
            flattened_scaled = self.scaler.fit_transform(flattened)
        else:
            self.scaler = scaler
            flattened_scaled = self.scaler.transform(flattened)
        
        # Reshape back to sequences
        self.features = flattened_scaled.reshape(original_shape)
        self.rewards = np.array(all_rewards)
        
        self.features = torch.FloatTensor(self.features)
        self.rewards = torch.FloatTensor(self.rewards)
    
    def _calculate_rewards(self, data):
        # This method is no longer used since rewards are calculated in __init__
        pass
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.rewards[idx]

class ContextualBandit(pl.LightningModule):
    def __init__(self, input_dim=15, lookback_days=5, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for Lightning
        
        self.learning_rate = learning_rate
        self.lookback_days = lookback_days
        
        # Option 1: GRU for sequence processing (recommended for shorter sequences)
        self.gru = nn.GRU(input_dim, 32, batch_first=True)
        self.output_layers = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # [mean, log_var] for Thompson sampling
        )
        
        
        # For online learning - will be set after Lightning training
        self.online_optimizer = None
    
    def forward(self, x):
        # Option 1: GRU processing (recommended)
        gru_out, hidden = self.gru(x)
        last_output = gru_out[:, -1, :]  # Use last output from sequence
        output = self.output_layers(last_output)
        
        
        return output[:, 0], output[:, 1]  # mean, log_var
    
    def predict_action(self, context):
        """Predict buy/hold using Thompson sampling"""
        self.eval()
        with torch.no_grad():
            if len(context.shape) == 2:  # Single sequence: (seq_len, features)
                context = context.unsqueeze(0)  # Add batch dim: (1, seq_len, features)
            
            mean, log_var = self.forward(context)
            std = torch.exp(0.5 * log_var)
            
            # Sample from posterior (Thompson sampling) 
            sampled_reward = mean + std * torch.randn_like(std)
            
            # Buy if predicted reward > (1% 
            action = (sampled_reward > (0.2)).float()
            
        return action.squeeze(), mean.squeeze(), std.squeeze()
    
    def setup_online_learning(self):
        """Initialize optimizer for online learning after Lightning training is complete"""
        self.online_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate*10)
    
    def online_update(self, context, action, reward):
        """Update model based on realized reward from Thompson sampling decision"""
        if self.online_optimizer is None:
            self.setup_online_learning()
        
        self.train()
        self.online_optimizer.zero_grad()
        
        if len(context.shape) == 2:  # Single sequence
            context = context.unsqueeze(0)  # Add batch dimension
        
        # Only update if we took the buy action
        if action > 0.5:
            mean, log_var = self.forward(context)
            
            # Negative log-likelihood loss for the realized reward
            var = torch.exp(log_var)
            loss = 0.5 * (log_var + (reward - mean) ** 2 / var).mean()
            
            loss.backward()
            self.online_optimizer.step()
            
            return loss.item()
        
        return 0.0
    
    def training_step(self, batch, batch_idx):
        contexts, rewards = batch
        mean, log_var = self.forward(contexts)
        
        # Negative log-likelihood loss
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + (rewards - mean) ** 2 / var).mean()
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        contexts, rewards = batch
        mean, log_var = self.forward(contexts)
        
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + (rewards - mean) ** 2 / var).mean()
        
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer for Lightning training phase"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def load_data(tickers):
    """Load and combine stock data"""
    all_data = []
    
    for ticker in tickers:
        files = glob.glob(f"data/{ticker}_features_*.parquet")
        if files:
            latest_file = max(files, key=os.path.getctime)
            df = pd.read_parquet(latest_file)
            all_data.append(df)
            print(f"Loaded {ticker}: {len(df)} rows")
    
    if not all_data:
        raise ValueError("No data files found!")
    
    # Combine and sort
    data = pd.concat(all_data, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    print(f"Total data: {len(data)} rows")
    return data

def train_model(data, lookback_days=5):
    """Train the contextual bandit"""
    # Split chronologically
    split_idx = int(0.8 * len(data))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Create datasets
    train_dataset = StockDataset(train_data, lookback_days=lookback_days)
    val_dataset = StockDataset(val_data, scaler=train_dataset.scaler, lookback_days=lookback_days)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # Determine input_dim from feature columns
    _feature_cols_list = [
        'open', 'high', 'low', 'close', 'volume', 'market_return',
        'stock_return', 'residual_return', 'residual_gap_pct', 'residual_price',
        'volume_gap_product', 'is_earnings_date', 'earnings_classification', 
        'event_strength', 'future_strength'
    ]
    input_dim = len(_feature_cols_list)

    # Train model
    model = ContextualBandit(input_dim=input_dim, 
                             lookback_days=lookback_days)
    trainer = pl.Trainer(max_epochs=50, enable_progress_bar=True, accelerator="cpu")
    trainer.fit(model, train_loader, val_loader)
    
    return model, train_dataset.scaler

def evaluate_model_online(model, data, scaler, tickers):
    """Evaluate model with online learning from Thompson sampling decisions"""
    dataset = StockDataset(data, scaler=scaler, lookback_days=model.lookback_days)
    buy_events = {ticker: [] for ticker in tickers}
    online_losses = []
    
    # Sort data chronologically for proper online learning
    data_sorted = data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # Store pending updates (context, action) waiting for realized rewards
    pending_updates = []
    # Store buy event IDs to update realized rewards later
    buy_event_ids = {}
    
    for idx, (context, true_reward) in enumerate(dataset):
        # Skip if we don't have enough lookback data yet
        if idx < model.lookback_days:
            continue
            
        row = data_sorted.iloc[idx + model.lookback_days]  # Adjust for sequence indexing
        current_date = row['date']
        
        # Process any pending updates where we now have realized rewards (30+ days later)
        updates_to_process = []
        remaining_updates = []
        
        for pending in pending_updates:
            pending_date = pending['date']
            days_elapsed = (current_date - pending_date).days
            
            if days_elapsed >=30:  # Reward is now realized
                # Calculate actual reward for this past decision
                ticker = pending['ticker']
                entry_price = pending['entry_price']
                
                # Find the best price achieved in the 30 days after the buy decision
                future_data = data_sorted[
                    (data_sorted['ticker'] == ticker) &
                    (data_sorted['date'] > pending_date) &
                    (data_sorted['date'] <= pending_date + pd.Timedelta(days=30))
                ]
                
                if not future_data.empty:
                    max_price = future_data['close'].max()
                    realized_reward = (max_price - entry_price) / entry_price
                else:
                    realized_reward = 0.0
                
                pending['realized_reward'] = realized_reward
                updates_to_process.append(pending)
                
                # Update the corresponding buy event with the realized reward
                event_id = f"{ticker}_{pending_date}"
                if event_id in buy_event_ids:
                    ticker_events = buy_events[ticker]
                    event_idx = buy_event_ids[event_id]
                    if event_idx < len(ticker_events):
                        ticker_events[event_idx]['realized_reward'] = realized_reward
            else:
                remaining_updates.append(pending)
        
        # Update model parameters with realized rewards
        for update in updates_to_process:
            loss = model.online_update(
                update['context'], 
                update['action'], 
                torch.tensor(update['realized_reward'])
            )
            if loss > 0:
                online_losses.append(loss)
        
        # Keep only pending updates that haven't matured yet
        pending_updates = remaining_updates
        
        # Make Thompson sampling decision for current timestep
        action, pred_mean, pred_std = model.predict_action(context)
        
        if action.item() > 0.5:  # Buy decision
            # Add buy event with a unique ID to track it
            event_id = f"{row['ticker']}_{row['date']}"
            buy_events[row['ticker']].append({
                'date': row['date'],
                'price': row['close'],
                'actual_return': true_reward.item(),  # Keep for reference
                'predicted_return': pred_mean.item(),
                'uncertainty': pred_std.item(),
                'realized_reward': None,  # Will be updated later
                'event_id': event_id
            })
            buy_event_ids[event_id] = len(buy_events[row['ticker']]) - 1
            
            # Add to pending updates - we'll learn from this decision in 30 days
            pending_updates.append({
                'context': context,
                'action': action,
                'date': current_date,
                'ticker': row['ticker'],
                'entry_price': row['close']
            })
    
    print(f"Online learning updates: {len(online_losses)} (avg loss: {np.mean(online_losses):.4f})")
    print(f"Pending updates at end: {len(pending_updates)}")
    return buy_events

def visualize_results(buy_events, data, tickers):
    """Create visualizations"""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(len(tickers), 1, figsize=(15, 4*len(tickers)))
    if len(tickers) == 1:
        axes = [axes]
    
    for i, ticker in enumerate(tickers):
        # Plot stock price
        ticker_data = data[data['ticker'] == ticker].sort_values('date')
        axes[i].plot(ticker_data['date'], ticker_data['close'], 
                    label=f'{ticker} Price', alpha=0.7)
        
        # Plot buy events
        events = buy_events[ticker]
        if events:
            # Filter events that have realized rewards
            realized_events = [e for e in events if e['realized_reward'] is not None]
            unrealized_events = [e for e in events if e['realized_reward'] is None]
            
            # Plot events with realized rewards
            if realized_events:
                dates = [e['date'] for e in realized_events]
                prices = [e['price'] for e in realized_events]
                returns = [e['realized_reward'] for e in realized_events]
                
                scatter = axes[i].scatter(dates, prices, c=returns, cmap='RdYlGn', 
                                        s=100, alpha=0.8, edgecolors='black')
                plt.colorbar(scatter, ax=axes[i], label='Realized 60-day Return')
            
            # Plot events without realized rewards (pending) in gray
            if unrealized_events:
                dates = [e['date'] for e in unrealized_events]
                prices = [e['price'] for e in unrealized_events]
                
                axes[i].scatter(dates, prices, c='gray', 
                               s=100, alpha=0.5, edgecolors='black', marker='x',
                               label='Pending (No realized return yet)')
        
        axes[i].set_title(f'{ticker} - Buy Decisions (Online Learning)')
        axes[i].set_ylabel('Price ($)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('plots/online_buy_decisions.png', dpi=300)
    plt.show()

def print_summary(buy_events, tickers):
    """Print performance summary"""
    print("\n=== ONLINE CONTEXTUAL BANDIT PERFORMANCE ===")
    
    for ticker in tickers:
        events = buy_events[ticker]
        
        if events:
            returns = [e['actual_return'] for e in events]
            avg_return = np.mean(returns)
            positive_pct = (np.array(returns) > 0).mean() * 100
            total_buys = len(events)
        else:
            avg_return = positive_pct = total_buys = 0
        
        print(f"{ticker:>6}: {total_buys:>3} buys, "
              f"{avg_return:>6.1%} avg return, "
              f"{positive_pct:>5.1f}% positive")
    
    # Overall stats
    all_returns = [e['actual_return'] for events in buy_events.values() for e in events]
    if all_returns:
        total_return = sum(all_returns)
        print(f"\nTotal portfolio return: {total_return:.1%}")
        print(f"Average trade return: {np.mean(all_returns):.1%}")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    parser = argparse.ArgumentParser(description="Online Contextual Bandit Stock Trading Model")
    parser.add_argument("tickers", nargs="+", help="Stock tickers to process (e.g., AVAV EH)")
    parser.add_argument("--lookback_days", type=int, default=5,
                        help="Number of days to look back for context (default: 5)")
    args = parser.parse_args()
    
    print(f"Training online contextual bandit for: {', '.join(args.tickers)}")
    print(f"Using lookback period: {args.lookback_days} days")
    
    # Load data
    data = load_data(args.tickers)
    
    # Train initial model
    print("\nTraining initial model...")
    model, scaler = train_model(data, lookback_days=args.lookback_days)
    
    # Setup online learning (creates separate optimizer for post-Lightning updates)
    model.setup_online_learning()
    
    # Online evaluation with learning
    print("\nRunning online evaluation with Thompson sampling updates...")
    buy_events = evaluate_model_online(model, data, scaler, args.tickers)
    
    # Visualize and summarize
    visualize_results(buy_events, data, args.tickers)
    print_summary(buy_events, args.tickers)

if __name__ == "__main__":
    main()