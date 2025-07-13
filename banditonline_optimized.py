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
from collections import defaultdict

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
        all_tickers = []
        
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
                    (data['date'] <= current_row['date'] + pd.Timedelta(days=20))
                ]
                
                if future_data.empty:
                    reward = 0.0
                else:
                    max_return = (future_data['close'].max() - current_row['close']) / current_row['close']
                    reward = max_return
                
                all_sequences.append(sequence)
                all_rewards.append(reward)
                all_tickers.append(ticker)
        
        # Convert to arrays
        self.sequences = np.array(all_sequences)  # Shape: (N, lookback_days+1, num_features)
        self.tickers = all_tickers
        
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
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # For training: only return features and rewards (vectorized)
        return self.features[idx], self.rewards[idx]
    
    def get_ticker(self, idx):
        # Separate method to get ticker when needed (online learning)
        return self.tickers[idx]

class ContextualBandit(pl.LightningModule):
    def __init__(self, input_dim=15, lookback_days=5, learning_rate=0.001, tickers=None):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for Lightning
        
        self.learning_rate = learning_rate
        self.lookback_days = lookback_days
        self.tickers = tickers or []
        
        # Main shared architecture for training (fast vectorized)
        self.shared_gru = nn.GRU(input_dim, 32, batch_first=True)
        self.shared_output_layers = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # [mean, log_var] for Thompson sampling
        )
        
        # Stock-specific layers for online learning only (initialized later)
        self.stock_specific_layers = nn.ModuleDict()
        self.online_optimizers = {}
        self.stock_performance = defaultdict(list)  # Track per-stock performance
        self.online_mode = False  # Flag to switch between training and online modes
    
    def setup_stock_specific_layers(self):
        """Initialize stock-specific layers after training is complete"""
        for ticker in self.tickers:
            # Copy shared architecture for each stock
            self.stock_specific_layers[ticker] = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2)  # [mean, log_var] for Thompson sampling
            )
            # Initialize with shared weights
            with torch.no_grad():
                for stock_param, shared_param in zip(
                    self.stock_specific_layers[ticker].parameters(),
                    self.shared_output_layers.parameters()
                ):
                    stock_param.copy_(shared_param)
        
        self.online_mode = True
        print(f"Initialized stock-specific layers for {len(self.tickers)} stocks")
    
    def forward(self, x, ticker=None):
        # Shared GRU processing (always used)
        gru_out, hidden = self.shared_gru(x)
        last_output = gru_out[:, -1, :]  # Use last output from sequence
        
        # Choose output layers based on mode
        if self.online_mode and ticker and ticker in self.stock_specific_layers:
            # Online learning: use stock-specific layers
            output = self.stock_specific_layers[ticker](last_output)
        else:
            # Training or fallback: use shared layers (vectorized)
            output = self.shared_output_layers(last_output)
        
        return output[:, 0], output[:, 1]  # mean, log_var
    
    def predict_action(self, context, ticker=None, threshold=0.01):
        """Predict buy/hold using Thompson sampling with stock-specific parameters"""
        self.eval()
        with torch.no_grad():
            if len(context.shape) == 2:  # Single sequence: (seq_len, features)
                context = context.unsqueeze(0)  # Add batch dim: (1, seq_len, features)
            
            mean, log_var = self.forward(context, ticker)
            std = torch.exp(0.5 * log_var)
            
            # Sample from posterior (Thompson sampling)
            sampled_reward = mean + std * torch.randn_like(std)
            
            # Buy if predicted reward > threshold (default 1%)
            action = (sampled_reward > threshold).float()
            
        return action.squeeze(), mean.squeeze(), std.squeeze()
    
    def setup_online_learning(self):
        """Initialize optimizers for online learning after Lightning training is complete"""
        # First setup stock-specific layers
        self.setup_stock_specific_layers()
        
        # Create separate optimizers for each stock's specific layers
        for ticker in self.tickers:
            if ticker in self.stock_specific_layers:
                params = list(self.stock_specific_layers[ticker].parameters())
                self.online_optimizers[ticker] = torch.optim.Adam(params, lr=self.learning_rate*10)
    
    def online_update(self, context, action, reward, ticker=None):
        """Update model based on realized reward from Thompson sampling decision"""
        if not self.online_optimizers:
            self.setup_online_learning()
        
        # Only update if we have stock-specific layers and took the buy action
        if action > 0.5 and ticker and ticker in self.online_optimizers:
            self.train()
            
            optimizer = self.online_optimizers[ticker]
            optimizer.zero_grad()
            
            if len(context.shape) == 2:  # Single sequence
                context = context.unsqueeze(0)  # Add batch dimension
            
            mean, log_var = self.forward(context, ticker)
            
            # Negative log-likelihood loss for the realized reward
            var = torch.exp(log_var)
            loss = 0.5 * (log_var + (reward - mean) ** 2 / var).mean()
            
            loss.backward()
            optimizer.step()
            
            # Track stock-specific performance
            self.stock_performance[ticker].append({
                'predicted': mean.item(),
                'actual': reward.item(),
                'loss': loss.item()
            })
            
            return loss.item()
        
        return 0.0
    
    def get_stock_performance_summary(self):
        """Get performance summary for each stock"""
        summary = {}
        for ticker, performance in self.stock_performance.items():
            if performance:
                predictions = [p['predicted'] for p in performance]
                actuals = [p['actual'] for p in performance]
                losses = [p['loss'] for p in performance]
                
                summary[ticker] = {
                    'num_updates': len(performance),
                    'avg_predicted': np.mean(predictions),
                    'avg_actual': np.mean(actuals),
                    'avg_loss': np.mean(losses),
                    'prediction_accuracy': np.corrcoef(predictions, actuals)[0,1] if len(predictions) > 1 else 0
                }
        return summary
    
    def training_step(self, batch, batch_idx):
        """Fast vectorized training using shared layers only"""
        contexts, rewards = batch
        mean, log_var = self.forward(contexts)  # Vectorized forward pass
        
        # Negative log-likelihood loss (vectorized)
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + (rewards - mean) ** 2 / var).mean()
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Fast vectorized validation using shared layers only"""
        contexts, rewards = batch
        mean, log_var = self.forward(contexts)  # Vectorized forward pass
        
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

def train_model(data, tickers, lookback_days=5):
    """Train the contextual bandit with fast vectorized training"""
    # Split chronologically
    split_idx = int(0.8 * len(data))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Create datasets (no ticker info during training for speed)
    train_dataset = StockDataset(train_data, lookback_days=lookback_days)
    val_dataset = StockDataset(val_data, scaler=train_dataset.scaler, lookback_days=lookback_days)
    
    # Create data loaders (standard fast loaders)
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
                             lookback_days=lookback_days,
                             tickers=tickers)
    trainer = pl.Trainer(max_epochs=50, enable_progress_bar=True, accelerator="cpu")
    trainer.fit(model, train_loader, val_loader)
    
    return model, train_dataset.scaler

def evaluate_model_online(model, data, scaler, tickers, threshold=0.01):
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
            
        # Get ticker for this sample
        ticker = dataset.get_ticker(idx)
        
        row = data_sorted.iloc[idx + model.lookback_days]  # Adjust for sequence indexing
        current_date = row['date']
        
        # Process any pending updates where we now have realized rewards (30+ days later)
        updates_to_process = []
        remaining_updates = []
        
        for pending in pending_updates:
            pending_date = pending['date']
            days_elapsed = (current_date - pending_date).days
            
            if days_elapsed >= 20:  # Reward is now realized
                # Calculate actual reward for this past decision
                pending_ticker = pending['ticker']
                entry_price = pending['entry_price']
                
                # Find the best price achieved in the 30 days after the buy decision
                future_data = data_sorted[
                    (data_sorted['ticker'] == pending_ticker) &
                    (data_sorted['date'] > pending_date) &
                    (data_sorted['date'] <= pending_date + pd.Timedelta(days=20))
                ]
                
                if not future_data.empty:
                    max_price = future_data['close'].max()
                    realized_reward = (max_price - entry_price) / entry_price
                else:
                    realized_reward = 0.0
                
                pending['realized_reward'] = realized_reward
                updates_to_process.append(pending)
                
                # Update the corresponding buy event with the realized reward
                event_id = f"{pending_ticker}_{pending_date}"
                if event_id in buy_event_ids:
                    ticker_events = buy_events[pending_ticker]
                    event_idx = buy_event_ids[event_id]
                    if event_idx < len(ticker_events):
                        ticker_events[event_idx]['realized_reward'] = realized_reward
            else:
                remaining_updates.append(pending)
        
        # Update model parameters with realized rewards (stock-specific)
        for update in updates_to_process:
            loss = model.online_update(
                update['context'], 
                update['action'], 
                torch.tensor(update['realized_reward']),
                ticker=update['ticker']
            )
            if loss > 0:
                online_losses.append(loss)
        
        # Keep only pending updates that haven't matured yet
        pending_updates = remaining_updates
        
        # Make Thompson sampling decision for current timestep (stock-specific)
        action, pred_mean, pred_std = model.predict_action(context, ticker=ticker, threshold=threshold)
        
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
    
    # Print stock-specific performance
    performance_summary = model.get_stock_performance_summary()
    print("\n=== STOCK-SPECIFIC LEARNING PERFORMANCE ===")
    for ticker, perf in performance_summary.items():
        print(f"{ticker}: {perf['num_updates']} updates, "
              f"corr={perf['prediction_accuracy']:.3f}, "
              f"avg_loss={perf['avg_loss']:.4f}")
    
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
                plt.colorbar(scatter, ax=axes[i], label='Realized 30-day Return')
            
            # Plot events without realized rewards (pending) in gray
            if unrealized_events:
                dates = [e['date'] for e in unrealized_events]
                prices = [e['price'] for e in unrealized_events]
                
                axes[i].scatter(dates, prices, c='gray', 
                               s=100, alpha=0.5, edgecolors='black', marker='x',
                               label='Pending (No realized return yet)')
        
        axes[i].set_title(f'{ticker} - Optimized: Fast Training + Stock-Specific Online Learning')
        axes[i].set_ylabel('Price ($)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('plots/online_buy_decisions_optimized.png', dpi=300)
    plt.show()

def print_summary(buy_events, tickers):
    """Print performance summary"""
    print("\n=== OPTIMIZED CONTEXTUAL BANDIT PERFORMANCE ===")
    
    for ticker in tickers:
        events = buy_events[ticker]
        
        if events:
            returns = [e['actual_return'] for e in events]
            realized_events = [e for e in events if e['realized_reward'] is not None]
            realized_returns = [e['realized_reward'] for e in realized_events] if realized_events else []
            
            avg_return = np.mean(returns)
            avg_realized = np.mean(realized_returns) if realized_returns else 0
            positive_pct = (np.array(returns) > 0).mean() * 100
            total_buys = len(events)
            realized_buys = len(realized_events)
        else:
            avg_return = avg_realized = positive_pct = total_buys = realized_buys = 0
        
        print(f"{ticker:>6}: {total_buys:>3} buys ({realized_buys} realized), "
              f"{avg_return:>6.1%} avg return, "
              f"{avg_realized:>6.1%} avg realized, "
              f"{positive_pct:>5.1f}% positive")
    
    # Overall stats
    all_returns = [e['actual_return'] for events in buy_events.values() for e in events]
    all_realized = [e['realized_reward'] for events in buy_events.values() for e in events if e['realized_reward'] is not None]
    
    if all_returns:
        total_return = sum(all_returns)
        total_realized = sum(all_realized) if all_realized else 0
        print(f"\nTotal portfolio return: {total_return:.1%}")
        print(f"Total realized return: {total_realized:.1%}")
        print(f"Average trade return: {np.mean(all_returns):.1%}")
        if all_realized:
            print(f"Average realized return: {np.mean(all_realized):.1%}")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    parser = argparse.ArgumentParser(description="Optimized: Fast Training + Stock-Specific Online Learning")
    parser.add_argument("tickers", nargs="+", help="Stock tickers to process (e.g., AVAV EH)")
    parser.add_argument("--lookback_days", type=int, default=5,
                        help="Number of days to look back for context (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Buy threshold for expected return (default: 0.01 = 1%)")
    args = parser.parse_args()
    
    print(f"Training optimized contextual bandit for: {', '.join(args.tickers)}")
    print(f"Strategy: Fast vectorized training + Stock-specific online learning")
    print(f"Using lookback period: {args.lookback_days} days")
    print(f"Buy threshold: {args.threshold:.1%}")
    
    # Load data
    data = load_data(args.tickers)
    
    # Train initial model (FAST - vectorized training)
    print("\nTraining initial model with fast vectorized approach...")
    model, scaler = train_model(data, args.tickers, lookback_days=args.lookback_days)
    
    # Setup online learning (creates stock-specific layers and optimizers)
    print("\nSetting up stock-specific online learning...")
    model.setup_online_learning()
    
    # Online evaluation with learning
    print("\nRunning online evaluation with stock-specific Thompson sampling updates...")
    buy_events = evaluate_model_online(model, data, scaler, args.tickers, threshold=args.threshold)
    
    # Visualize and summarize
    visualize_results(buy_events, data, args.tickers)
    print_summary(buy_events, args.tickers)

if __name__ == "__main__":
    main()