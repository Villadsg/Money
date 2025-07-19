
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
    def __init__(self, data, scaler=None, lookback_days=0, reward_days=30):
        self.lookback_days = lookback_days
        self.reward_days = reward_days
        
        # Select features
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'market_return',
            'stock_return', 'residual_return', 'residual_gap_pct', 'residual_price',
            'volume_gap_product', 'is_earnings_date', 'earnings_classification', 
            'event_strength', 'future_strength'
        ]
        
        # If lookback_days is 0, use the original approach (no sequences)
        if lookback_days == 0:
            features_df = data[feature_cols].copy()
            for col in feature_cols:
                # Attempt to convert to numeric; strings that can't be converted become NaN
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            
            # Now, fill all NaN values (original NaNs or those from coerced strings) with 0
            self.features = features_df.fillna(0).values
            
            # Scale features
            if scaler is None:
                self.scaler = StandardScaler()
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
            
            # Calculate combined forward returns (best + worst)
            self.rewards = self._calculate_rewards(data, self.reward_days)
            
            self.features = torch.FloatTensor(self.features)
            self.rewards = torch.FloatTensor(self.rewards)
        else:
            # Create sequences for each ticker, similar to banditonline.py
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
                    # Get lookback_days + current day features
                    sequence = features_df.iloc[i-lookback_days:i+1].values  # Shape: (lookback_days+1, num_features)
                    
                    # Calculate reward for this decision point
                    current_row = ticker_data.iloc[i]
                    future_data = data[
                        (data['ticker'] == ticker) &
                        (data['date'] > current_row['date']) &
                        (data['date'] <= current_row['date'] + pd.Timedelta(days=self.reward_days))
                    ]
                    
                    if future_data.empty:
                        reward = 0.0
                    else:
                        # Calculate best possible return (max price)
                        max_return = (future_data['close'].max() - current_row['close']) / current_row['close']
                        # Calculate worst possible return (min price)
                        min_return = (future_data['close'].min() - current_row['close']) / current_row['close']
                        
                        # Combined reward: best return + worst return
                        reward = max_return + min_return
                    
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
    
    def _calculate_rewards(self, data, reward_days=30):
        rewards = []
        for idx, row in data.iterrows():
            # Find both max and min returns in next reward_days days for same ticker
            future = data[
                (data['ticker'] == row['ticker']) &
                (data['date'] > row['date']) &
                (data['date'] <= row['date'] + pd.Timedelta(days=reward_days))
            ]
            
            if future.empty:
                rewards.append(0.0)
            else:
                # Calculate best possible return (max price)
                max_return = (future['close'].max() - row['close']) / row['close']
                # Calculate worst possible return (min price)
                min_return = (future['close'].min() - row['close']) / row['close']
                
                # Combined reward: best return + worst return
                # If worst return is positive, reward is boosted
                # If worst return is negative, reward is reduced
                combined_reward = max_return + min_return
                rewards.append(combined_reward)
        
        return rewards
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.rewards[idx]

class ContextualBandit(pl.LightningModule):
    def __init__(self, input_dim=15, transaction_cost_pct=0.005, lookback_days=0, uncertainty_penalty=1.0): # Added uncertainty_penalty parameter
        super().__init__()
        self.transaction_cost_pct = transaction_cost_pct  # Store the cost
        self.lookback_days = lookback_days  # Store lookback days
        self.uncertainty_penalty = uncertainty_penalty  # Store uncertainty penalty
        
        if lookback_days > 0:
            # Use GRU for sequence processing when lookback_days > 0
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
        else:
            # Simple 3-layer network for non-sequence data
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2)  # [mean, log_var] for Thompson sampling
            )
    
    def forward(self, x):
        if self.lookback_days > 0:
            # Process sequence data with GRU
            gru_out, hidden = self.gru(x)
            last_output = gru_out[:, -1, :]  # Use last output from sequence
            output = self.output_layers(last_output)
        else:
            # Process non-sequence data with regular network
            output = self.network(x)
        
        return output[:, 0], output[:, 1]  # mean, log_var
    
    def predict_action(self, context):
        """Predict buy/hold using Thompson sampling"""
        self.eval()
        with torch.no_grad():
            if self.lookback_days > 0:
                # Handle sequence data
                if len(context.shape) == 2:  # Single sequence: (seq_len, features)
                    context = context.unsqueeze(0)  # Add batch dim: (1, seq_len, features)
            else:
                # Handle non-sequence data
                if len(context.shape) == 1:
                    context = context.unsqueeze(0)
            
            mean, log_var = self.forward(context)
            std = torch.exp(0.5 * log_var)
            
            # Sample from posterior (Thompson sampling)
            sampled_reward = mean + std * torch.randn_like(std)
            
            # Uncertainty-adjusted threshold: higher uncertainty = higher threshold
            base_threshold = 0 + self.transaction_cost_pct
            uncertainty_adjusted_threshold = base_threshold + (self.uncertainty_penalty * std)
            
            # Buy if sampled reward exceeds uncertainty-adjusted threshold
            action = (sampled_reward > uncertainty_adjusted_threshold).float()
            
        return action.squeeze(), mean.squeeze(), std.squeeze()
    
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
        return torch.optim.Adam(self.parameters(), lr=0.001)

def load_data(tickers):
    """Load and combine stock data from analysis files"""
    all_data = []
    
    for ticker in tickers:
        # Look for analysis files with date pattern
        files = glob.glob(f"data/{ticker}_analysis_*.csv")
        if files:
            latest_file = max(files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            
            # The first column is the ticker, rename it
            df.columns = ['ticker'] + df.columns[1:].tolist()
            
            # Add missing future_strength column (set to 0 for now)
            if 'future_strength' not in df.columns:
                df['future_strength'] = 0.0
            
            all_data.append(df)
            print(f"Loaded {ticker}: {len(df)} rows")
    
    if not all_data:
        raise ValueError("No analysis data files found!")
    
    # Combine and sort
    data = pd.concat(all_data, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    print(f"Total data: {len(data)} rows")
    return data

def train_model(data, transaction_cost_pct=0.005, lookback_days=0, reward_days=30, uncertainty_penalty=1.0, max_epochs=50): # Added max_epochs parameter
    """Train the contextual bandit"""
    # Split chronologically
    split_idx = int(0.8 * len(data))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Create datasets with lookback_days and reward_days
    train_dataset = StockDataset(train_data, lookback_days=lookback_days, reward_days=reward_days)
    val_dataset = StockDataset(val_data, scaler=train_dataset.scaler, lookback_days=lookback_days, reward_days=reward_days)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # Determine input_dim from the dataset
    # Create a temporary dataset instance just to get feature shape, or get from feature_cols if available
    # This assumes StockDataset is defined and accessible
    # A bit of a workaround to get input_dim without fully processing train_dataset yet for it.
    # More robustly, feature_cols could be a global constant or passed around.
    temp_train_df_for_dim = data.iloc[:int(0.8 * len(data))]
    if not temp_train_df_for_dim.empty:
        # Need to define feature_cols here or pass it, or access it from StockDataset if static
        # For now, assuming the feature_cols list used in StockDataset is fixed at 15 features
        # as per original hardcoding. A more dynamic way is preferred if feature_cols can change.
        # Let's get it from a temporary dataset instance, assuming data has the columns.
        _feature_cols_list = [
            'open', 'high', 'low', 'close', 'volume', 'market_return',
            'stock_return', 'residual_return', 'residual_gap_pct', 'residual_price',
            'volume_gap_product', 'is_earnings_date', 'earnings_classification', 
            'event_strength', 'future_strength'
        ]
        # Ensure all feature columns exist in the dataframe before trying to access them
        cols_exist = all(col in temp_train_df_for_dim.columns for col in _feature_cols_list)
        if cols_exist:
            input_dim = len(_feature_cols_list)
        else:
            print("Warning: Not all feature columns found in data for input_dim calculation. Defaulting to 15.")
            input_dim = 15 # Fallback
    else:
        print("Warning: Training data is empty for input_dim calculation. Defaulting to 15.")
        input_dim = 15 # Fallback if data is empty

    # Train model
    model = ContextualBandit(input_dim=input_dim, 
                             transaction_cost_pct=transaction_cost_pct,
                             lookback_days=lookback_days,
                             uncertainty_penalty=uncertainty_penalty) # Pass uncertainty_penalty parameter
    
    # Enable CSV logging to track training progress
    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger("logs", name="bandit_training")
    
    trainer = pl.Trainer(max_epochs=max_epochs, enable_progress_bar=True, accelerator="cpu", logger=logger)
    trainer.fit(model, train_loader, val_loader)
    
    # Plot training curves
    plot_training_curves("logs/bandit_training")
    
    return model, train_dataset.scaler

def plot_training_curves(log_dir):
    """Plot training and validation loss curves"""
    import pandas as pd
    
    # Find the metrics CSV file
    import glob
    csv_files = glob.glob(f"{log_dir}/version_*/metrics.csv")
    if not csv_files:
        print("No training logs found for visualization")
        return
    
    # Load the latest log file
    latest_csv = max(csv_files, key=os.path.getctime)
    df = pd.read_csv(latest_csv)
    
    # Check which columns are available
    has_train_loss = 'train_loss' in df.columns
    has_val_loss = 'val_loss' in df.columns
    
    if not has_val_loss:
        print("No validation loss data found for visualization")
        return
    
    # Filter out rows with missing values
    if has_train_loss:
        train_df = df[df['train_loss'].notna()].copy()
    else:
        train_df = pd.DataFrame()
    
    val_df = df[df['val_loss'].notna()].copy()
    
    if len(val_df) == 0:
        print("No valid validation data found for visualization")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    if has_train_loss and len(train_df) > 0:
        plt.plot(train_df['epoch'], train_df['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(val_df['epoch'], val_df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate (if available)
    plt.subplot(1, 2, 2)
    if 'lr-Adam' in df.columns:
        lr_df = df[df['lr-Adam'].notna()]
        plt.plot(lr_df['epoch'], lr_df['lr-Adam'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
    else:
        # Plot loss difference (overfitting indicator) if training loss is available
        if has_train_loss and len(train_df) > 0:
            min_len = min(len(train_df), len(val_df))
            if min_len > 1:
                loss_diff = val_df['val_loss'].iloc[:min_len].values - train_df['train_loss'].iloc[:min_len].values
                plt.plot(range(min_len), loss_diff, 'm-', linewidth=2)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plt.xlabel('Epoch')
                plt.ylabel('Val Loss - Train Loss')
                plt.title('Overfitting Indicator\n(Higher = More Overfitting)')
        else:
            # Just plot validation loss over time
            plt.plot(val_df['epoch'], val_df['val_loss'], 'r-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('Validation Loss Over Time')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print recommendations
    print(f"\n=== TRAINING ANALYSIS ===")
    if has_train_loss and len(train_df) > 0:
        print(f"Final training loss: {train_df['train_loss'].iloc[-1]:.4f}")
    print(f"Final validation loss: {val_df['val_loss'].iloc[-1]:.4f}")
    
    # Check for overfitting
    if len(val_df) > 10:
        early_val_loss = val_df['val_loss'].iloc[:len(val_df)//3].mean()
        late_val_loss = val_df['val_loss'].iloc[-len(val_df)//3:].mean()
        
        if late_val_loss > early_val_loss * 1.1:
            print("⚠️  OVERFITTING DETECTED: Validation loss is increasing")
            print("   → Recommendation: Reduce epochs or add regularization")
        elif has_train_loss and len(train_df) > 0 and train_df['train_loss'].iloc[-1] > train_df['train_loss'].iloc[-10:].mean() * 0.95:
            print("⚠️  UNDERFITTING: Training loss still decreasing")
            print("   → Recommendation: Increase epochs")
        else:
            print("✅ TRAINING COMPLETE: Model training finished")

def evaluate_model(model, data, scaler, tickers, reward_days=30):
    """Evaluate model and collect buy decisions"""
    dataset = StockDataset(data, scaler=scaler, lookback_days=model.lookback_days, reward_days=reward_days)
    buy_events = {ticker: [] for ticker in tickers}
    
    # When using lookback_days, we need to map dataset indices to original data indices
    if model.lookback_days > 0:
        # Create mapping from dataset indices to original data
        data_indices = []
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            ticker_indices = data[data['ticker'] == ticker].index.tolist()
            # Skip first lookback_days entries for each ticker
            for i in range(model.lookback_days, len(ticker_data)):
                original_idx = ticker_indices[i]
                data_indices.append(original_idx)
    else:
        data_indices = list(range(len(data)))
    
    model.eval()
    for idx, (context, true_reward) in enumerate(dataset):
        if idx >= len(data_indices):
            break
            
        original_idx = data_indices[idx]
        row = data.iloc[original_idx]
        
        action, pred_mean, pred_std = model.predict_action(context)
        
        if action.item() > 0.5:  # Buy decision
            # Check if this is a recent buy (within last reward_days from max date)
            max_date = data['date'].max()
            days_from_end = (max_date - row['date']).days
            is_recent = days_from_end < reward_days
            
            buy_events[row['ticker']].append({
                'date': row['date'],
                'price': row['close'],
                'actual_return': true_reward.item() if not is_recent else None,
                'predicted_return': pred_mean.item(),
                'uncertainty': pred_std.item(),
                'is_recent': is_recent
            })
    
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
            # Separate recent and historical buy events
            historical_events = [e for e in events if not e.get('is_recent', False)]
            recent_events = [e for e in events if e.get('is_recent', False)]
            
            # Plot historical buy events with color mapping
            if historical_events:
                hist_dates = [e['date'] for e in historical_events]
                hist_prices = [e['price'] for e in historical_events]
                hist_returns = [e['actual_return'] for e in historical_events]
                
                scatter = axes[i].scatter(hist_dates, hist_prices, c=hist_returns, 
                                        cmap='RdYlGn', s=100, alpha=0.8, 
                                        edgecolors='black', label='Historical Buys')
                plt.colorbar(scatter, ax=axes[i], label='Combined Return (Best + Worst)')
            
            # Plot recent buy events without color (grey)
            if recent_events:
                recent_dates = [e['date'] for e in recent_events]
                recent_prices = [e['price'] for e in recent_events]
                
                axes[i].scatter(recent_dates, recent_prices, c='grey', 
                              s=100, alpha=0.8, edgecolors='black', 
                              label='Recent Buys (No return data)', marker='^')
        
        axes[i].set_title(f'{ticker} - Buy Decisions')
        axes[i].set_ylabel('Price ($)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def print_summary(buy_events, tickers):
    """Print performance summary"""
    print("\n=== PERFORMANCE SUMMARY ===")
    
    for ticker in tickers:
        events = buy_events[ticker]
        
        if events:
            returns = [e['actual_return'] for e in events if e['actual_return'] is not None]
            if returns:
                avg_return = np.mean(returns)
                positive_pct = (np.array(returns) > 0).mean() * 100
            else:
                avg_return = positive_pct = 0
            total_buys = len(events)
        else:
            avg_return = positive_pct = total_buys = 0
        
        print(f"{ticker:>6}: {total_buys:>3} buys, "
              f"{avg_return:>6.1%} avg return, "
              f"{positive_pct:>5.1f}% positive")
    
    # Overall stats
    all_returns = [e['actual_return'] for events in buy_events.values() for e in events if e['actual_return'] is not None]
    if all_returns:
        total_return = sum(all_returns)
        print(f"\nTotal portfolio return: {total_return:.1%}")
        print(f"Average trade return: {np.mean(all_returns):.1%}")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    parser = argparse.ArgumentParser(description="Contextual Bandit Stock Trading Model")
    parser.add_argument("tickers", nargs="+", help="Stock tickers to process (e.g., AVAV EH)")
    parser.add_argument("--transaction_cost", type=float, default=0, 
                        help="Transaction cost as a percentage (e.g., 0.005 for 0.5%%)")
    parser.add_argument("--lookback_days", type=int, default=0,
                        help="Number of lookback days for sequence modeling (0 for no lookback)")
    parser.add_argument("--reward_days", type=int, default=30,
                        help="Number of days to look ahead for reward calculation (best + worst returns)")
    parser.add_argument("--uncertainty_penalty", type=float, default=1.0,
                        help="Penalty factor for uncertainty (higher = more conservative when uncertain)")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Number of training epochs")
    args = parser.parse_args()
    
    print(f"Training contextual bandit for: {', '.join(args.tickers)}")
    print(f"Using transaction cost: {args.transaction_cost:.3%}")
    print(f"Using lookback days: {args.lookback_days}")
    print(f"Using reward calculation days: {args.reward_days}")
    print(f"Using uncertainty penalty: {args.uncertainty_penalty:.2f}")
    print(f"Using epochs: {args.epochs}")
    
    print(f"Training contextual bandit for: {', '.join(args.tickers)}")
    
    # Load data
    data = load_data(args.tickers)
    
    # Train model
    print("\nTraining model...")
    model, scaler = train_model(data, transaction_cost_pct=args.transaction_cost, 
                               lookback_days=args.lookback_days, reward_days=args.reward_days,
                               uncertainty_penalty=args.uncertainty_penalty, max_epochs=args.epochs)
    
    # Evaluate
    print("\nEvaluating model...")
    buy_events = evaluate_model(model, data, scaler, args.tickers, reward_days=args.reward_days)
    
    # Visualize and summarize
    visualize_results(buy_events, data, args.tickers)
    print_summary(buy_events, args.tickers)

if __name__ == "__main__":
    main()