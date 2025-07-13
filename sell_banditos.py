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
    def __init__(self, data, scaler=None, lookback_days=0, loss_days=30):
        self.lookback_days = lookback_days
        self.loss_days = loss_days
        
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
            
            # Calculate combined loss/gain rewards over loss_days  
            self.rewards = self._calculate_loss_rewards(data)
            
            self.features = torch.FloatTensor(self.features)
            self.rewards = torch.FloatTensor(self.rewards)
        else:
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
                    # Get lookback_days + current day features
                    sequence = features_df.iloc[i-lookback_days:i+1].values  # Shape: (lookback_days+1, num_features)
                    
                    # Calculate reward for this decision point
                    current_row = ticker_data.iloc[i]
                    future_data = data[
                        (data['ticker'] == ticker) &
                        (data['date'] > current_row['date']) &
                        (data['date'] <= current_row['date'] + pd.Timedelta(days=self.loss_days))
                    ]
                    
                    if future_data.empty:
                        reward = 0.0
                    else:
                        # Calculate maximum potential loss (worst case - how much you could lose by holding)
                        min_price = future_data['close'].min()
                        max_loss = (current_row['close'] - min_price) / current_row['close']
                        
                        # Calculate maximum potential gain (best case - opportunity cost of selling)
                        max_price = future_data['close'].max()
                        max_gain = (max_price - current_row['close']) / current_row['close']
                        
                        # Combined reward for selling: max_loss - max_gain
                        reward = max_loss - max_gain
                    
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
    
    def _calculate_loss_rewards(self, data):
        rewards = []
        for idx, row in data.iterrows():
            # Find both maximum loss and maximum gain in next loss_days days for same ticker
            future = data[
                (data['ticker'] == row['ticker']) &
                (data['date'] > row['date']) &
                (data['date'] <= row['date'] + pd.Timedelta(days=self.loss_days))
            ]
            
            if future.empty:
                rewards.append(0.0)
            else:
                # Calculate maximum potential loss (worst case - how much you could lose by holding)
                min_price = future['close'].min()
                max_loss = (row['close'] - min_price) / row['close']
                
                # Calculate maximum potential gain (best case - opportunity cost of selling)
                max_price = future['close'].max()
                max_gain = (max_price - row['close']) / row['close']
                
                # Combined reward for selling: max_loss - max_gain
                # High loss potential + low gain potential = high sell reward
                # Low loss potential + high gain potential = low/negative sell reward (don't sell)
                combined_reward = max_loss - max_gain
                rewards.append(combined_reward)
        
        return rewards
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.rewards[idx]

class ContextualBandit(pl.LightningModule):
    def __init__(self, input_dim=15, transaction_cost_pct=0.005, lookback_days=0):
        super().__init__()
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback_days = lookback_days
        
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
        """Predict sell/hold using Thompson sampling"""
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
            sampled_loss = mean + std * torch.randn_like(std)
            
            # Sell if predicted loss > (threshold + transaction_cost)
            # Higher threshold means more conservative (fewer sells)
            action = (sampled_loss > (0.05 + self.transaction_cost_pct)).float()
            
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

def train_model(data, transaction_cost_pct=0.005, lookback_days=0, loss_days=30):
    """Train the contextual bandit"""
    # Split chronologically
    split_idx = int(0.8 * len(data))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Create datasets with lookback_days and loss_days
    train_dataset = StockDataset(train_data, lookback_days=lookback_days, loss_days=loss_days)
    val_dataset = StockDataset(val_data, scaler=train_dataset.scaler, lookback_days=lookback_days, loss_days=loss_days)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # Determine input_dim from the dataset
    temp_train_df_for_dim = data.iloc[:int(0.8 * len(data))]
    if not temp_train_df_for_dim.empty:
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
            input_dim = 15
    else:
        print("Warning: Training data is empty for input_dim calculation. Defaulting to 15.")
        input_dim = 15

    # Train model
    model = ContextualBandit(input_dim=input_dim, 
                             transaction_cost_pct=transaction_cost_pct,
                             lookback_days=lookback_days)
    trainer = pl.Trainer(max_epochs=50, enable_progress_bar=True, accelerator="cpu")
    trainer.fit(model, train_loader, val_loader)
    
    return model, train_dataset.scaler

def evaluate_model(model, data, scaler, tickers, loss_days=30):
    """Evaluate model and collect sell decisions"""
    dataset = StockDataset(data, scaler=scaler, lookback_days=model.lookback_days, loss_days=loss_days)
    sell_events = {ticker: [] for ticker in tickers}
    
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
    for idx, (context, true_loss) in enumerate(dataset):
        if idx >= len(data_indices):
            break
            
        original_idx = data_indices[idx]
        row = data.iloc[original_idx]
        
        action, pred_mean, pred_std = model.predict_action(context)
        
        if action.item() > 0.5:  # Sell decision
            # Check if this is a recent sell (within last loss_days from max date)
            max_date = data['date'].max()
            days_from_end = (max_date - row['date']).days
            is_recent = days_from_end < loss_days
            
            sell_events[row['ticker']].append({
                'date': row['date'],
                'price': row['close'],
                'actual_loss': true_loss.item() if not is_recent else None,
                'predicted_loss': pred_mean.item(),
                'uncertainty': pred_std.item(),
                'is_recent': is_recent
            })
    
    return sell_events

def visualize_results(sell_events, data, tickers):
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
        
        # Plot sell events
        events = sell_events[ticker]
        if events:
            # Separate recent and historical sell events
            historical_events = [e for e in events if not e.get('is_recent', False)]
            recent_events = [e for e in events if e.get('is_recent', False)]
            
            # Plot historical sell events with color mapping (red=high loss avoided, green=low loss)
            if historical_events:
                hist_dates = [e['date'] for e in historical_events]
                hist_prices = [e['price'] for e in historical_events]
                hist_losses = [e['actual_loss'] for e in historical_events]
                
                scatter = axes[i].scatter(hist_dates, hist_prices, c=hist_losses, 
                                        cmap='RdYlGn_r', s=100, alpha=0.8, 
                                        edgecolors='black', label='Historical Sells', marker='v')
                plt.colorbar(scatter, ax=axes[i], label='Sell Reward (Loss - Gain Potential)')
            
            # Plot recent sell events without color (grey)
            if recent_events:
                recent_dates = [e['date'] for e in recent_events]
                recent_prices = [e['price'] for e in recent_events]
                
                axes[i].scatter(recent_dates, recent_prices, c='grey', 
                              s=100, alpha=0.8, edgecolors='black', 
                              label='Recent Sells (No loss data)', marker='s')
        
        axes[i].set_title(f'{ticker} - Sell Decisions')
        axes[i].set_ylabel('Price ($)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def print_summary(sell_events, tickers):
    """Print performance summary"""
    print("\n=== SELL PERFORMANCE SUMMARY ===")
    
    for ticker in tickers:
        events = sell_events[ticker]
        
        if events:
            rewards = [e['actual_loss'] for e in events if e['actual_loss'] is not None]
            if rewards:
                avg_sell_reward = np.mean(rewards)
                positive_reward_pct = (np.array(rewards) > 0.02).mean() * 100  # >2% positive sell reward
            else:
                avg_sell_reward = positive_reward_pct = 0
            total_sells = len(events)
        else:
            avg_sell_reward = positive_reward_pct = total_sells = 0
        
        print(f"{ticker:>6}: {total_sells:>3} sells, "
              f"{avg_sell_reward:>6.1%} avg sell reward, "
              f"{positive_reward_pct:>5.1f}% positive rewards")
    
    # Overall stats
    all_rewards = [e['actual_loss'] for events in sell_events.values() for e in events if e['actual_loss'] is not None]
    if all_rewards:
        total_sell_reward = sum(all_rewards)
        print(f"\nTotal sell reward: {total_sell_reward:.1%}")
        print(f"Average sell reward per trade: {np.mean(all_rewards):.1%}")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    parser = argparse.ArgumentParser(description="Contextual Bandit Stock Sell Model")
    parser.add_argument("tickers", nargs="+", help="Stock tickers to process (e.g., AVAV EH)")
    parser.add_argument("--transaction_cost", type=float, default=0.005, 
                        help="Transaction cost as a percentage (e.g., 0.005 for 0.5%%)")
    parser.add_argument("--lookback_days", type=int, default=0,
                        help="Number of lookback days for sequence modeling (0 for no lookback)")
    parser.add_argument("--loss_days", type=int, default=30,
                        help="Number of days to look ahead for maximum potential loss calculation")
    args = parser.parse_args()
    
    print(f"Training contextual bandit for SELL decisions: {', '.join(args.tickers)}")
    print(f"Using transaction cost: {args.transaction_cost:.3%}")
    print(f"Using lookback days: {args.lookback_days}")
    print(f"Using loss calculation days: {args.loss_days}")
    
    # Load data
    data = load_data(args.tickers)
    
    # Train model
    print("\nTraining sell model...")
    model, scaler = train_model(data, transaction_cost_pct=args.transaction_cost, 
                               lookback_days=args.lookback_days, loss_days=args.loss_days)
    
    # Evaluate
    print("\nEvaluating sell model...")
    sell_events = evaluate_model(model, data, scaler, args.tickers, loss_days=args.loss_days)
    
    # Visualize and summarize
    visualize_results(sell_events, data, args.tickers)
    print_summary(sell_events, args.tickers)

if __name__ == "__main__":
    main()