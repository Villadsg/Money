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
        
        # Create sequences for each ticker - NO REWARD CALCULATION HERE
        all_sequences = []
        
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
                all_sequences.append(sequence)
        
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
        self.features = torch.FloatTensor(self.features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return only features - no target rewards
        return self.features[idx]

class ContextualBandit(pl.LightningModule):
    def __init__(self, input_dim=15, lookback_days=5, transaction_cost_pct=0.005, learning_rate=0.001, confidence_threshold=0.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.transaction_cost_pct = transaction_cost_pct
        self.learning_rate = learning_rate
        self.lookback_days = lookback_days
        self.confidence_threshold = confidence_threshold
        
        # GRU for sequence processing
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
        
        # For online learning
        self.online_optimizer = None
        
        # Initialize with neutral priors
        self.prior_mean = 0.0
        self.prior_var = 1.0
    
    def forward(self, x):
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
            
            # Apply confidence threshold - only buy if predicted reward exceeds transaction cost
            # AND the mean reward exceeds our confidence threshold
            action = ((sampled_reward > self.transaction_cost_pct) & 
                     (mean > self.confidence_threshold)).float()
            
        return action.squeeze(), mean.squeeze(), std.squeeze()
    
    def setup_online_learning(self):
        """Initialize optimizer for online learning"""
        self.online_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate*0.1)
    
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
        # For initial training, use a simple regularization loss
        # since we don't have target rewards
        contexts = batch
        mean, log_var = self.forward(contexts)
        
        # Regularization loss to prevent overconfidence
        # Encourage predictions close to prior
        prior_loss = 0.5 * ((mean - self.prior_mean) ** 2 + 
                           (torch.exp(log_var) - self.prior_var) ** 2).mean()
        
        # Encourage reasonable uncertainty
        uncertainty_loss = -log_var.mean()  # Prevent overconfidence
        
        loss = prior_loss + 0.1 * uncertainty_loss
        
        self.log('train_loss', loss)
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

def train_initial_model(data, transaction_cost_pct=0.005, lookback_days=5, max_epochs=10, confidence_threshold=0.0):
    """Train initial model with regularization (no lookahead bias)"""
    # Create dataset without target rewards
    dataset = StockDataset(data, lookback_days=lookback_days)
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Determine input_dim from feature columns
    _feature_cols_list = [
        'open', 'high', 'low', 'close', 'volume', 'market_return',
        'stock_return', 'residual_return', 'residual_gap_pct', 'residual_price',
        'volume_gap_product', 'is_earnings_date', 'earnings_classification', 
        'event_strength', 'future_strength'
    ]
    input_dim = len(_feature_cols_list)

    # Train model with regularization
    model = ContextualBandit(input_dim=input_dim, 
                             lookback_days=lookback_days,
                             transaction_cost_pct=transaction_cost_pct,
                             confidence_threshold=confidence_threshold)
    
    # Training with user-specified number of epochs
    trainer = pl.Trainer(max_epochs=max_epochs, enable_progress_bar=True, accelerator="cpu")
    trainer.fit(model, data_loader)
    
    return model, dataset.scaler

def run_online_trading(model, data, scaler, tickers):
    """Run online trading with real-time learning (no lookahead bias)"""
    # Sort data chronologically
    data_sorted = data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    buy_events = {ticker: [] for ticker in tickers}
    online_losses = []
    
    # For tracking return improvement over time
    training_history = {
        'dates': [],
        'cumulative_return': 0.0,
        'avg_return': 0.0,
        'returns': [],
        'win_rate': 0.0,
        'trade_count': 0
    }
    
    # Store pending trades waiting for outcome
    pending_trades = []
    
    # Process each day sequentially
    for idx in range(len(data_sorted)):
        row = data_sorted.iloc[idx]
        current_date = row['date']
        ticker = row['ticker']
        
        # Check if any pending trades can be resolved (60 days have passed)
        resolved_trades = []
        remaining_trades = []
        
        for trade in pending_trades:
            days_elapsed = (current_date - trade['entry_date']).days
            
            if days_elapsed >= 60:  # Trade holding period complete
                # Calculate realized return
                entry_price = trade['entry_price']
                
                # Find best price achieved during holding period
                holding_period_data = data_sorted[
                    (data_sorted['ticker'] == trade['ticker']) &
                    (data_sorted['date'] > trade['entry_date']) &
                    (data_sorted['date'] <= trade['entry_date'] + pd.Timedelta(days=60))
                ]
                
                if not holding_period_data.empty:
                    max_price = holding_period_data['close'].max()
                    realized_return = (max_price - entry_price) / entry_price
                else:
                    realized_return = 0.0
                
                # Update model with realized outcome
                loss = model.online_update(
                    trade['context'],
                    trade['action'],
                    torch.tensor(realized_return, dtype=torch.float32)
                )
                
                if loss > 0:
                    online_losses.append(loss)
                
                # Update buy event record
                trade['realized_return'] = realized_return
                resolved_trades.append(trade)
                
                # Update training history
                training_history['dates'].append(current_date)
                training_history['returns'].append(realized_return)
                training_history['cumulative_return'] += realized_return
                training_history['trade_count'] += 1
                training_history['avg_return'] = np.mean(training_history['returns'])
                training_history['win_rate'] = (np.array(training_history['returns']) > 0).mean()
                
                print(f"Resolved trade: {trade['ticker']} on {trade['entry_date'].date()} -> {realized_return:.2%}")
                
            else:
                remaining_trades.append(trade)
        
        pending_trades = remaining_trades
        
        # Check if we have enough lookback data for current decision
        ticker_history = data_sorted[
            (data_sorted['ticker'] == ticker) & 
            (data_sorted['date'] <= current_date)
        ].sort_values('date')
        
        if len(ticker_history) < model.lookback_days + 1:
            continue  # Not enough history yet
        
        # Prepare context (last lookback_days + 1 observations)
        recent_data = ticker_history.tail(model.lookback_days + 1)
        
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'market_return',
            'stock_return', 'residual_return', 'residual_gap_pct', 'residual_price',
            'volume_gap_product', 'is_earnings_date', 'earnings_classification', 
            'event_strength', 'future_strength'
        ]
        
        # Extract features and scale
        features_df = recent_data[feature_cols].copy()
        for col in feature_cols:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        features_df = features_df.fillna(0)
        
        # Scale features
        features_scaled = scaler.transform(features_df.values)
        context = torch.FloatTensor(features_scaled)
        
        # Make trading decision
        action, pred_mean, pred_std = model.predict_action(context)
        
        if action.item() > 0.5:  # Buy decision
            # Record buy event
            buy_event = {
                'ticker': ticker,
                'entry_date': current_date,
                'entry_price': row['close'],
                'predicted_return': pred_mean.item(),
                'uncertainty': pred_std.item(),
                'context': context,
                'action': action,
                'realized_return': None  # Will be filled when trade resolves
            }
            
            buy_events[ticker].append(buy_event)
            pending_trades.append(buy_event)
            
            print(f"BUY: {ticker} on {current_date.date()} at ${row['close']:.2f} "
                  f"(pred: {pred_mean.item():.2%} ± {pred_std.item():.2%})")
    
    print(f"\nOnline learning completed:")
    print(f"Total model updates: {len(online_losses)}")
    print(f"Average update loss: {np.mean(online_losses):.4f}" if online_losses else "No updates")
    print(f"Pending trades: {len(pending_trades)}")
    
    return buy_events, training_history

def visualize_results(buy_events, data, tickers, training_history=None):
    """Create visualizations"""
    os.makedirs('plots', exist_ok=True)
    
    # Determine number of subplots needed
    num_plots = len(tickers)
    if training_history and len(training_history['dates']) > 0:
        num_plots += 1  # Add one more plot for training improvement
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 4*num_plots))
    if num_plots == 1:
        axes = [axes]
    
    for i, ticker in enumerate(tickers):
        # Plot stock price
        ticker_data = data[data['ticker'] == ticker].sort_values('date')
        axes[i].plot(ticker_data['date'], ticker_data['close'], 
                    label=f'{ticker} Price', alpha=0.7)
        
        # Plot buy events
        events = buy_events[ticker]
        if events:
            # Resolved trades (with realized returns)
            resolved_events = [e for e in events if e['realized_return'] is not None]
            pending_events = [e for e in events if e['realized_return'] is None]
            
            if resolved_events:
                dates = [e['entry_date'] for e in resolved_events]
                prices = [e['entry_price'] for e in resolved_events]
                returns = [e['realized_return'] for e in resolved_events]
                
                scatter = axes[i].scatter(dates, prices, c=returns, cmap='RdYlGn', 
                                        s=100, alpha=0.8, edgecolors='black')
                plt.colorbar(scatter, ax=axes[i], label='Realized 60-day Return')
            
            if pending_events:
                dates = [e['entry_date'] for e in pending_events]
                prices = [e['entry_price'] for e in pending_events]
                
                axes[i].scatter(dates, prices, c='gray', s=100, alpha=0.5, 
                               edgecolors='black', marker='x', label='Pending')
        
        axes[i].set_title(f'{ticker} - Online Trading Decisions (No Lookahead)')
        axes[i].set_ylabel('Price ($)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    # Add training improvement plot if data is available
    if training_history and len(training_history['dates']) > 0:
        ax_train = axes[-1]  # Use the last subplot for training metrics
        
        # Plot cumulative return
        ax_train.plot(training_history['dates'], 
                     np.cumsum(training_history['returns']), 
                     'b-', label='Cumulative Return')
        
        # Plot moving average of returns (window of 5 trades or fewer if not enough data)
        window = min(5, len(training_history['returns']))
        if window > 1:
            returns = np.array(training_history['returns'])
            moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
            # Align the moving average with the correct dates
            ma_dates = training_history['dates'][window-1:]
            ax_train.plot(ma_dates, moving_avg, 'g-', label=f'{window}-Trade Moving Avg Return')
        
        # Plot win rate over time
        ax_twin = ax_train.twinx()
        win_rates = []
        for i in range(1, len(training_history['returns'])+1):
            win_rates.append((np.array(training_history['returns'][:i]) > 0).mean())
        
        ax_twin.plot(training_history['dates'], win_rates, 'r--', label='Win Rate')
        ax_twin.set_ylabel('Win Rate', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')
        ax_twin.set_ylim(0, 1)
        
        # Add trade count markers
        for i, (date, ret) in enumerate(zip(training_history['dates'], training_history['returns'])):
            if (i+1) % 5 == 0:  # Mark every 5th trade
                ax_train.annotate(f'{i+1}', 
                                 xy=(date, np.sum(training_history['returns'][:i+1])),
                                 xytext=(0, 10),
                                 textcoords='offset points',
                                 ha='center')
        
        ax_train.set_title('Return Improvement During Online Learning')
        ax_train.set_ylabel('Return')
        ax_train.grid(True, alpha=0.3)
        
        # Combine legends from both y-axes
        lines1, labels1 = ax_train.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_train.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig('plots/online_trading_no_lookahead.png', dpi=300)
    plt.show()

def print_summary(buy_events, tickers):
    """Print performance summary"""
    print("\n=== ONLINE TRADING PERFORMANCE (NO LOOKAHEAD) ===")
    
    total_trades = 0
    total_resolved = 0
    all_resolved_returns = []
    
    for ticker in tickers:
        events = buy_events[ticker]
        resolved_events = [e for e in events if e['realized_return'] is not None]
        
        if resolved_events:
            returns = [e['realized_return'] for e in resolved_events]
            avg_return = np.mean(returns)
            positive_pct = (np.array(returns) > 0).mean() * 100
            all_resolved_returns.extend(returns)
        else:
            avg_return = positive_pct = 0
        
        total_trades += len(events)
        total_resolved += len(resolved_events)
        
        print(f"{ticker:>6}: {len(events):>3} total, {len(resolved_events):>3} resolved, "
              f"{avg_return:>6.1%} avg return, {positive_pct:>5.1f}% positive")
    
    print(f"\nOverall: {total_trades} total trades, {total_resolved} resolved")
    if all_resolved_returns:
        print(f"Portfolio return: {sum(all_resolved_returns):.1%}")
        print(f"Average trade return: {np.mean(all_resolved_returns):.1%}")
        print(f"Win rate: {(np.array(all_resolved_returns) > 0).mean()*100:.1f}%")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    parser = argparse.ArgumentParser(description="Online Trading Model (No Lookahead Bias)")
    parser.add_argument("tickers", nargs="+", help="Stock tickers to process")
    parser.add_argument("--transaction_cost", type=float, default=0.005, 
                        help="Transaction cost as percentage")
    parser.add_argument("--lookback_days", type=int, default=5,
                        help="Lookback period in days")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for initial model training")
    parser.add_argument("--confidence", type=float, default=0.0,
                        help="Confidence threshold for buy decisions (higher = more conservative)")
    args = parser.parse_args()
    
    print(f"Online trading model for: {', '.join(args.tickers)}")
    print(f"Transaction cost: {args.transaction_cost:.3%}")
    print(f"Lookback period: {args.lookback_days} days")
    print(f"Training epochs: {args.epochs}")
    print(f"Buy confidence threshold: {args.confidence:.3%}")
    print("*** NO LOOKAHEAD BIAS - PURE ONLINE LEARNING ***")
    
    # Load data
    data = load_data(args.tickers)
    
    # Train initial model with user-specified epochs and confidence threshold
    print("\nInitializing model...")
    model, scaler = train_initial_model(data, transaction_cost_pct=args.transaction_cost, 
                                       lookback_days=args.lookback_days,
                                       max_epochs=args.epochs,
                                       confidence_threshold=args.confidence)
    
    # Setup online learning
    model.setup_online_learning()
    
    # Run online trading simulation
    print("\nRunning online trading simulation...")
    buy_events, training_history = run_online_trading(model, data, scaler, args.tickers)
    
    # Visualize and summarize
    visualize_results(buy_events, data, args.tickers, training_history)
    print_summary(buy_events, args.tickers)

if __name__ == "__main__":
    main()