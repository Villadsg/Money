#!/usr/bin/env python3
"""
Stock Analyzer Web Application - A Streamlit interface for analyzing stocks
and making buy/sell suggestions based on residual analysis.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stock_analyzer import StockAnalyzer

# Try to import RL components
try:
    import torch
    import numpy as np
    from torch_rl_agent import train_rl_agent, evaluate_rl_agent, DQNAgent
    RL_AVAILABLE = True
    print("PyTorch RL components loaded successfully!")
except ImportError:
    RL_AVAILABLE = False
    print("PyTorch RL components not available. Some features will be disabled.")

# Set page configuration
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .stAlert > div {
        padding: 0.5rem 0.5rem;
        border-radius: 0.25rem;
    }
    .buy-signal {
        color: green;
        font-weight: bold;
    }
    .sell-signal {
        color: red;
        font-weight: bold;
    }
    .hold-signal {
        color: orange;
        font-weight: bold;
    }
    .unknown-signal {
        color: gray;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Analyzer")
st.markdown("""
This tool analyzes a stock against a market benchmark (default: MSCI World ETF) 
and provides buy/sell suggestions based on residual analysis.
""")

# Sidebar for inputs
st.sidebar.header("Analysis Parameters")

# Stock ticker input
ticker_symbol = st.sidebar.text_input("Stock Ticker Symbol", "AAPL").upper()

# Market segment analysis option
use_segment_analysis = st.sidebar.checkbox("Enable Market Segment Analysis", False, 
                                         help="Analyze multiple tickers as a market segment to detect correlation changes")

# Multiple ticker selection for market segment
segment_tickers_input = ""
if use_segment_analysis:
    segment_tickers_input = st.sidebar.text_area(
        "Market Segment Tickers (comma-separated)", 
        "MSFT, GOOGL, AMZN, META",
        help="Enter ticker symbols that constitute a market segment along with the main ticker")
    
    # Parse segment tickers
    segment_tickers = [ticker.strip().upper() for ticker in segment_tickers_input.split(",") if ticker.strip()]
    
    # Remove the main ticker from segment tickers if it's there
    if ticker_symbol in segment_tickers:
        segment_tickers.remove(ticker_symbol)
    
    # Display info about segment analysis
    if segment_tickers:
        st.sidebar.info(f"Analyzing {ticker_symbol} against {len(segment_tickers)} segment tickers")
    else:
        st.sidebar.warning("Please enter at least one segment ticker for segment analysis")
else:
    segment_tickers = []

# Market benchmark selection
market_options = {
    "MSCI World ETF": "URTH",
    "S&P 500": "SPY",
    "NASDAQ": "QQQ",
    "Dow Jones": "DIA",
    "Russell 2000": "IWM"
}
market_selection = st.sidebar.selectbox(
    "Market Benchmark", 
    list(market_options.keys()),
    index=0
)
market_symbol = market_options[market_selection]

# Timeframe selection
timeframe_options = {
    "1 Month": "1m",
    "3 Months": "3m",
    "6 Months": "6m",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "Maximum": "max"
}
timeframe_selection = st.sidebar.selectbox(
    "Analysis Timeframe",
    list(timeframe_options.keys()),
    index=3  # Default to 1 Year
)
timeframe = timeframe_options[timeframe_selection]

# Advanced options
with st.sidebar.expander("Advanced Options"):
    # Custom date range
    use_custom_dates = st.checkbox("Use Custom Date Range", False)
    
    if use_custom_dates:
        end_date = st.date_input(
            "End Date",
            date.today()
        )
        
        # Calculate default start date based on timeframe
        default_start = end_date - timedelta(days=365)
        start_date = st.date_input(
            "Start Date",
            default_start
        )
        
        # Validate date range
        if start_date >= end_date:
            st.error("Start date must be before end date")
    else:
        start_date = None
        end_date = None
    
    # Signal parameters
    residual_threshold = st.slider(
        "Residual Threshold",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Threshold for residual to generate buy/sell signals"
    )
    
    lookback_period = st.slider(
        "Lookback Period (days)",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
        help="Number of days to look back for context"
    )
    
    # Reinforcement Learning options
    if RL_AVAILABLE:
        st.markdown("### Reinforcement Learning")  
        use_rl = st.checkbox("Use Reinforcement Learning", False,
                            help="Use a reinforcement learning agent to generate trading signals")
        
        rl_episodes = st.slider(
            "Training Episodes",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of episodes to train the RL agent"
        ) if use_rl else 50
    else:
        use_rl = False
        rl_episodes = 50
        st.warning("Reinforcement Learning functionality is not available. Install TensorFlow and Keras to enable it.")

# Run analysis button
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# Function to format signal with color
def format_signal(signal):
    if "BUY" in signal:
        return f'<span class="buy-signal">{signal}</span>'
    elif "SELL" in signal:
        return f'<span class="sell-signal">{signal}</span>'
    elif signal == "HOLD":
        return f'<span class="hold-signal">{signal}</span>'
    else:
        return f'<span class="unknown-signal">{signal}</span>'

# Function to perform RL analysis
def run_rl_analysis(stock_data, market_data, episodes=10, batch_size=32):
    """Run reinforcement learning analysis on stock data"""
    if not RL_AVAILABLE:
        return {
            "signal": "HOLD",
            "reason": "Reinforcement learning components not available"
        }
    
    try:
        # Train the RL agent
        with st.spinner("Training RL agent..."):
            agent, history = train_rl_agent(stock_data, market_data, episodes=episodes, batch_size=batch_size)
        
        # Evaluate the agent
        with st.spinner("Evaluating RL agent..."):
            evaluation = evaluate_rl_agent(agent, stock_data, market_data)
        
        # Determine signal based on agent's actions
        actions = evaluation['actions_taken']
        total_actions = sum(actions.values())
        buy_ratio = actions[1] / total_actions if total_actions > 0 else 0
        sell_ratio = actions[2] / total_actions if total_actions > 0 else 0
        
        # Generate signal based on agent's behavior
        if buy_ratio > 0.6:
            signal = "BUY"
            reason = f"RL agent preferred buying ({buy_ratio:.1%} of actions). Final ROI: {evaluation['roi']:.2f}%"
        elif sell_ratio > 0.6:
            signal = "SELL"
            reason = f"RL agent preferred selling ({sell_ratio:.1%} of actions). Final ROI: {evaluation['roi']:.2f}%"
        else:
            signal = "HOLD"
            reason = f"RL agent was neutral (Buy: {buy_ratio:.1%}, Sell: {sell_ratio:.1%}). Final ROI: {evaluation['roi']:.2f}%"
        
        # Create visualization of training progress
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history['episode_rewards'])
        ax[0].set_title('Episode Rewards')
        ax[0].set_xlabel('Episode')
        ax[0].set_ylabel('Total Reward')
        
        ax[1].plot(history['portfolio_values'])
        ax[1].set_title('Portfolio Value')
        ax[1].set_xlabel('Episode')
        ax[1].set_ylabel('Value ($)')
        
        plt.tight_layout()
        
        return {
            "signal": signal,
            "reason": reason,
            "evaluation": evaluation,
            "history": history,
            "visualization": fig
        }
    except Exception as e:
        st.error(f"Error in RL analysis: {str(e)}")
        return {
            "signal": "HOLD",
            "reason": f"Error in RL analysis: {str(e)}"
        }

# Main content
if run_analysis:
    try:
        # Show loading spinner
        with st.spinner(f"Analyzing {ticker_symbol} against {market_selection}..."):
            # Create analyzer
            analyzer = StockAnalyzer(
                ticker_symbol=ticker_symbol,
                market_symbol=market_symbol,
                start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
                end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
                timeframe=timeframe,
                segment_tickers=segment_tickers if use_segment_analysis else None
            )
            
            # Run analysis
            analyzer.prepare_data()
            if analyzer.run_regression():
                # Get buy/sell signal
                signal_info = analyzer.get_buy_sell_signal(
                    residual_threshold=residual_threshold,
                    lookback_period=lookback_period
                )
                
                # Train RL agent if requested
                rl_signal = None
                rl_visualization = None
                if use_rl and RL_AVAILABLE:
                    with st.spinner(f"Training reinforcement learning agent for {ticker_symbol}..."):
                        # Get stock and market data from the analyzer
                        stock_data = analyzer.stock_data['Close']
                        market_data = analyzer.market_data['Close']
                        
                        # Make sure the data series have names
                        stock_data.name = ticker_symbol
                        market_data.name = market_symbol
                        
                        # Run RL analysis
                        rl_results = run_rl_analysis(stock_data, market_data, episodes=rl_episodes, batch_size=32)
                        rl_signal = {
                            "signal": rl_results["signal"],
                            "reason": rl_results["reason"]
                        }
                        
                        # Store visualization if available
                        if "visualization" in rl_results:
                            rl_visualization = rl_results["visualization"]
                
                # Get correlation signals if segment analysis is enabled
                correlation_signals = None
                divergence_signals = None
                if use_segment_analysis and segment_tickers:
                    correlation_signals = analyzer.get_segment_correlation_signals(
                        lookback_period=lookback_period,
                        correlation_threshold=0.3  # Default threshold
                    )
                    
                    # Also get the residual divergence analysis
                    divergence_signals = analyzer.analyze_residual_divergence(
                        lookback_period=lookback_period,
                        divergence_threshold=1.5  # Default threshold
                    )
                
                # Display results in columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Analysis Results")
                    
                    # Create tabs for different analysis methods
                    if rl_signal and RL_AVAILABLE and use_rl:
                        analysis_tabs = st.tabs(["Residual Analysis", "Reinforcement Learning"])
                    else:
                        analysis_tabs = st.tabs(["Residual Analysis"])
                    
                    # Residual Analysis Tab
                    with analysis_tabs[0]:
                        # Display signal
                        st.markdown(f"### Signal: {format_signal(signal_info['signal'])}", unsafe_allow_html=True)
                        st.markdown(f"**Reason:** {signal_info['reason']}")
                        
                        # Display latest data
                        context = signal_info['context']
                        st.markdown(f"**Latest Data (as of {context['latest_date']}):**")
                        
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Price", f"${context['latest_price']:.2f}")
                        with metrics_col2:
                            st.metric("Residual", f"{context['latest_residual']:.4f}")
                        with metrics_col3:
                            st.metric("Z-Score", f"{context['z_score']:.2f}")
                        
                        # Interpretation
                        st.markdown("### Interpretation")
                        st.markdown("""
                        - **Negative residuals** suggest the stock is **undervalued** relative to the market
                        - **Positive residuals** suggest the stock is **overvalued** relative to the market
                        - The **strength** of the signal depends on how far the residual is from its recent average
                        """)
                    
                    # RL Analysis Tab
                    if rl_signal and RL_AVAILABLE and use_rl and len(analysis_tabs) > 1:
                        with analysis_tabs[1]:
                            # Display RL signal
                            st.markdown(f"### RL Signal: {format_signal(rl_signal['signal'])}", unsafe_allow_html=True)
                            st.markdown(f"**Reason:** {rl_signal['reason']}")
                            
                            # Display RL visualization if available
                            if rl_visualization:
                                st.pyplot(rl_visualization)
                                
                            # Add explanation of PyTorch-based RL approach
                            st.markdown("### About the Reinforcement Learning Analysis")
                            st.markdown("""
                            This analysis uses a **Deep Q-Network (DQN)** implemented with PyTorch to learn optimal trading strategies:
                            
                            - The agent observes market conditions, stock prices, and portfolio state
                            - It learns to take actions (buy, sell, hold) to maximize returns
                            - Training occurs over multiple episodes to discover patterns
                            - The final signal is based on the agent's preferred actions during evaluation
                            
                            *Note: RL analysis is experimental and should be used alongside traditional analysis methods.*
                            """)
                            st.markdown(f"**Confidence:** {rl_signal['confidence']}")
                            
                            # Display RL performance metrics
                            if 'context' in rl_signal and rl_signal['context']:
                                context = rl_signal['context']
                                
                                # Performance metrics
                                st.markdown("### Performance Metrics")
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    st.metric("Total Return", f"{context.get('total_return', 0):.2f}%")
                                with metrics_col2:
                                    st.metric("Buy & Hold Return", f"{context.get('buy_hold_return', 0):.2f}%")
                                with metrics_col3:
                                    st.metric("Sharpe Ratio", f"{context.get('sharpe_ratio', 0):.4f}")
                                
                                # Latest portfolio state
                                st.markdown(f"**Latest Portfolio State (as of {context.get('latest_date', 'N/A')}):**")
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    st.metric("Portfolio Value", f"${context.get('portfolio_value', 0):.2f}")
                                with metrics_col2:
                                    st.metric("Stock Owned", f"{context.get('stock_owned', 0):.2f} units")
                                with metrics_col3:
                                    st.metric("Cash Balance", f"${context.get('cash_balance', 0):.2f}")
                                
                                # Display RL performance chart
                                st.markdown("### RL Agent Performance")
                                st.image(f"plots/{ticker_symbol}_rl_performance.png", use_column_width=True)
                                
                            # Interpretation
                            st.markdown("### Interpretation")
                            st.markdown("""
                            - The RL agent learns optimal trading strategies from historical price patterns
                            - **BUY** signals indicate the agent expects price increases
                            - **SELL** signals indicate the agent expects price decreases
                            - **HOLD** signals indicate the agent expects minimal price changes
                            - The agent's performance is compared to a simple buy & hold strategy
                            """)
                
                with col2:
                    # Display regression stats
                    st.subheader("Regression Statistics")
                    
                    # Extract key stats from model
                    model_summary = analyzer.model.summary()
                    r_squared = analyzer.model.rsquared
                    adj_r_squared = analyzer.model.rsquared_adj
                    
                    # Display metrics
                    st.metric("R-squared", f"{r_squared:.4f}")
                    st.metric("Adjusted R-squared", f"{adj_r_squared:.4f}")
                    
                    # Beta (slope coefficient)
                    beta = analyzer.model.params.iloc[1] if len(analyzer.model.params) > 1 else 0
                    st.metric("Beta", f"{beta:.4f}")
                    
                    # Alpha (intercept)
                    alpha = analyzer.model.params.iloc[0] if len(analyzer.model.params) > 0 else 0
                    st.metric("Alpha", f"{alpha:.4f}")
                
                # Create interactive plots with Plotly
                st.subheader("Interactive Charts")
                
                # Create tabs for different charts
                if use_segment_analysis and segment_tickers:
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price & Volume", "Residuals & Money Flow", "Regression", "Correlation Analysis", "Divergence Analysis"])
                else:
                    tab1, tab2, tab3 = st.tabs(["Price & Volume", "Residuals & Money Flow", "Regression"])
                
                with tab1:
                    # Price and volume chart
                    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add price line
                    fig1.add_trace(
                        go.Scatter(
                            x=analyzer.aligned_data.index,
                            y=analyzer.aligned_data[ticker_symbol],
                            name=f"{ticker_symbol} Price",
                            line=dict(color="green")
                        ),
                        secondary_y=False
                    )
                    
                    # Add volume bars
                    fig1.add_trace(
                        go.Bar(
                            x=analyzer.weekly_volume.index,
                            y=analyzer.weekly_volume,
                            name="Weekly Volume",
                            marker=dict(color="gray", opacity=0.5)
                        ),
                        secondary_y=True
                    )
                    
                    # Update layout
                    fig1.update_layout(
                        title=f"Stock Price Movement & Weekly Volume of {ticker_symbol}",
                        xaxis_title="Date",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    fig1.update_yaxes(title_text="Stock Price ($)", secondary_y=False, color="green")
                    fig1.update_yaxes(title_text="Weekly Volume", secondary_y=True, color="gray")
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with tab2:
                    # Residuals and money flow chart
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                    
                # Add segment correlation analysis tab if enabled
                if use_segment_analysis and segment_tickers and 'tab4' in locals():
                    with tab4:
                        st.subheader("Market Segment Correlation Analysis")
                        
                        # Display correlation matrix if available
                        if hasattr(analyzer, 'correlation_matrix') and analyzer.correlation_matrix is not None:
                            st.write("### Residual Correlation Matrix")
                            st.write("This matrix shows how the residuals of each stock correlate with others in the segment.")
                            st.dataframe(analyzer.correlation_matrix.style.background_gradient(cmap='coolwarm'))
                        
                        # Display correlation signals
                        if correlation_signals:
                            st.write("### Correlation-Based Signals")
                            st.write("Signals based on changes in correlation between stocks in the segment:")
                            
                            # Create a DataFrame for better display
                            signal_data = []
                            for ticker, signal_info in correlation_signals.items():
                                signal_data.append({
                                    "Ticker": ticker,
                                    "Signal": signal_info["signal"],
                                    "Latest Correlation": f"{signal_info['latest_correlation']:.3f}",
                                    "Correlation Change": f"{signal_info['correlation_change']:.3f}",
                                    "Latest Residual": f"{signal_info['latest_residual']:.3f}",
                                    "Latest Price": f"${signal_info['latest_price']:.2f}",
                                    "Reason": signal_info["reason"]
                                })
                            
                            if signal_data:
                                signal_df = pd.DataFrame(signal_data)
                                
                                # Apply styling based on signal
                                def style_signal(val):
                                    if "STRONG BUY" in val:
                                        return 'background-color: #CCFFCC; color: darkgreen; font-weight: bold'
                                    elif "BUY" in val:
                                        return 'background-color: #E6FFE6; color: green; font-weight: bold'
                                    elif "STRONG SELL" in val:
                                        return 'background-color: #FFCCCC; color: darkred; font-weight: bold'
                                    elif "SELL" in val:
                                        return 'background-color: #FFE6E6; color: red; font-weight: bold'
                                    elif val == "HOLD":
                                        return 'background-color: #FFF9CC; color: orange; font-weight: bold'
                                    return ''
                                
                                # Display styled dataframe
                                st.dataframe(signal_df.style.applymap(style_signal, subset=['Signal']))
                            else:
                                st.info("No correlation signals generated. This could be due to insufficient data or low correlation changes.")
                                
                # Add segment divergence analysis tab if enabled
                if use_segment_analysis and segment_tickers and 'tab5' in locals():
                    with tab5:
                        st.subheader("Residual Divergence Analysis")
                        st.write("### What is Residual Divergence Analysis?")
                        st.write("""
                        This analysis detects when a stock's behavior significantly deviates from the typical pattern of its sector or segment. 
                        Rather than using correlations, it measures how much each stock's residuals differ from the median behavior of the segment.
                        
                        - **High divergence + negative residual** = Potential BUY signal (stock falling unusually compared to peers)
                        - **High divergence + positive residual** = Potential SELL signal (stock rising unusually compared to peers)
                        """)
                        
                        # Display divergence signals
                        if divergence_signals:
                            st.write("### Divergence Signals")
                            st.write("Signals based on how each stock's residuals diverge from the segment median:")
                            
                            # Create a DataFrame for better display
                            signal_data = []
                            for ticker, signal_info in divergence_signals.items():
                                signal_data.append({
                                    "Ticker": ticker,
                                    "Signal": signal_info["signal"],
                                    "Z-Score": f"{signal_info['latest_z_score']:.2f}",
                                    "Residual": f"{signal_info['latest_residual']:.3f}",
                                    "Deviation from Segment": f"{signal_info['latest_deviation']:.3f}",
                                    "Latest Price": f"${signal_info['latest_price']:.2f}" if signal_info['latest_price'] else "N/A",
                                    "Reason": signal_info["reason"]
                                })
                            
                            if signal_data:
                                signal_df = pd.DataFrame(signal_data)
                                
                                # Apply styling based on signal
                                def style_divergence_signal(val):
                                    if "STRONG BUY" in val:
                                        return 'background-color: #CCFFCC; color: darkgreen; font-weight: bold'
                                    elif "BUY" in val:
                                        return 'background-color: #E6FFE6; color: green; font-weight: bold'
                                    elif "STRONG SELL" in val:
                                        return 'background-color: #FFCCCC; color: darkred; font-weight: bold'
                                    elif "SELL" in val:
                                        return 'background-color: #FFE6E6; color: red; font-weight: bold'
                                    elif val == "HOLD":
                                        return 'background-color: #FFF9CC; color: orange; font-weight: bold'
                                    return ''
                                
                                # Display styled dataframe
                                st.dataframe(signal_df.style.applymap(style_divergence_signal, subset=['Signal']))
                                
                                # Visualize the divergence of each stock from the segment median
                                st.write("### Visualizing Divergence from Segment")
                                
                                # Plot the most recent divergence z-scores
                                z_scores = {ticker: info['latest_z_score'] for ticker, info in divergence_signals.items()}
                                
                                if z_scores:
                                    # Create bar chart of z-scores
                                    fig_z = go.Figure()
                                    
                                    for ticker, z_score in z_scores.items():
                                        color = 'green' if divergence_signals[ticker]['latest_residual'] < 0 else 'red'
                                        fig_z.add_trace(go.Bar(
                                            x=[ticker],
                                            y=[z_score],
                                            name=ticker,
                                            marker_color=color
                                        ))
                                    
                                    fig_z.update_layout(
                                        title="Divergence Z-Scores by Ticker",
                                        xaxis_title="Ticker",
                                        yaxis_title="Z-Score",
                                        showlegend=False
                                    )
                                    
                                    # Add a horizontal line at the threshold
                                    fig_z.add_shape(
                                        type="line",
                                        x0=-0.5,
                                        y0=1.5,  # default threshold
                                        x1=len(z_scores) - 0.5,
                                        y1=1.5,
                                        line=dict(color="orange", width=2, dash="dash"),
                                    )
                                    
                                    st.plotly_chart(fig_z, use_container_width=True)
                            else:
                                st.info("No significant divergence signals generated.")
                            
                            # Plot historical correlations
                            if hasattr(analyzer, 'historical_correlations') and analyzer.historical_correlations:
                                st.write("### Historical Average Correlations")
                                st.write("This chart shows how each stock's correlation with the segment has changed over time:")
                                
                                # Create a figure for historical correlations
                                fig_corr = go.Figure()
                                
                                # Add a line for each ticker's historical correlation
                                for ticker, corr_series in analyzer.historical_correlations.items():
                                    if not corr_series.empty:
                                        fig_corr.add_trace(go.Scatter(
                                            x=corr_series.index,
                                            y=corr_series.values,
                                            mode='lines',
                                            name=ticker
                                        ))
                                
                                # Update layout
                                fig_corr.update_layout(
                                    title="Historical Correlations with Segment",
                                    xaxis_title="Date",
                                    yaxis_title="Average Correlation",
                                    height=500
                                )
                                
                                # Add a horizontal line at correlation = 0
                                fig_corr.add_shape(
                                    type="line",
                                    x0=min(corr_series.index) if not corr_series.empty else 0,
                                    y0=0,
                                    x1=max(corr_series.index) if not corr_series.empty else 1,
                                    y1=0,
                                    line=dict(color="gray", width=1, dash="dash")
                                )
                                
                                st.plotly_chart(fig_corr, use_container_width=True)
                        else:
                            st.info("No correlation signals available. This could be due to insufficient data or too few tickers in the segment.")
                    
                    # Add residuals line
                    fig2.add_trace(
                        go.Scatter(
                            x=analyzer.aligned_data.index,
                            y=analyzer.aligned_data["Residuals"],
                            name="Residuals",
                            line=dict(color="blue")
                        ),
                        secondary_y=False
                    )
                    
                    # Add horizontal line at y=0
                    fig2.add_shape(
                        type="line",
                        x0=analyzer.aligned_data.index[0],
                        y0=0,
                        x1=analyzer.aligned_data.index[-1],
                        y1=0,
                        line=dict(color="red", width=1, dash="dash")
                    )
                    
                    # Add money flow bars
                    fig2.add_trace(
                        go.Bar(
                            x=analyzer.weekly_money_flow.index,
                            y=analyzer.weekly_money_flow,
                            name="Weekly Money Flow",
                            marker=dict(color="gray", opacity=0.5)
                        ),
                        secondary_y=True
                    )
                    
                    # Update layout
                    fig2.update_layout(
                        title=f"Residual Price Movements & Weekly Money Flow for {ticker_symbol}",
                        xaxis_title="Date",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    fig2.update_yaxes(title_text="Residuals", secondary_y=False, color="blue")
                    fig2.update_yaxes(title_text="Weekly Money Flow ($)", secondary_y=True, color="gray")
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    # Regression scatter plot
                    fig3 = go.Figure()
                    
                    # Add scatter plot
                    fig3.add_trace(
                        go.Scatter(
                            x=analyzer.aligned_data["Market"],
                            y=analyzer.aligned_data[ticker_symbol],
                            mode="markers",
                            name="Data Points",
                            marker=dict(color="blue", opacity=0.6)
                        )
                    )
                    
                    # Add regression line
                    x_range = [analyzer.aligned_data["Market"].min(), analyzer.aligned_data["Market"].max()]
                    y_range = [alpha + beta * x for x in x_range]
                    
                    fig3.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_range,
                            mode="lines",
                            name="Regression Line",
                            line=dict(color="red")
                        )
                    )
                    
                    # Update layout
                    fig3.update_layout(
                        title=f"Regression: {ticker_symbol} vs {market_selection}",
                        xaxis_title=f"{market_selection} Price ($)",
                        yaxis_title=f"{ticker_symbol} Price ($)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Display full regression summary in expander
                with st.expander("View Full Regression Summary"):
                    st.text(str(analyzer.model.summary()))
                
                # Historical signals
                st.subheader("Historical Signals")
                
                # Calculate historical signals
                historical_data = analyzer.aligned_data.copy()
                historical_data["Signal"] = "UNKNOWN"
                
                for i in range(lookback_period, len(historical_data)):
                    # Calculate rolling stats
                    lookback_slice = historical_data["Residuals"].iloc[i-lookback_period:i]
                    avg_residual = lookback_slice.mean()
                    std_residual = lookback_slice.std()
                    
                    current_residual = historical_data["Residuals"].iloc[i]
                    z_score = (current_residual - avg_residual) / std_residual if std_residual > 0 else 0
                    
                    # Determine signal
                    if current_residual < residual_threshold:
                        if z_score < -1.0:
                            historical_data.loc[historical_data.index[i], "Signal"] = "STRONG BUY"
                        else:
                            historical_data.loc[historical_data.index[i], "Signal"] = "BUY"
                    elif current_residual > abs(residual_threshold):
                        if z_score > 1.0:
                            historical_data.loc[historical_data.index[i], "Signal"] = "STRONG SELL"
                        else:
                            historical_data.loc[historical_data.index[i], "Signal"] = "SELL"
                    else:
                        historical_data.loc[historical_data.index[i], "Signal"] = "HOLD"
                
                # Display historical signals table
                signal_df = historical_data[["Signal", ticker_symbol, "Residuals"]].tail(30).copy()
                signal_df.columns = ["Signal", "Price", "Residual"]
                signal_df.index = signal_df.index.strftime("%Y-%m-%d")
                signal_df = signal_df.sort_index(ascending=False)
                
                st.dataframe(signal_df, use_container_width=True)
                
                # Download data button
                csv_data = historical_data[[ticker_symbol, "Market", "Residuals", "Signal"]].to_csv().encode("utf-8")
                st.download_button(
                    label="Download Analysis Data",
                    data=csv_data,
                    file_name=f"{ticker_symbol}_analysis.csv",
                    mime="text/csv"
                )
            
            else:
                st.error(f"Unable to analyze {ticker_symbol}. Insufficient data available.")
    
    except Exception as e:
        st.error(f"Error analyzing {ticker_symbol}: {str(e)}")
        st.info("Please check if the ticker symbol is valid and try again.")

else:
    # Display placeholder content
    if use_segment_analysis:
        st.info("👈 Enter a main stock ticker, segment tickers, and analysis parameters, then click 'Run Analysis'")
    else:
        st.info("👈 Enter a stock ticker and analysis parameters, then click 'Run Analysis'")
    
    # Display sample image
    st.image("https://www.investopedia.com/thmb/eG-Ym9Sf9-Q3CYpQFdQBwJ7WFpk=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dotdash_Final_Regression_Analysis_Sep_2020-01-f8b7f9b1536b46918178e1a6a6d15f2b.jpg", 
             caption="Sample Regression Analysis")
    
    # Explanation
    st.markdown("""
    ### How It Works
    
    This tool uses **linear regression** to analyze how a stock performs relative to a market benchmark.
    
    1. **Residuals** are the differences between the actual stock price and the predicted price based on the market.
    2. **Negative residuals** suggest the stock is trading below its expected value (potentially undervalued).
    3. **Positive residuals** suggest the stock is trading above its expected value (potentially overvalued).
    
    The tool provides buy/sell signals based on these residuals, considering both the absolute value and the recent trend.
    """)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Python, Streamlit, and yfinance")
