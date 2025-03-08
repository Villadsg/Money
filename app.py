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
                
                # Get correlation signals if segment analysis is enabled
                correlation_signals = None
                if use_segment_analysis and segment_tickers:
                    correlation_signals = analyzer.get_segment_correlation_signals(
                        lookback_period=lookback_period,
                        correlation_threshold=0.3  # Default threshold
                    )
                
                # Display results in columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Analysis Results")
                    
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
                    beta = analyzer.model.params[1]
                    st.metric("Beta", f"{beta:.4f}")
                    
                    # Alpha (intercept)
                    alpha = analyzer.model.params[0]
                    st.metric("Alpha", f"{alpha:.4f}")
                
                # Create interactive plots with Plotly
                st.subheader("Interactive Charts")
                
                # Create tabs for different charts
                if use_segment_analysis and segment_tickers:
                    tab1, tab2, tab3, tab4 = st.tabs(["Price & Volume", "Residuals & Money Flow", "Regression", "Segment Analysis"])
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
                    
                # Add segment analysis tab if enabled
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
