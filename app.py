import streamlit as st
import subprocess
import sys
import os
from datetime import datetime
import pandas as pd

def run_analysis(script_name, ticker, benchmark, days, min_events, additional_args=None):
    """Run the selected analysis script with given parameters"""
    cmd = [sys.executable, script_name, ticker, "--benchmark", benchmark, "--days", str(days), "--min-events", str(min_events)]
    
    if additional_args:
        cmd.extend(additional_args)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def main():
    st.set_page_config(page_title="Stock Analysis Tool", page_icon="📈", layout="wide")
    
    st.title("📈 Stock Analysis Tool")
    st.markdown("Choose between two analysis approaches and configure parameters")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Script selection
        script_choice = st.selectbox(
            "Select Analysis Approach",
            options=["funny.py", "onlyant.py"],
            help="funny.py: Standard earnings analysis\nonlyant.py: Anticipation events only analysis"
        )
        
        # Input parameters
        ticker = st.text_input("Stock Ticker", value="GMAB", help="Stock symbol to analyze")
        benchmark = st.text_input("Benchmark Ticker", value="SPY", help="Market benchmark for comparison")
        days = st.number_input("Days to Analyze", min_value=30, max_value=2000, value=500, help="Historical data period")
        min_events = st.number_input("Target Events", min_value=1, max_value=50, value=20, help="Number of events to identify")
        
        # Additional parameters for onlyant.py
        if script_choice == "onlyant.py":
            st.subheader("Advanced Options")
            future_days = st.number_input("Future Days Lookhead", min_value=1, max_value=10, value=1, help="Days to look ahead for delayed anticipation")
            use_intraday = st.checkbox("Use Intraday Data", help="Enable intraday analysis features")
            
            if use_intraday:
                intraday_timespan = st.selectbox("Intraday Timespan", ["minute", "hour"], index=0)
                intraday_multiplier = st.number_input("Intraday Multiplier", min_value=1, max_value=60, value=15, help="E.g., 15 for 15-minute bars")
        
        # Run button
        run_analysis_btn = st.button("🚀 Run Analysis", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Analysis Output")
        
        # Create placeholder for output
        output_placeholder = st.empty()
        
        if run_analysis_btn:
            with st.spinner("Running analysis..."):
                # Prepare additional arguments for onlyant.py
                additional_args = []
                if script_choice == "onlyant.py":
                    additional_args.extend(["--future-days", str(future_days)])
                    if use_intraday:
                        additional_args.append("--use-intraday")
                        additional_args.extend(["--intraday-timespan", intraday_timespan])
                        additional_args.extend(["--intraday-multiplier", str(intraday_multiplier)])
                
                # Run the analysis
                stdout, stderr, returncode = run_analysis(script_choice, ticker, benchmark, days, min_events, additional_args)
                
                # Display results
                if returncode == 0:
                    st.success("Analysis completed successfully!")
                    with output_placeholder.container():
                        st.text_area("Output", stdout, height=400)
                        
                        # Check for generated files
                        st.subheader("Generated Files")
                        
                        # Look for CSV files
                        csv_pattern = f"{ticker}_analysis_*.csv"
                        import glob
                        csv_files = glob.glob(csv_pattern)
                        
                        # Look for PNG files
                        png_files = []
                        if script_choice == "funny.py":
                            png_files = glob.glob(f"data/{ticker}_analysis.png")
                        else:
                            png_files = glob.glob(f"data/{ticker}_anticipation_only_analysis.png")
                        
                        # Look for Parquet files (onlyant.py)
                        parquet_files = []
                        if script_choice == "onlyant.py":
                            parquet_files = glob.glob(f"data/{ticker}_features_*.parquet")
                        
                        # Display file information
                        if csv_files:
                            st.write("📊 CSV Results:")
                            for file in csv_files:
                                st.write(f"- {file}")
                        
                        if png_files:
                            st.write("📈 Plot Files:")
                            for file in png_files:
                                st.write(f"- {file}")
                                # Display the plot
                                if os.path.exists(file):
                                    st.image(file, caption=f"Analysis Plot: {file}")
                        
                        if parquet_files:
                            st.write("🔬 ML Feature Files:")
                            for file in parquet_files:
                                st.write(f"- {file}")
                else:
                    st.error("Analysis failed!")
                    with output_placeholder.container():
                        if stderr:
                            st.text_area("Error Output", stderr, height=200)
                        if stdout:
                            st.text_area("Standard Output", stdout, height=200)
        else:
            with output_placeholder.container():
                st.info("Configure parameters and click 'Run Analysis' to start")
    
    with col2:
        st.header("Script Information")
        
        if script_choice == "funny.py":
            st.markdown("""
            **funny.py - Standard Analysis**
            
            - Identifies earnings events using volume × gap formula
            - Classifies events as:
              - Surprising positive/negative
              - Positive/negative anticipated
            - Market-filtered movements using residual returns
            - Comprehensive visualization with 5 plots
            """)
        else:
            st.markdown("""
            **onlyant.py - Anticipation Events Only**
            
            - Focuses only on anticipation events
            - Filters out non-anticipation events
            - Enhanced future anticipation detection
            - Intraday data support for additional features
            - ML-ready feature export to Parquet
            - Filters events with non-negative residual returns
            """)
        
        st.subheader("Current Parameters")
        st.json({
            "script": script_choice,
            "ticker": ticker,
            "benchmark": benchmark,
            "days": days,
            "min_events": min_events,
            **({"future_days": future_days, "use_intraday": use_intraday} if script_choice == "onlyant.py" else {})
        })

if __name__ == "__main__":
    main()