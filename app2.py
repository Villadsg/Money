import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import sys
import os
import glob
from PIL import Image, ImageTk
import threading

class StockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Analysis Tool")
        self.root.geometry("1400x1000")
        
        # Variables
        self.script_var = tk.StringVar(value="funny.py")
        self.ticker_var = tk.StringVar(value="GMAB")
        self.benchmark_var = tk.StringVar(value="SPY")
        self.days_var = tk.IntVar(value=500)
        self.min_events_var = tk.IntVar(value=20)
        self.future_days_var = tk.IntVar(value=1)
        self.use_intraday_var = tk.BooleanVar(value=False)
        self.intraday_timespan_var = tk.StringVar(value="minute")
        self.intraday_multiplier_var = tk.IntVar(value=15)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Left panel - Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, rowspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Script selection
        ttk.Label(config_frame, text="Analysis Script:").grid(row=0, column=0, sticky=tk.W, pady=5)
        script_frame = ttk.Frame(config_frame)
        script_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Radiobutton(script_frame, text="funny.py", variable=self.script_var, value="funny.py", command=self.on_script_change).pack(anchor=tk.W)
        ttk.Radiobutton(script_frame, text="onlyant.py", variable=self.script_var, value="onlyant.py", command=self.on_script_change).pack(anchor=tk.W)
        
        # Basic parameters
        ttk.Label(config_frame, text="Stock Ticker:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.ticker_var, width=20).grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(config_frame, text="Benchmark:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.benchmark_var, width=20).grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(config_frame, text="Days to Analyze:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.days_var, width=20).grid(row=7, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(config_frame, text="Target Events:").grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.min_events_var, width=20).grid(row=9, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Advanced options frame (for onlyant.py)
        self.advanced_frame = ttk.LabelFrame(config_frame, text="Advanced Options", padding="5")
        self.advanced_frame.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(self.advanced_frame, text="Future Days:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.advanced_frame, textvariable=self.future_days_var, width=20).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Checkbutton(self.advanced_frame, text="Use Intraday Data", variable=self.use_intraday_var, command=self.on_intraday_change).grid(row=2, column=0, sticky=tk.W, pady=5)
        
        self.intraday_frame = ttk.Frame(self.advanced_frame)
        self.intraday_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(self.intraday_frame, text="Timespan:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(self.intraday_frame, textvariable=self.intraday_timespan_var, values=["minute", "hour"], width=17).grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(self.intraday_frame, text="Multiplier:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(self.intraday_frame, textvariable=self.intraday_multiplier_var, width=20).grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Run button
        ttk.Button(config_frame, text="🚀 Run Analysis", command=self.run_analysis_threaded).grid(row=11, column=0, sticky=(tk.W, tk.E), pady=20)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(config_frame, textvariable=self.status_var, foreground="blue").grid(row=12, column=0, sticky=tk.W, pady=5)
        
        # Right panel - Results
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=0, column=1, rowspan=4, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(3, weight=1)  # Output text
        results_frame.rowconfigure(4, weight=2)  # Image gets more space
        
        # Command preview
        ttk.Label(results_frame, text="Command:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.command_text = tk.Text(results_frame, height=2, wrap=tk.WORD)
        self.command_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Output text
        ttk.Label(results_frame, text="Output:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        self.output_text = scrolledtext.ScrolledText(results_frame, height=10, wrap=tk.WORD)
        self.output_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Image frame with scrollable canvas
        self.image_frame = ttk.LabelFrame(results_frame, text="Analysis Plot", padding="10")
        self.image_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Create canvas with scrollbars for large images
        self.canvas_frame = ttk.Frame(self.image_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.image_canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Add initial text
        self.image_canvas.create_text(200, 100, text="Plot will appear here after analysis", anchor=tk.CENTER)
        
        # Files info
        self.files_frame = ttk.LabelFrame(results_frame, text="Generated Files", padding="10")
        self.files_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.files_text = tk.Text(self.files_frame, height=4, wrap=tk.WORD)
        self.files_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize UI state
        self.on_script_change()
        self.on_intraday_change()
        self.update_command_preview()
        
        # Bind events to update command preview
        for var in [self.script_var, self.ticker_var, self.benchmark_var, self.days_var, 
                   self.min_events_var, self.future_days_var, self.use_intraday_var,
                   self.intraday_timespan_var, self.intraday_multiplier_var]:
            if isinstance(var, tk.StringVar):
                var.trace('w', lambda *args: self.update_command_preview())
            else:
                var.trace('w', lambda *args: self.update_command_preview())
    
    def on_script_change(self):
        """Show/hide advanced options based on script selection"""
        if self.script_var.get() == "onlyant.py":
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()
    
    def on_intraday_change(self):
        """Show/hide intraday options based on checkbox"""
        if self.use_intraday_var.get():
            self.intraday_frame.grid()
        else:
            self.intraday_frame.grid_remove()
    
    def update_command_preview(self):
        """Update the command preview text"""
        cmd = self.build_command()
        self.command_text.delete(1.0, tk.END)
        self.command_text.insert(1.0, " ".join(cmd))
    
    def build_command(self):
        """Build the command based on current inputs"""
        cmd = [sys.executable, self.script_var.get(), self.ticker_var.get()]
        cmd.extend(["--benchmark", self.benchmark_var.get()])
        cmd.extend(["--days", str(self.days_var.get())])
        cmd.extend(["--min-events", str(self.min_events_var.get())])
        
        if self.script_var.get() == "onlyant.py":
            cmd.extend(["--future-days", str(self.future_days_var.get())])
            if self.use_intraday_var.get():
                cmd.append("--use-intraday")
                cmd.extend(["--intraday-timespan", self.intraday_timespan_var.get()])
                cmd.extend(["--intraday-multiplier", str(self.intraday_multiplier_var.get())])
        
        return cmd
    
    def cleanup_previous_data(self, ticker):
        """Delete previous datasets and analysis files for the specified ticker"""
        patterns = [
            f"{ticker}_analysis_*.csv",
            f"{ticker}_*.csv", 
            f"data/{ticker}_*.csv",
            f"data/{ticker}_*.parquet",
            f"data/{ticker}_analysis.png",
            f"data/{ticker}_anticipation_only_analysis.png"
        ]
        
        deleted_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    os.remove(file)
                    deleted_files.append(file)
                except OSError:
                    pass
        
        return deleted_files
    
    def run_analysis_threaded(self):
        """Run analysis in a separate thread to avoid blocking UI"""
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        """Run the analysis and update UI with results"""
        try:
            ticker = self.ticker_var.get()
            
            # Update status
            self.status_var.set("Cleaning up previous files...")
            self.root.update()
            
            # Clean up previous files
            deleted_files = self.cleanup_previous_data(ticker)
            cleanup_msg = f"Cleaned up {len(deleted_files)} previous files for {ticker}"
            
            # Update output
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"=== CLEANUP ===\n{cleanup_msg}\n")
            if deleted_files:
                self.output_text.insert(tk.END, "Deleted files:\n")
                for file in deleted_files:
                    self.output_text.insert(tk.END, f"  - {file}\n")
            self.output_text.insert(tk.END, "\n")
            self.root.update()
            
            # Update status
            self.status_var.set("Running analysis...")
            self.root.update()
            
            # Build and run command
            cmd = self.build_command()
            self.output_text.insert(tk.END, f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
            self.output_text.insert(tk.END, "=== OUTPUT ===\n")
            self.root.update()
            
            # Run the analysis
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), timeout=300)
            
            # Display output
            self.output_text.insert(tk.END, result.stdout)
            if result.stderr:
                self.output_text.insert(tk.END, f"\n=== ERRORS ===\n{result.stderr}")
            
            self.output_text.insert(tk.END, f"\n=== Return Code: {result.returncode} ===\n")
            
            if result.returncode == 0:
                self.status_var.set("Analysis completed successfully!")
                self.display_results(ticker)
            else:
                self.status_var.set("Analysis failed!")
                
        except subprocess.TimeoutExpired:
            self.output_text.insert(tk.END, "\nAnalysis timed out after 5 minutes.")
            self.status_var.set("Analysis timed out!")
        except Exception as e:
            self.output_text.insert(tk.END, f"\nError: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
        
        # Scroll to bottom
        self.output_text.see(tk.END)
        self.root.update()
    
    def display_results(self, ticker):
        """Display generated files and plot"""
        # Look for generated files
        csv_files = glob.glob(f"{ticker}_analysis_*.csv")
        
        script = self.script_var.get()
        png_files = []
        if script == "funny.py":
            png_files = glob.glob(f"data/{ticker}_analysis.png")
        else:
            png_files = glob.glob(f"data/{ticker}_anticipation_only_analysis.png")
        
        parquet_files = []
        if script == "onlyant.py":
            parquet_files = glob.glob(f"data/{ticker}_features_*.parquet")
        
        # Update files info
        self.files_text.delete(1.0, tk.END)
        
        if csv_files:
            self.files_text.insert(tk.END, "📊 CSV Files:\n")
            for file in csv_files:
                try:
                    import pandas as pd
                    df = pd.read_csv(file)
                    self.files_text.insert(tk.END, f"  • {file} ({len(df)} rows, {len(df.columns)} columns)\n")
                except:
                    self.files_text.insert(tk.END, f"  • {file}\n")
        
        if png_files:
            self.files_text.insert(tk.END, "\n📈 Plot Files:\n")
            for file in png_files:
                file_size = os.path.getsize(file) / 1024
                self.files_text.insert(tk.END, f"  • {file} ({file_size:.1f} KB)\n")
        
        if parquet_files:
            self.files_text.insert(tk.END, "\n🔬 ML Feature Files:\n")
            for file in parquet_files:
                try:
                    import pandas as pd
                    df = pd.read_parquet(file)
                    self.files_text.insert(tk.END, f"  • {file} ({len(df)} rows, {len(df.columns)} columns)\n")
                except:
                    self.files_text.insert(tk.END, f"  • {file}\n")
        
        # Display plot
        if png_files:
            self.display_plot(png_files[0])
        else:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(400, 200, text="No plot file generated", anchor=tk.CENTER)
    
    def display_plot(self, image_path):
        """Display the generated plot in the canvas"""
        try:
            # Clear the canvas
            self.image_canvas.delete("all")
            
            # Open and resize image to fit in a larger space
            image = Image.open(image_path)
            
            # Get canvas size
            self.root.update_idletasks()  # Make sure canvas is rendered
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Use a reasonable minimum size if canvas isn't rendered yet
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600
            
            # Calculate new size - make it bigger but respect canvas size
            max_width = max(canvas_width, 1200)  # At least 1200px wide
            max_height = max(canvas_height, 900)  # At least 900px tall
            
            # Scale the image to be larger while maintaining aspect ratio
            ratio = min(max_width / image.width, max_height / image.height)
            # Make it at least 80% of original size for better visibility
            ratio = max(ratio, 0.8)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Add image to canvas
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.image_canvas.image = photo  # Keep a reference
            
            # Update scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(400, 200, text=f"Error loading image: {str(e)}", anchor=tk.CENTER)

def main():
    # Check if required packages are available
    try:
        import PIL
    except ImportError:
        print("PIL (Pillow) is required for image display. Install it with: pip install Pillow")
        return
    
    root = tk.Tk()
    app = StockAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()