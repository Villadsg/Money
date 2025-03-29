import streamlit.web.bootstrap
from pathlib import Path
import os
import sys

# Get the directory of the current file
current_dir = Path(__file__).parent.parent

# Add the project root to the path
sys.path.insert(0, str(current_dir))

# Point to the app.py file
app_path = str(current_dir / "app.py")

# Set Streamlit configuration
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"

# Function to run the Streamlit app
def handler(event, context):
    streamlit.web.bootstrap.run(app_path, "", [], flag_options={})
    return {"statusCode": 200, "body": "Streamlit app is running"}
