#!/usr/bin/env python3
"""
Streamlit App Launcher
======================

Simple script to launch the Healthcare Sentiment Analysis Dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    print("🏥 Healthcare Sentiment Analysis Dashboard")
    print("=" * 50)
    print("Starting Streamlit application...")
    print("The app will open in your default web browser.")
    print("Press Ctrl+C to stop the application.")
    print("=" * 50)
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app_integrated.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
        print("\nMake sure you have installed the requirements:")
        print("pip install -r streamlit_requirements.txt")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it:")
        print("pip install streamlit")

if __name__ == "__main__":
    main()
