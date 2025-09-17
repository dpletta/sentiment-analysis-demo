#!/usr/bin/env python3
"""
Healthcare Sentiment Analysis Dashboard - Setup Script
=====================================================

Automated setup script for the healthcare sentiment analysis dashboard.
Supports multiple deployment modes and automatic dependency management.

Author: AI Assistant
Date: September 2025
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header."""
    print("=" * 60)
    print("🏥 Healthcare Sentiment Analysis Dashboard Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", f"{version.major}.{version.minor}")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment."""
    print("\n🔧 Setting up virtual environment...")
    
    venv_name = "streamlit_env"
    
    if os.path.exists(venv_name):
        print(f"✅ Virtual environment '{venv_name}' already exists")
        return venv_name
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print(f"✅ Virtual environment '{venv_name}' created successfully")
        return venv_name
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        sys.exit(1)

def get_pip_command(venv_name):
    """Get the correct pip command for the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(venv_name, "Scripts", "pip")
    else:
        return os.path.join(venv_name, "bin", "pip")

def install_dependencies(venv_name, use_full_requirements=False):
    """Install dependencies."""
    print("\n📦 Installing dependencies...")
    
    pip_cmd = get_pip_command(venv_name)
    
    # Upgrade pip first
    try:
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        print("✅ pip upgraded successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to upgrade pip: {e}")
    
    # Choose requirements file
    if use_full_requirements and os.path.exists("requirements-full.txt"):
        req_file = "requirements-full.txt"
        print("🤖 Installing with full AI capabilities...")
    else:
        req_file = "requirements.txt"
        print("📊 Installing basic requirements (AI features will use fallback)...")
    
    if not os.path.exists(req_file):
        print(f"❌ Requirements file '{req_file}' not found")
        sys.exit(1)
    
    try:
        subprocess.run([pip_cmd, "install", "-r", req_file], check=True)
        print(f"✅ Dependencies installed from {req_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def download_nltk_data():
    """Download required NLTK data."""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except ImportError:
        print("⚠️ NLTK not available, skipping data download")
    except Exception as e:
        print(f"⚠️ Failed to download NLTK data: {e}")

def create_directories():
    """Create required directories."""
    print("\n📁 Creating required directories...")
    
    directories = ["demo_output", "ui_reference"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory '{directory}' ready")

def test_installation(venv_name):
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    python_cmd = os.path.join(venv_name, "bin", "python") if platform.system() != "Windows" else os.path.join(venv_name, "Scripts", "python")
    
    tests = [
        ("Streamlit app", "from streamlit_app import main; print('✅ Streamlit app works')"),
        ("Sentiment analyzer", "from simplified_demo import SimplifiedSentimentAnalyzer; print('✅ Sentiment analyzer works')"),
        ("Simplified chatbot", "from simple_ai_chatbot import SimpleHealthcareChatbot; print('✅ Simplified chatbot works')"),
    ]
    
    for test_name, test_code in tests:
        try:
            result = subprocess.run([python_cmd, "-c", test_code], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {test_name} test passed")
            else:
                print(f"⚠️ {test_name} test failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⚠️ {test_name} test timed out")
        except Exception as e:
            print(f"⚠️ {test_name} test error: {e}")

def print_completion_message(venv_name):
    """Print completion message with instructions."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print()
    
    if platform.system() == "Windows":
        print(f"1. Activate virtual environment:")
        print(f"   {venv_name}\\Scripts\\activate")
    else:
        print(f"1. Activate virtual environment:")
        print(f"   source {venv_name}/bin/activate")
    
    print()
    print("2. Run the dashboard:")
    print("   streamlit run streamlit_app_integrated.py")
    print()
    print("3. Open your browser to: http://localhost:8501")
    print()
    print("🔒 HIPAA Compliance:")
    print("- All processing happens locally")
    print("- No external data transmission")
    print("- Use offline mode for maximum compliance")
    print()
    print("📚 Documentation:")
    print("- README.md: General information")
    print("- TROUBLESHOOTING.md: Common issues and solutions")
    print("- HIPAA_COMPLIANCE.md: Compliance details")
    print()

def main():
    """Main setup function."""
    print_header()
    
    # Check for command line arguments
    use_full_requirements = "--full" in sys.argv or "--ai" in sys.argv
    
    # Run setup steps
    check_python_version()
    venv_name = create_virtual_environment()
    install_dependencies(venv_name, use_full_requirements)
    download_nltk_data()
    create_directories()
    test_installation(venv_name)
    print_completion_message(venv_name)

if __name__ == "__main__":
    main()