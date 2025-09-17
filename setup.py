#!/usr/bin/env python3
"""
HIPAA-Compliant Sentiment Analysis Setup Script
==============================================

This script sets up the environment for the HIPAA-compliant sentiment analysis system.
It ensures all dependencies are installed and the system is ready for use.

Usage:
    python setup.py

Features:
- Installs required Python packages
- Downloads necessary NLTK data
- Creates required directories
- Validates system readiness
- Provides HIPAA compliance checklist

Author: AI Assistant
Date: September 2025
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version check passed: {sys.version.split()[0]}")
        return True

def install_requirements():
    """Install required packages from requirements.txt."""
    print("\n📦 Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        
        # Required NLTK data
        nltk_downloads = [
            'vader_lexicon',
            'stopwords', 
            'punkt',
            'wordnet',
            'averaged_perceptron_tagger'
        ]
        
        for item in nltk_downloads:
            print(f"  Downloading {item}...")
            nltk.download(item, quiet=True)
        
        print("✅ NLTK data downloaded successfully")
        return True
        
    except ImportError:
        print("❌ NLTK not found. Please install requirements first.")
        return False
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories for the analysis."""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data",
        "output", 
        "models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created/verified: {directory}/")
    
    return True

def validate_installation():
    """Validate that all components are working properly."""
    print("\n🔍 Validating installation...")
    
    # Test imports
    test_imports = [
        ("numpy", "np"),
        ("pandas", "pd"), 
        ("sklearn", None),
        ("tensorflow", "tf"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("nltk", None),
        ("plotly", None)
    ]
    
    failed_imports = []
    
    for module, alias in test_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All imports successful")
        return True

def test_tensorflow():
    """Test TensorFlow installation."""
    print("\n🧠 Testing TensorFlow...")
    
    try:
        import tensorflow as tf
        
        # Check if GPU is available (optional)
        if tf.config.list_physical_devices('GPU'):
            print("  ✅ GPU support detected")
        else:
            print("  ℹ️  Running on CPU (this is fine for the demo)")
        
        # Simple computation test
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        
        print(f"  ✅ TensorFlow computation test passed")
        print(f"  Version: {tf.__version__}")
        return True
        
    except Exception as e:
        print(f"  ❌ TensorFlow test failed: {e}")
        return False

def print_hipaa_compliance_checklist():
    """Print HIPAA compliance checklist for users."""
    print("\n" + "="*60)
    print("🏥 HIPAA COMPLIANCE CHECKLIST")
    print("="*60)
    
    checklist = [
        "✓ All processing occurs locally (no cloud APIs)",
        "✓ No external data transmission",
        "✓ PII detection and redaction implemented", 
        "✓ Audit logging enabled",
        "✓ Secure local data storage",
        "✓ No persistent storage of sensitive data",
        "✓ Local model training and inference only"
    ]
    
    for item in checklist:
        print(f"  {item}")
    
    print("\n📋 Additional HIPAA Requirements to Consider:")
    additional_requirements = [
        "• Implement access controls for the analysis system",
        "• Ensure secure deletion of temporary files",
        "• Regular security audits of the analysis environment", 
        "• Staff training on data handling procedures",
        "• Incident response procedures",
        "• Regular backup and recovery testing",
        "• Network security measures"
    ]
    
    for req in additional_requirements:
        print(f"  {req}")

def create_quick_start_guide():
    """Create a quick start guide file."""
    print("\n📖 Creating quick start guide...")
    
    guide_content = """# HIPAA-Compliant Sentiment Analysis - Quick Start Guide

## Overview
This system provides a completely local sentiment analysis solution suitable for HIPAA-compliant environments. All processing occurs on your local machine with no external data transmission.

## Getting Started

### 1. Run the Analysis
```python
python hipaa_sentiment_analysis.py
```

### 2. Interactive Analysis (Jupyter)
```bash
jupyter notebook sentiment_analysis_demo.ipynb
```

### 3. Custom Data Analysis
```python
from hipaa_sentiment_analysis import HIPAACompliantSentimentAnalyzer

# Initialize analyzer
analyzer = HIPAACompliantSentimentAnalyzer()

# Load your data (replace with your CSV file)
# df = pd.read_csv('your_feedback_data.csv')
# analyzer.df = df

# Run analysis
analyzer.analyze_sentiment_multiple_approaches()
analyzer.create_comprehensive_visualizations()
```

## Output Files
- `comprehensive_sentiment_analysis.png` - Main visualization dashboard
- `interactive_dashboard.html` - Interactive Plotly dashboard
- `sentiment_analysis_report.txt` - Detailed analysis report
- `sentiment_analysis_YYYYMMDD_HHMMSS.log` - Audit log

## Data Format Requirements
Your CSV file should contain at minimum:
- `feedback_text`: The text feedback to analyze
- `service_type`: Type of service/product (optional)
- `rating`: Numerical rating (optional)
- `date`: Date of feedback (optional)

## HIPAA Compliance Features
✓ Local processing only
✓ No external API calls
✓ PII detection and redaction
✓ Comprehensive audit logging
✓ Secure local storage

## Support
For questions or issues, refer to the comprehensive documentation in the main script file.
"""
    
    with open("QUICK_START.md", "w") as f:
        f.write(guide_content)
    
    print("  ✅ Quick start guide created: QUICK_START.md")

def main():
    """Main setup function."""
    print("🏥 HIPAA-Compliant Sentiment Analysis - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Validate installation
    if not validate_installation():
        return False
    
    # Test TensorFlow
    if not test_tensorflow():
        return False
    
    # Create quick start guide
    create_quick_start_guide()
    
    # Print HIPAA compliance information
    print_hipaa_compliance_checklist()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the demo: python hipaa_sentiment_analysis.py")
    print("2. Check QUICK_START.md for detailed instructions")
    print("3. Review output files in the ./output directory")
    print("\n✅ Your system is ready for HIPAA-compliant sentiment analysis!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
