#!/usr/bin/env python3
"""
Test Script for Streamlit Dashboard
===================================

Quick test to verify all components work correctly before deployment.
"""

import sys
import subprocess
import importlib
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    required_modules = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'plotly.express',
        'plotly.graph_objects'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Install missing modules with: pip install -r streamlit_requirements.txt")
        return False
    
    print("✅ All imports successful!")
    return True

def test_sentiment_analyzer():
    """Test that the sentiment analyzer works."""
    print("\n🔍 Testing sentiment analyzer...")
    
    try:
        from simplified_demo import SimplifiedSentimentAnalyzer
        
        analyzer = SimplifiedSentimentAnalyzer()
        data = analyzer.load_sample_data()
        analyzed_data = analyzer.simple_sentiment_analysis()
        
        print(f"  ✅ Generated {len(data)} sample entries")
        print(f"  ✅ Analyzed sentiment for all entries")
        
        # Test service analysis
        service_analysis = analyzer.analyze_service_patterns()
        print(f"  ✅ Analyzed {len(service_analysis)} service types")
        
        # Test insights generation
        insights = analyzer.generate_insights()
        print(f"  ✅ Generated insights: {insights['total_feedback']} total feedback")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing sentiment analyzer: {e}")
        return False

def test_streamlit_app():
    """Test that the Streamlit app can be loaded."""
    print("\n🔍 Testing Streamlit app...")
    
    try:
        # Check if the app file exists
        app_file = Path("streamlit_app_integrated.py")
        if not app_file.exists():
            print("  ❌ streamlit_app_integrated.py not found")
            return False
        
        print("  ✅ streamlit_app_integrated.py found")
        
        # Try to import the main function
        import streamlit_app_integrated
        print("  ✅ App module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing Streamlit app: {e}")
        return False

def test_config_files():
    """Test that configuration files exist."""
    print("\n🔍 Testing configuration files...")
    
    config_files = [
        "streamlit_requirements.txt",
        ".streamlit/config.toml",
        "run_dashboard.py"
    ]
    
    all_exist = True
    
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} not found")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("🏥 Healthcare Sentiment Analysis Dashboard - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Sentiment Analyzer Test", test_sentiment_analyzer),
        ("Streamlit App Test", test_streamlit_app),
        ("Configuration Files Test", test_config_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your dashboard is ready to deploy.")
        print("\nNext steps:")
        print("1. Run: python run_dashboard.py")
        print("2. Or: streamlit run streamlit_app_integrated.py")
        print("3. Deploy to Streamlit Cloud using streamlit_deployment_guide.md")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before deploying.")
        print("\nCommon solutions:")
        print("- Install requirements: pip install -r streamlit_requirements.txt")
        print("- Check file paths and permissions")
        print("- Verify Python version compatibility")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
