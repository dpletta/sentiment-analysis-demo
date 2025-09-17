#!/bin/bash

# Run the integrated HIPAA-compliant sentiment analysis dashboard

echo "🏥 Healthcare Sentiment Analysis Dashboard"
echo "=========================================="
echo ""
echo "Starting integrated dashboard with:"
echo "✨ One-click HIPAA Compliant Analysis"
echo "🤖 AI-powered chatbot for result insights"
echo "📊 Automatic data generation and analysis"
echo ""
echo "The app will open in your browser automatically..."
echo ""

# Run the integrated Streamlit app
streamlit run streamlit_app_integrated.py --server.port 8502 --browser.gatherUsageStats false

echo ""
echo "Dashboard closed. Thank you for using the Healthcare Sentiment Analysis Dashboard!"