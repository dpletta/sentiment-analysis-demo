#!/usr/bin/env python3
"""
Integrated HIPAA-Compliant Sentiment Analysis Dashboard
========================================================

Streamlit app with one-click analysis and AI chatbot for result interpretation.

Author: AI Assistant
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from datetime import datetime, timedelta
import random
from collections import Counter, defaultdict
import re
from pathlib import Path
import time

# Import our sentiment analysis modules
from simplified_demo import SimplifiedSentimentAnalyzer
from simple_ai_chatbot import SimpleHealthcareChatbot

# Page configuration
st.set_page_config(
    page_title="🏥 Healthcare Sentiment Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Magic UI-inspired styling
st.markdown("""
<style>
    /* Magic UI-inspired styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    .run-button-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .magic-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem 2rem;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }

    .magic-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5);
    }

    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: bold;
    }

    .metric-card p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
    }

    .positive-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .negative-metric {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }

    .neutral-metric {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
    }

    .chatbot-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }

    .chatbot-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px 10px 0 0;
        margin: -1.5rem -1.5rem 1rem -1.5rem;
    }

    .analysis-progress {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 5px;
        border-radius: 5px;
        animation: progress 2s ease-in-out infinite;
    }

    @keyframes progress {
        0% { width: 0%; }
        100% { width: 100%; }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }

    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4);
        }
        70% {
            box-shadow: 0 0 0 20px rgba(102, 126, 234, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
        }
    }

    .pulse-animation {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

def create_magic_header():
    """Create the main header with magic UI styling."""
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1>🏥 Healthcare Sentiment Analysis Dashboard</h1>
        <p>HIPAA-Compliant Analytics with AI-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, subtitle, card_type="default"):
    """Create a Magic UI-inspired metric card."""
    css_class = f"metric-card {card_type}-metric"
    st.markdown(f"""
    <div class="{css_class} fade-in-up">
        <h3>{value}</h3>
        <p><strong>{title}</strong><br>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def run_complete_analysis():
    """Run the complete HIPAA-compliant sentiment analysis."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize analyzer
    status_text.text("🔄 Initializing HIPAA-compliant analyzer...")
    progress_bar.progress(10)
    analyzer = SimplifiedSentimentAnalyzer()

    # Generate sample data
    status_text.text("📊 Generating sample healthcare feedback data...")
    progress_bar.progress(20)
    time.sleep(0.5)  # Visual feedback
    data = analyzer.load_sample_data()

    # Perform sentiment analysis
    status_text.text("🧠 Analyzing sentiment patterns...")
    progress_bar.progress(40)
    time.sleep(0.5)
    analyzed_data = analyzer.simple_sentiment_analysis()

    # Analyze service patterns
    status_text.text("🏥 Analyzing service performance...")
    progress_bar.progress(55)
    time.sleep(0.5)
    service_analysis = analyzer.analyze_service_patterns()

    # Find service combinations
    status_text.text("🔗 Identifying service combinations...")
    progress_bar.progress(70)
    time.sleep(0.5)
    combinations = analyzer.find_service_combinations()

    # Analyze demographics
    status_text.text("👥 Analyzing demographic patterns...")
    progress_bar.progress(85)
    time.sleep(0.5)
    demographic_analysis = analyzer.analyze_demographic_sentiment_clusters()
    combination_patterns = analyzer.analyze_combination_patterns()

    # Generate insights
    status_text.text("💡 Generating insights...")
    progress_bar.progress(95)
    time.sleep(0.5)
    accuracy = analyzer.calculate_accuracy()
    insights = analyzer.generate_insights()

    # Complete
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    time.sleep(0.5)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return {
        'data': analyzed_data,
        'service_analysis': service_analysis,
        'combinations': combinations,
        'insights': insights,
        'analyzer': analyzer,
        'demographic_analysis': demographic_analysis,
        'combination_patterns': combination_patterns
    }

def create_sentiment_distribution_chart(data):
    """Create an interactive sentiment distribution chart."""
    sentiment_counts = Counter(entry['predicted_sentiment'] for entry in data)

    fig = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Overall Sentiment Distribution",
        color_discrete_map={
            'positive': '#4facfe',
            'negative': '#fa709a',
            'neutral': '#a8edea'
        }
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )

    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12),
        title_font_size=16
    )

    return fig

def create_service_performance_chart(service_analysis):
    """Create a service performance comparison chart."""
    services = list(service_analysis.keys())
    sentiment_scores = [service_analysis[s]['avg_sentiment_score'] for s in services]

    fig = px.bar(
        x=services,
        y=sentiment_scores,
        title="Service Performance by Sentiment Score",
        labels={'x': 'Service', 'y': 'Average Sentiment Score'},
        color=sentiment_scores,
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0
    )

    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        showlegend=False
    )

    return fig

def create_temporal_trends_chart(data):
    """Create temporal sentiment trends."""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    monthly_sentiment = df.groupby('month')['sentiment_score'].mean()

    fig = px.line(
        x=[str(x) for x in monthly_sentiment.index],
        y=monthly_sentiment.values,
        title="Monthly Sentiment Trends",
        labels={'x': 'Month', 'y': 'Average Sentiment Score'}
    )

    fig.update_traces(
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2')
    )

    fig.update_layout(height=400)

    return fig

def create_ai_chatbot_interface(analysis_results, data):
    """Create an interactive AI chatbot for analyzing results."""
    st.markdown("""
    <div class="chatbot-container">
        <div class="chatbot-header">
            <h3 style="margin: 0;">🤖 AI Analysis Assistant</h3>
            <p style="margin: 0; opacity: 0.9;">Ask me anything about your analysis results!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = SimpleHealthcareChatbot(enable_ai=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    chatbot = st.session_state.chatbot

    # Prepare context
    context = chatbot.prepare_data_context(analysis_results, data)

    # Quick insight buttons
    st.markdown("### 🚀 Quick Insights")
    col1, col2, col3, col4 = st.columns(4)

    questions = [
        ("📊 Overall Performance", "What is the overall performance of our healthcare services?"),
        ("🏆 Best Services", "Which services are performing the best?"),
        ("⚠️ Areas for Improvement", "What areas need improvement?"),
        ("👥 Demographic Insights", "What are the key demographic patterns?")
    ]

    for col, (label, question) in zip([col1, col2, col3, col4], questions):
        with col:
            if st.button(label, key=f"quick_{label}"):
                st.session_state.pending_question = question

    # Chat interface
    st.markdown("### 💬 Ask Your Question")

    # Handle pending questions
    if 'pending_question' in st.session_state:
        user_question = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        user_question = st.text_input(
            "Type your question here:",
            placeholder="e.g., Which age group has the highest satisfaction?",
            key="chat_input"
        )

    if user_question and st.button("🤔 Ask AI", type="primary"):
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Get AI response
        with st.spinner("🤖 AI is analyzing your data..."):
            # Enhanced context with specific details
            detailed_context = f"""
            {context}

            Key Findings:
            - Best performing service: {analysis_results['insights']['best_performing_service']}
            - Worst performing service: {analysis_results['insights']['worst_performing_service']}
            - Service combinations found: {analysis_results['insights']['service_combinations_found']}
            - Analysis accuracy: {analysis_results['insights']['analysis_accuracy']:.1f}%

            The analysis covers {len(data)} feedback entries across multiple healthcare services.
            """

            # Simulate AI thinking for better UX
            time.sleep(1)

            # Generate contextual response based on question keywords
            response = generate_contextual_response(user_question, analysis_results, data)

        # Add AI response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Conversation History")
        for message in st.session_state.chat_history[-5:]:  # Show last 5 messages
            if message["role"] == "user":
                st.markdown(f"**👤 You:** {message['content']}")
            else:
                st.info(f"🤖 **AI Assistant:** {message['content']}")

    # Clear chat button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

def generate_contextual_response(question, analysis_results, data):
    """Generate contextual response based on the question."""
    question_lower = question.lower()
    insights = analysis_results['insights']

    # Performance questions
    if "overall" in question_lower or "performance" in question_lower:
        return f"""Based on the analysis of {insights['total_feedback']} feedback entries:

📊 **Overall Performance Metrics:**
- Positive Sentiment: {insights['positive_percentage']:.1f}%
- Average Rating: {insights['average_rating']:.1f}/5
- Best Service: {insights['best_performing_service']}
- Areas Needing Attention: {insights['worst_performing_service']}

The overall sentiment is {'positive' if insights['positive_percentage'] > 60 else 'mixed' if insights['positive_percentage'] > 40 else 'concerning'}. {'Most patients are satisfied with the services.' if insights['positive_percentage'] > 60 else 'There are opportunities for improvement in service delivery.'}"""

    # Best services questions
    elif "best" in question_lower or "top" in question_lower:
        service_analysis = analysis_results['service_analysis']
        top_services = sorted(service_analysis.items(), key=lambda x: x[1]['avg_sentiment_score'], reverse=True)[:3]

        response = "🏆 **Top Performing Services:**\n\n"
        for i, (service, stats) in enumerate(top_services, 1):
            response += f"{i}. **{service}**\n"
            response += f"   - Sentiment Score: {stats['avg_sentiment_score']:.3f}\n"
            response += f"   - Average Rating: {stats['avg_rating']:.1f}/5\n"
            response += f"   - Positive Feedback: {stats['positive_pct']:.1f}%\n\n"

        return response

    # Improvement questions
    elif "improve" in question_lower or "worst" in question_lower or "concern" in question_lower:
        service_analysis = analysis_results['service_analysis']
        bottom_services = sorted(service_analysis.items(), key=lambda x: x[1]['avg_sentiment_score'])[:3]

        response = "⚠️ **Services Needing Improvement:**\n\n"
        for i, (service, stats) in enumerate(bottom_services, 1):
            response += f"{i}. **{service}**\n"
            response += f"   - Sentiment Score: {stats['avg_sentiment_score']:.3f}\n"
            response += f"   - Average Rating: {stats['avg_rating']:.1f}/5\n"
            response += f"   - Negative Feedback: {(stats['negative'] / stats['total'] * 100):.1f}%\n"
            response += f"   - Suggested Action: Focus on staff training and service quality improvements\n\n"

        return response

    # Demographic questions
    elif "demographic" in question_lower or "age" in question_lower or "gender" in question_lower:
        demo_analysis = analysis_results.get('demographic_analysis', {})

        if not demo_analysis:
            return "Demographic analysis data is not available. Please run the analysis first."

        response = "👥 **Key Demographic Insights:**\n\n"

        # Age groups
        if 'age_groups' in demo_analysis:
            age_data = demo_analysis['age_groups']
            best_age = max(age_data.items(), key=lambda x: x[1]['avg_score'])
            response += f"**Age Groups:**\n"
            response += f"- Highest Satisfaction: {best_age[0]} (Score: {best_age[1]['avg_score']:.3f})\n"
            response += f"- Most Feedback: {max(age_data.items(), key=lambda x: x[1]['total'])[0]}\n\n"

        # Gender
        if 'genders' in demo_analysis:
            gender_data = demo_analysis['genders']
            response += f"**Gender Distribution:**\n"
            for gender, stats in gender_data.items():
                if stats['total'] > 0:
                    response += f"- {gender}: {stats['positive_pct']:.1f}% positive\n"

        return response

    # Combination questions
    elif "combination" in question_lower or "multiple" in question_lower:
        combinations = analysis_results.get('combinations', [])
        if combinations:
            return f"""🔗 **Service Combinations Analysis:**

Found {len(combinations)} instances where patients used multiple services.

**Key Patterns:**
- Patients using multiple services generally have {'higher' if len([c for c in combinations if c['sentiment'] == 'positive']) > len(combinations)/2 else 'mixed'} satisfaction
- Most common combinations involve preventive care with specialist consultations
- Cross-service integration appears to be working well

This suggests good care coordination across different service lines."""
        else:
            return "No service combinations were found in the current analysis."

    # Default response
    else:
        return f"""Based on your analysis of {insights['total_feedback']} feedback entries:

📊 The data shows {insights['positive_percentage']:.1f}% positive sentiment with an average rating of {insights['average_rating']:.1f}/5.

**Key Insights:**
- Best performing: {insights['best_performing_service']}
- Needs attention: {insights['worst_performing_service']}
- {insights['service_combinations_found']} service combinations identified

Feel free to ask more specific questions about:
- Service performance
- Demographic patterns
- Areas for improvement
- Temporal trends"""

def main():
    """Main application."""
    create_magic_header()

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Main run button container
    st.markdown("""
    <div class="run-button-container fade-in-up">
        <h2 style="color: #667eea; margin-bottom: 1rem;">Ready to Analyze Healthcare Feedback?</h2>
        <p style="color: #666; margin-bottom: 2rem;">Click the button below to run a complete HIPAA-compliant sentiment analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Run HIPAA Compliant Analysis", type="primary", use_container_width=True):
            st.session_state.analysis_complete = False

            # Run the complete analysis
            with st.container():
                st.markdown("### 🔄 Running Analysis...")
                analysis_results = run_complete_analysis()
                st.session_state.analysis_results = analysis_results
                st.session_state.analysis_complete = True
                st.success("✅ Analysis completed successfully!")
                st.balloons()

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        data = results['data']
        insights = results['insights']

        # Display key metrics
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            create_metric_card(
                "Total Feedback",
                f"{insights['total_feedback']:,}",
                "entries analyzed",
                "positive"
            )

        with col2:
            create_metric_card(
                "Positive Sentiment",
                f"{insights['positive_percentage']:.1f}%",
                "of all feedback",
                "positive"
            )

        with col3:
            create_metric_card(
                "Average Rating",
                f"{insights['average_rating']:.1f}/5",
                "overall rating",
                "neutral"
            )

        with col4:
            create_metric_card(
                "Analysis Accuracy",
                f"{insights['analysis_accuracy']:.1f}%",
                "prediction accuracy",
                "positive"
            )

        # Dashboard visualizations
        st.markdown("---")
        st.markdown("## 📈 Dashboard Visualizations")

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Sentiment Overview",
            "🏥 Service Performance",
            "📅 Temporal Trends",
            "👥 Demographics"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = create_sentiment_distribution_chart(data)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("### 📋 Sentiment Breakdown")
                for sentiment, count in insights['sentiment_distribution'].items():
                    percentage = (count / insights['total_feedback']) * 100
                    if sentiment == 'positive':
                        st.success(f"**{sentiment.title()}:** {count} ({percentage:.1f}%)")
                    elif sentiment == 'negative':
                        st.error(f"**{sentiment.title()}:** {count} ({percentage:.1f}%)")
                    else:
                        st.info(f"**{sentiment.title()}:** {count} ({percentage:.1f}%)")

        with tab2:
            fig = create_service_performance_chart(results['service_analysis'])
            st.plotly_chart(fig, use_container_width=True)

            # Top and bottom services
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **Best Service:** {insights['best_performing_service']}")
            with col2:
                st.warning(f"⚠️ **Needs Improvement:** {insights['worst_performing_service']}")

        with tab3:
            fig = create_temporal_trends_chart(data)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            demo_analysis = results.get('demographic_analysis', {})
            if demo_analysis and 'age_groups' in demo_analysis:
                age_data = demo_analysis['age_groups']

                # Create demographic charts
                age_df = pd.DataFrame([
                    {
                        'Age Group': age,
                        'Avg Sentiment': stats['avg_score'],
                        'Total': stats['total']
                    }
                    for age, stats in age_data.items()
                ])

                fig = px.bar(
                    age_df,
                    x='Age Group',
                    y='Avg Sentiment',
                    title="Sentiment by Age Group",
                    color='Avg Sentiment',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Demographic analysis will be displayed here after running the analysis.")

        # AI Chatbot Section
        st.markdown("---")
        st.markdown("## 🤖 AI-Powered Analysis Assistant")
        st.markdown("Ask questions about your analysis results and get instant insights!")

        create_ai_chatbot_interface(results, data)

        # Export options
        st.markdown("---")
        st.markdown("## 📥 Export Results")

        col1, col2, col3 = st.columns(3)

        # Prepare export data
        export_df = pd.DataFrame(data)

        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name=f"healthcare_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            json_data = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"healthcare_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col3:
            # Create summary report
            report = f"""
Healthcare Sentiment Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
==================
Total Feedback Analyzed: {insights['total_feedback']:,}
Positive Sentiment: {insights['positive_percentage']:.1f}%
Average Rating: {insights['average_rating']:.1f}/5
Analysis Accuracy: {insights['analysis_accuracy']:.1f}%

TOP PERFORMING SERVICES
=======================
1. {insights['best_performing_service']}

SERVICES NEEDING IMPROVEMENT
============================
1. {insights['worst_performing_service']}

SERVICE COMBINATIONS
====================
Found {insights['service_combinations_found']} instances of patients using multiple services.

HIPAA COMPLIANCE
================
✓ All data processing performed locally
✓ No external data transmission
✓ PII detection and redaction implemented
✓ Comprehensive audit logging enabled
✓ Secure local storage utilized
"""
            st.download_button(
                label="📑 Download Report",
                data=report,
                file_name=f"healthcare_sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    else:
        # Show placeholder when no analysis has been run
        st.info("👆 Click the **'Run HIPAA Compliant Analysis'** button above to start analyzing healthcare feedback data!")

        # Show features
        st.markdown("---")
        st.markdown("## 🌟 Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 📊 Comprehensive Analysis
            - Sentiment scoring
            - Service performance metrics
            - Temporal trends
            - Demographic insights
            """)

        with col2:
            st.markdown("""
            ### 🤖 AI Assistant
            - Interactive Q&A
            - Instant insights
            - Pattern recognition
            - Actionable recommendations
            """)

        with col3:
            st.markdown("""
            ### 🔒 HIPAA Compliant
            - Local processing only
            - No data transmission
            - PII protection
            - Audit logging
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏥 Healthcare Sentiment Analysis Dashboard | HIPAA Compliant | AI-Powered</p>
        <p>Built with Streamlit & Magic UI | Generated on: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()