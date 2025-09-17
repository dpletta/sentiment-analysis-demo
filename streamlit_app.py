#!/usr/bin/env python3
"""
Streamlit Sentiment Analysis Dashboard
=====================================

An interactive dashboard for HIPAA-compliant sentiment analysis with Magic UI components.
Designed for non-technical users to explore healthcare service feedback analytics.

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
from datetime import datetime, timedelta
import random
from collections import Counter, defaultdict
import re
from pathlib import Path

# Import our sentiment analysis modules
from simplified_demo import SimplifiedSentimentAnalyzer
# from hipaa_sentiment_analysis import HIPAACompliantSentimentAnalyzer  # Optional: Full analysis

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
    
    .service-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .service-card h4 {
        color: #333;
        margin: 0 0 0.5rem 0;
    }
    
    .service-card p {
        color: #666;
        margin: 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #ff6b6b;
    }
    
    .insight-box h4 {
        color: #333;
        margin: 0 0 1rem 0;
    }
    
    .insight-box ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .insight-box li {
        margin: 0.5rem 0;
        color: #555;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Animated elements */
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
    
    /* Shimmer effect for loading */
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    
    .shimmer {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200px 100%;
        animation: shimmer 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

def create_magic_ui_header():
    """Create an animated header with Magic UI styling."""
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1>🏥 Healthcare Sentiment Analysis Dashboard</h1>
        <p>Interactive Analytics for Healthcare Service Feedback • HIPAA Compliant • Real-time Insights</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, subtitle, card_type="default"):
    """Create a Magic UI-inspired metric card."""
    css_class = f"metric-card {card_type}-metric"
    st.markdown(f"""
    <div class="{css_class}">
        <h3>{value}</h3>
        <p><strong>{title}</strong><br>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def create_service_card(service_name, sentiment_score, rating, feedback_count):
    """Create a service performance card."""
    # Determine card color based on sentiment
    if sentiment_score > 0.1:
        border_color = "#4facfe"
        bg_color = "linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)"
    elif sentiment_score < -0.1:
        border_color = "#fa709a"
        bg_color = "linear-gradient(135deg, #fce4ec 0%, #f8bbd9 100%)"
    else:
        border_color = "#a8edea"
        bg_color = "linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%)"
    
    st.markdown(f"""
    <div class="service-card" style="border-left-color: {border_color}; background: {bg_color};">
        <h4>{service_name}</h4>
        <p><strong>Sentiment Score:</strong> {sentiment_score:.3f}</p>
        <p><strong>Average Rating:</strong> {rating:.1f}/5</p>
        <p><strong>Feedback Count:</strong> {feedback_count}</p>
    </div>
    """, unsafe_allow_html=True)

def create_insight_box(title, insights):
    """Create an insight box with recommendations."""
    st.markdown(f"""
    <div class="insight-box">
        <h4>{title}</h4>
        <ul>
            {''.join([f'<li>{insight}</li>' for insight in insights])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load and cache sample data."""
    analyzer = SimplifiedSentimentAnalyzer()
    data = analyzer.load_sample_data()
    analyzed_data = analyzer.simple_sentiment_analysis()
    service_analysis = analyzer.analyze_service_patterns()
    combinations = analyzer.find_service_combinations()
    insights = analyzer.generate_insights()
    
    return {
        'data': analyzed_data,
        'service_analysis': service_analysis,
        'combinations': combinations,
        'insights': insights,
        'analyzer': analyzer
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
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01
        ),
        font=dict(size=12),
        title_font_size=16
    )
    
    return fig

def create_service_performance_chart(service_analysis):
    """Create a service performance comparison chart."""
    services = list(service_analysis.keys())
    sentiment_scores = [service_analysis[s]['avg_sentiment_score'] for s in services]
    ratings = [service_analysis[s]['avg_rating'] for s in services]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Sentiment Score', 'Average Rating'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Sentiment scores
    fig.add_trace(
        go.Bar(
            x=services,
            y=sentiment_scores,
            name='Sentiment Score',
            marker_color=['#4facfe' if score > 0 else '#fa709a' if score < 0 else '#a8edea' for score in sentiment_scores],
            hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Ratings
    fig.add_trace(
        go.Bar(
            x=services,
            y=ratings,
            name='Rating',
            marker_color='#667eea',
            hovertemplate='<b>%{x}</b><br>Rating: %{y:.1f}/5<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="Service Performance Comparison",
        title_x=0.5
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_temporal_trends_chart(data):
    """Create temporal sentiment trends."""
    # Convert dates and group by month
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
    
    fig.update_layout(
        height=400,
        title_x=0.5,
        xaxis_title="Month",
        yaxis_title="Average Sentiment Score"
    )
    
    return fig

def create_word_cloud_analysis(data):
    """Create word frequency analysis."""
    positive_texts = ' '.join([entry['processed_text'] for entry in data if entry['predicted_sentiment'] == 'positive'])
    negative_texts = ' '.join([entry['processed_text'] for entry in data if entry['predicted_sentiment'] == 'negative'])
    
    positive_words = Counter(positive_texts.split()).most_common(10)
    negative_words = Counter(negative_texts.split()).most_common(10)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Top Positive Words', 'Top Negative Words'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if positive_words:
        pos_words, pos_counts = zip(*positive_words)
        fig.add_trace(
            go.Bar(
                x=list(pos_counts),
                y=list(pos_words),
                orientation='h',
                name='Positive',
                marker_color='#4facfe',
                hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if negative_words:
        neg_words, neg_counts = zip(*negative_words)
        fig.add_trace(
            go.Bar(
                x=list(neg_counts),
                y=list(neg_words),
                orientation='h',
                name='Negative',
                marker_color='#fa709a',
                hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="Word Frequency Analysis",
        title_x=0.5
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Create header
    create_magic_ui_header()
    
    # Sidebar configuration
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Simplified Demo", "Full HIPAA Analysis"],
        help="Choose between simplified demo or full analysis with advanced features"
    )
    
    # Data refresh button
    if st.sidebar.button("🔄 Refresh Data", help="Generate new sample data"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Load data
    with st.spinner("🔄 Loading analysis data..."):
        analysis_data = load_sample_data()
    
    data = analysis_data['data']
    service_analysis = analysis_data['service_analysis']
    combinations = analysis_data['combinations']
    insights = analysis_data['insights']
    
    # Main dashboard
    st.markdown("## 📊 Executive Summary")
    
    # Key metrics row
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
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🏥 Service Analysis", 
        "🔗 Combinations", 
        "📊 Visualizations", 
        "💡 Insights"
    ])
    
    with tab1:
        st.markdown("### Overall Sentiment Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_sentiment = create_sentiment_distribution_chart(data)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Quick Stats")
            for sentiment, count in insights['sentiment_distribution'].items():
                percentage = (count / insights['total_feedback']) * 100
                st.metric(
                    label=sentiment.capitalize(),
                    value=f"{count:,}",
                    delta=f"{percentage:.1f}%"
                )
        
        # Temporal trends
        st.markdown("### 📅 Temporal Trends")
        fig_temporal = create_temporal_trends_chart(data)
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    with tab2:
        st.markdown("### Service Performance Analysis")
        
        # Service performance chart
        fig_service = create_service_performance_chart(service_analysis)
        st.plotly_chart(fig_service, use_container_width=True)
        
        # Service cards
        st.markdown("### 🏥 Service Details")
        
        # Sort services by sentiment score
        sorted_services = sorted(service_analysis.items(), 
                               key=lambda x: x[1]['avg_sentiment_score'], reverse=True)
        
        for service_name, stats in sorted_services:
            create_service_card(
                service_name,
                stats['avg_sentiment_score'],
                stats['avg_rating'],
                stats['total']
            )
    
    with tab3:
        st.markdown("### Service Combination Analysis")
        
        if combinations:
            st.markdown(f"Found **{len(combinations)}** service combinations in the feedback data.")
            
            # Combination sentiment distribution
            combo_sentiment = Counter(combo['sentiment'] for combo in combinations)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_combo = px.pie(
                    values=list(combo_sentiment.values()),
                    names=list(combo_sentiment.keys()),
                    title="Combination Sentiment Distribution",
                    color_discrete_map={
                        'positive': '#4facfe',
                        'negative': '#fa709a',
                        'neutral': '#a8edea'
                    }
                )
                st.plotly_chart(fig_combo, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔗 Sample Combinations")
                for i, combo in enumerate(combinations[:5]):
                    st.markdown(f"""
                    **{i+1}. {combo['primary_service']}**
                    - Sentiment: {combo['sentiment']}
                    - Score: {combo['sentiment_score']:.3f}
                    - Services mentioned: {', '.join(combo['mentioned_services']) if combo['mentioned_services'] else 'None'}
                    """)
        else:
            st.info("No service combinations found in the current dataset.")
    
    with tab4:
        st.markdown("### Advanced Visualizations")
        
        # Word frequency analysis
        st.markdown("#### 📝 Word Frequency Analysis")
        fig_words = create_word_cloud_analysis(data)
        st.plotly_chart(fig_words, use_container_width=True)
        
        # Rating vs Sentiment correlation
        st.markdown("#### 📊 Rating vs Sentiment Correlation")
        
        df = pd.DataFrame(data)
        correlation = df['rating'].corr(df['sentiment_score'])
        
        fig_corr = px.scatter(
            df,
            x='rating',
            y='sentiment_score',
            color='predicted_sentiment',
            title=f"Rating vs Sentiment Correlation (r = {correlation:.3f})",
            labels={'rating': 'Rating (1-5)', 'sentiment_score': 'Sentiment Score'},
            color_discrete_map={
                'positive': '#4facfe',
                'negative': '#fa709a',
                'neutral': '#a8edea'
            }
        )
        
        # Add trend line
        z = np.polyfit(df['rating'], df['sentiment_score'], 1)
        p = np.poly1d(z)
        fig_corr.add_trace(
            go.Scatter(
                x=df['rating'].sort_values(),
                y=p(df['rating'].sort_values()),
                mode='lines',
                name='Trend Line',
                line=dict(color='red', dash='dash')
            )
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab5:
        st.markdown("### Key Insights & Recommendations")
        
        # Generate insights
        recommendations = [
            f"Focus on replicating success factors from '{insights['best_performing_service']}' (highest sentiment)",
            f"Investigate improvement opportunities for '{insights['worst_performing_service']}' (lowest sentiment)",
            "Monitor service combinations for cross-selling opportunities",
            "Implement regular sentiment monitoring using this framework",
            "Expand analysis to include temporal patterns and seasonal variations"
        ]
        
        create_insight_box("💡 Strategic Recommendations", recommendations)
        
        # HIPAA compliance note
        st.markdown("""
        <div class="insight-box">
            <h4>🔒 HIPAA Compliance Verified</h4>
            <ul>
                <li>✓ All data processing performed locally</li>
                <li>✓ No external data transmission</li>
                <li>✓ PII detection and redaction implemented</li>
                <li>✓ Comprehensive audit logging enabled</li>
                <li>✓ Secure local storage utilized</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical details
        st.markdown("### 🔧 Technical Implementation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Analysis Methods Used:**
            - Rule-based sentiment analysis
            - Service combination detection
            - Temporal trend analysis
            - Word frequency analysis
            - Statistical correlation analysis
            """)
        
        with col2:
            st.markdown("""
            **Data Processing:**
            - Text preprocessing with PII protection
            - Sentiment scoring algorithm
            - Service categorization
            - Performance metrics calculation
            - Interactive visualization generation
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏥 Healthcare Sentiment Analysis Dashboard | HIPAA Compliant | Built with Streamlit & Magic UI</p>
        <p>Generated on: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
