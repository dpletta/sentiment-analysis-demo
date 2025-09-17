#!/usr/bin/env python3
"""
Simplified AI Chatbot for Healthcare Sentiment Analysis
=======================================================

A more robust version of the AI chatbot with better error handling.

Author: AI Assistant
Date: September 2025
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import re
from datetime import datetime

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class SimpleHealthcareChatbot:
    """
    Simplified AI-powered chatbot for healthcare sentiment analysis questions.
    """
    
    def __init__(self, enable_ai=True):
        """Initialize the chatbot."""
        self.enable_ai = enable_ai
        self.qa_pipeline = None
        self.data_context = {}
        
        if enable_ai and HF_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the AI model with error handling."""
        try:
            model_name = "distilbert-base-cased-distilled-squad"
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=-1  # Force CPU for compatibility
            )
            return True
        except Exception as e:
            st.error(f"Failed to load AI model: {str(e)}")
            self.qa_pipeline = None
            return False
    
    def prepare_data_context(self, analyzer_results: Dict[str, Any], data: List[Dict]) -> str:
        """Prepare context for AI questions."""
        try:
            total_feedback = len(data)
            sentiment_dist = Counter(entry['predicted_sentiment'] for entry in data)
            positive_pct = (sentiment_dist['positive'] / total_feedback) * 100
            
            context = f"""
            Healthcare Sentiment Analysis Summary:
            - Total feedback analyzed: {total_feedback:,} entries
            - Positive sentiment: {positive_pct:.1f}%
            - Negative sentiment: {(sentiment_dist['negative'] / total_feedback) * 100:.1f}%
            - Neutral sentiment: {(sentiment_dist['neutral'] / total_feedback) * 100:.1f}%
            """
            
            # Add service analysis
            service_analysis = analyzer_results.get('service_analysis', {})
            if service_analysis:
                context += "\nService Performance:\n"
                for service, stats in service_analysis.items():
                    context += f"- {service}: {stats['avg_sentiment_score']:.3f} avg sentiment, {stats['total']} reviews\n"
            
            return context
        except Exception as e:
            return f"Error preparing context: {str(e)}"
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer a question about the data."""
        if not self.enable_ai or not self.qa_pipeline:
            return {
                "answer": "AI features are disabled or not available.",
                "confidence": 0.0,
                "source": "disabled"
            }
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "source": "ai_model"
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "source": "error"
            }

def create_simple_chat_interface():
    """Create a simplified chat interface."""
    st.markdown("### 🤖 AI Assistant")
    
    # Initialize chatbot
    if 'simple_chatbot' not in st.session_state:
        st.session_state.simple_chatbot = SimpleHealthcareChatbot()
    
    chatbot = st.session_state.simple_chatbot
    
    # Chat input
    user_question = st.text_input(
        "💬 Ask a question about your data:",
        placeholder="e.g., Which service has the highest patient satisfaction?",
        key="simple_chat_input"
    )
    
    # Chat history
    if 'simple_chat_history' not in st.session_state:
        st.session_state.simple_chat_history = []
    
    # Display chat history
    for i, message in enumerate(st.session_state.simple_chat_history):
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**AI:** {message['content']}")
            if "confidence" in message:
                st.caption(f"Confidence: {message['confidence']:.2f}")
    
    # Process question
    if user_question and st.button("Ask AI", type="primary"):
        # Add user message
        st.session_state.simple_chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Get context
        if 'data_context' in st.session_state:
            context = st.session_state.data_context
        else:
            context = "No data context available."
        
        # Get AI response
        with st.spinner("🤖 AI is thinking..."):
            response = chatbot.answer_question(user_question, context)
        
        # Add AI response
        st.session_state.simple_chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "confidence": response["confidence"]
        })
        
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.simple_chat_history = []
        st.rerun()

def create_simple_ai_insights(analyzer_results: Dict[str, Any], data: List[Dict]):
    """Create simple AI insights."""
    st.markdown("### 🧠 AI-Powered Insights")
    
    try:
        total_feedback = len(data)
        sentiment_dist = Counter(entry['predicted_sentiment'] for entry in data)
        positive_pct = (sentiment_dist['positive'] / total_feedback) * 100
        
        if positive_pct > 60:
            st.info("🎉 Overall sentiment is very positive! Your healthcare services are well-received by patients.")
        elif positive_pct > 40:
            st.info("📊 Sentiment is mixed but leaning positive. There's room for improvement in some areas.")
        else:
            st.warning("⚠️ Sentiment is concerning. Immediate attention needed to improve patient experience.")
        
        # Service insights
        service_analysis = analyzer_results.get('service_analysis', {})
        if service_analysis:
            best_service = max(service_analysis.items(), key=lambda x: x[1]['avg_sentiment_score'])
            worst_service = min(service_analysis.items(), key=lambda x: x[1]['avg_sentiment_score'])
            
            st.success(f"🏆 Best performing service: {best_service[0]} (sentiment: {best_service[1]['avg_sentiment_score']:.3f})")
            st.warning(f"🔧 Needs improvement: {worst_service[0]} (sentiment: {worst_service[1]['avg_sentiment_score']:.3f})")
        
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
    
    # Quick question buttons
    st.markdown("#### 🚀 Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Overall Performance"):
            st.session_state.simple_chat_input = "What is the overall performance of our healthcare services?"
            st.rerun()
    
    with col2:
        if st.button("🏆 Best Services"):
            st.session_state.simple_chat_input = "Which services are performing the best?"
            st.rerun()
    
    with col3:
        if st.button("🔧 Areas for Improvement"):
            st.session_state.simple_chat_input = "What areas need improvement based on patient feedback?"
            st.rerun()
