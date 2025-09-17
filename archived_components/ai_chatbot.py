#!/usr/bin/env python3
"""
AI Chatbot for Healthcare Sentiment Analysis
============================================

An intelligent chatbot powered by Hugging Face transformers that can answer
questions about sentiment analysis data and provide insights.

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
    st.warning("Hugging Face transformers not available. Install with: pip install transformers torch")

class HealthcareSentimentChatbot:
    """
    AI-powered chatbot for healthcare sentiment analysis questions.
    HIPAA-compliant with local-only processing.
    """
    
    def __init__(self, enable_ai=True, offline_mode=False):
        """
        Initialize the chatbot with a question-answering model.
        
        Args:
            enable_ai (bool): Whether to enable AI features
            offline_mode (bool): If True, uses only local models (no external downloads)
        """
        self.model = None
        self.tokenizer = None
        self.data_context = {}
        self.conversation_history = []
        self.enable_ai = enable_ai
        self.offline_mode = offline_mode
        
        if enable_ai and HF_AVAILABLE:
            self._load_model()
        elif enable_ai and not HF_AVAILABLE:
            st.warning("🤖 AI features requested but Hugging Face transformers not available. Install with: pip install transformers torch")
    
    def _load_model(self):
        """Load a lightweight question-answering model from Hugging Face."""
        try:
            if self.offline_mode:
                st.info("🔒 Offline mode enabled - AI features disabled for maximum HIPAA compliance")
                self.qa_pipeline = None
                return
            
            # Use a lightweight model for better performance
            model_name = "distilbert-base-cased-distilled-squad"
            
            # Initialize the QA pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            st.success("🤖 AI Chatbot loaded successfully! All inference happens locally.")
            st.info("🔒 HIPAA Compliance: Patient data never leaves your local machine")
            
        except Exception as e:
            st.error(f"Failed to load AI model: {str(e)}")
            self.qa_pipeline = None
    
    def prepare_data_context(self, analyzer_results: Dict[str, Any], data: List[Dict]) -> str:
        """
        Prepare a comprehensive context about the sentiment analysis data
        for the AI model to answer questions.
        """
        context_parts = []
        
        # Basic statistics
        total_feedback = len(data)
        sentiment_dist = Counter(entry['predicted_sentiment'] for entry in data)
        positive_pct = (sentiment_dist['positive'] / total_feedback) * 100
        
        context_parts.append(f"""
        Healthcare Sentiment Analysis Summary:
        - Total feedback analyzed: {total_feedback:,} entries
        - Positive sentiment: {positive_pct:.1f}%
        - Negative sentiment: {(sentiment_dist['negative'] / total_feedback) * 100:.1f}%
        - Neutral sentiment: {(sentiment_dist['neutral'] / total_feedback) * 100:.1f}%
        """)
        
        # Service analysis
        service_analysis = analyzer_results.get('service_analysis', {})
        if service_analysis:
            context_parts.append("\nService Performance:")
            for service, stats in service_analysis.items():
                context_parts.append(f"- {service}: {stats['avg_sentiment_score']:.3f} avg sentiment, {stats['total']} reviews")
        
        # Demographic analysis
        demographic_analysis = analyzer_results.get('demographic_analysis', {})
        if demographic_analysis:
            context_parts.append("\nDemographic Insights:")
            
            # Age groups
            age_data = demographic_analysis.get('age_groups', {})
            if age_data:
                best_age = max(age_data.items(), key=lambda x: x[1]['avg_score'])
                context_parts.append(f"- Highest sentiment age group: {best_age[0]} ({best_age[1]['avg_score']:.3f})")
            
            # Gender
            gender_data = demographic_analysis.get('genders', {})
            if gender_data:
                best_gender = max(gender_data.items(), key=lambda x: x[1]['avg_score'])
                context_parts.append(f"- Highest sentiment gender: {best_gender[0]} ({best_gender[1]['avg_score']:.3f})")
            
            # Insurance
            insurance_data = demographic_analysis.get('insurance_types', {})
            if insurance_data:
                best_insurance = max(insurance_data.items(), key=lambda x: x[1]['avg_score'])
                context_parts.append(f"- Highest sentiment insurance: {best_insurance[0]} ({best_insurance[1]['avg_score']:.3f})")
        
        # Combination analysis
        combinations = analyzer_results.get('combinations', [])
        if combinations:
            context_parts.append(f"\nService Combinations:")
            context_parts.append(f"- Found {len(combinations)} service combinations")
            
            combination_patterns = analyzer_results.get('combination_patterns', {})
            if combination_patterns:
                service_pairs = combination_patterns.get('service_pairs', {})
                if service_pairs:
                    top_pair = max(service_pairs.items(), key=lambda x: x[1]['count'])
                    context_parts.append(f"- Most common combination: {top_pair[0][0]} + {top_pair[0][1]} ({top_pair[1]['count']} occurrences)")
        
        # Sample feedback examples
        context_parts.append("\nSample Feedback Examples:")
        positive_examples = [entry for entry in data if entry['predicted_sentiment'] == 'positive'][:3]
        negative_examples = [entry for entry in data if entry['predicted_sentiment'] == 'negative'][:3]
        
        for i, example in enumerate(positive_examples, 1):
            context_parts.append(f"Positive example {i}: {example['feedback_text'][:100]}...")
        
        for i, example in enumerate(negative_examples, 1):
            context_parts.append(f"Negative example {i}: {example['feedback_text'][:100]}...")
        
        return "\n".join(context_parts)
    
    def generate_insights(self, analyzer_results: Dict[str, Any], data: List[Dict]) -> List[str]:
        """Generate AI-powered insights and recommendations."""
        insights = []
        
        # Analyze sentiment trends
        sentiment_dist = Counter(entry['predicted_sentiment'] for entry in data)
        total = len(data)
        positive_pct = (sentiment_dist['positive'] / total) * 100
        
        if positive_pct > 60:
            insights.append("🎉 Overall sentiment is very positive! Your healthcare services are well-received by patients.")
        elif positive_pct > 40:
            insights.append("📊 Sentiment is mixed but leaning positive. There's room for improvement in some areas.")
        else:
            insights.append("⚠️ Sentiment is concerning. Immediate attention needed to improve patient experience.")
        
        # Service-specific insights
        service_analysis = analyzer_results.get('service_analysis', {})
        if service_analysis:
            best_service = max(service_analysis.items(), key=lambda x: x[1]['avg_sentiment_score'])
            worst_service = min(service_analysis.items(), key=lambda x: x[1]['avg_sentiment_score'])
            
            insights.append(f"🏆 Best performing service: {best_service[0]} (sentiment: {best_service[1]['avg_sentiment_score']:.3f})")
            insights.append(f"🔧 Needs improvement: {worst_service[0]} (sentiment: {worst_service[1]['avg_sentiment_score']:.3f})")
        
        # Demographic insights
        demographic_analysis = analyzer_results.get('demographic_analysis', {})
        if demographic_analysis:
            age_data = demographic_analysis.get('age_groups', {})
            if age_data:
                best_age = max(age_data.items(), key=lambda x: x[1]['avg_score'])
                worst_age = min(age_data.items(), key=lambda x: x[1]['avg_score'])
                
                insights.append(f"👥 Age group {best_age[0]} has highest satisfaction ({best_age[1]['avg_score']:.3f})")
                insights.append(f"👥 Age group {worst_age[0]} needs attention ({worst_age[1]['avg_score']:.3f})")
        
        # Combination insights
        combinations = analyzer_results.get('combinations', [])
        if combinations:
            combination_patterns = analyzer_results.get('combination_patterns', {})
            if combination_patterns:
                avg_combo_sentiment = combination_patterns.get('avg_sentiment_score', 0)
                if avg_combo_sentiment > 0.1:
                    insights.append("🔗 Service combinations show positive sentiment - patients appreciate integrated care")
                elif avg_combo_sentiment < -0.1:
                    insights.append("🔗 Service combinations show negative sentiment - coordination issues may exist")
        
        return insights
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer a question about the sentiment analysis data using AI.
        HIPAA-compliant: All processing happens locally.
        """
        if not self.enable_ai:
            return {
                "answer": "AI features are disabled for maximum HIPAA compliance.",
                "confidence": 0.0,
                "source": "disabled"
            }
        
        if not HF_AVAILABLE or not self.qa_pipeline:
            return {
                "answer": "AI chatbot is not available. Please install transformers and torch, or enable AI features.",
                "confidence": 0.0,
                "source": "error"
            }
        
        try:
            # Use the QA pipeline to answer the question
            # HIPAA COMPLIANCE: All inference happens locally, no data transmission
            result = self.qa_pipeline(question=question, context=context)
            
            # Add some healthcare-specific follow-up suggestions
            follow_up_suggestions = self._get_follow_up_suggestions(question)
            
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "source": "ai_model_local",
                "follow_up_suggestions": follow_up_suggestions,
                "hipaa_compliant": True
            }
            
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error processing your question: {str(e)}",
                "confidence": 0.0,
                "source": "error"
            }
    
    def _get_follow_up_suggestions(self, question: str) -> List[str]:
        """Generate relevant follow-up question suggestions based on the current question."""
        suggestions = []
        
        question_lower = question.lower()
        
        if "service" in question_lower:
            suggestions.extend([
                "Which service has the highest patient satisfaction?",
                "What are the most common complaints about services?",
                "How do different services compare in terms of sentiment?"
            ])
        
        if "demographic" in question_lower or "age" in question_lower or "gender" in question_lower:
            suggestions.extend([
                "Which age group has the highest satisfaction?",
                "Are there gender differences in patient satisfaction?",
                "How does insurance type affect patient sentiment?"
            ])
        
        if "combination" in question_lower or "multiple" in question_lower:
            suggestions.extend([
                "What are the most common service combinations?",
                "Do service combinations improve or worsen patient satisfaction?",
                "Which demographic groups use service combinations most?"
            ])
        
        if "sentiment" in question_lower or "positive" in question_lower or "negative" in question_lower:
            suggestions.extend([
                "What factors contribute to positive sentiment?",
                "What are the main causes of negative sentiment?",
                "How can we improve overall patient satisfaction?"
            ])
        
        # Default suggestions if no specific keywords found
        if not suggestions:
            suggestions.extend([
                "What are the key insights from this analysis?",
                "Which services need the most improvement?",
                "What recommendations do you have for improving patient satisfaction?"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def add_to_history(self, question: str, answer: str, confidence: float):
        """Add a Q&A pair to the conversation history."""
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "confidence": confidence
        })
        
        # Keep only last 10 conversations
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history

def create_chat_interface():
    """Create the Streamlit chat interface."""
    st.markdown("### 🤖 AI Assistant")
    st.markdown("Ask me anything about your sentiment analysis data!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HealthcareSentimentChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Chat input
    user_question = st.text_input(
        "💬 Ask a question about your data:",
        placeholder="e.g., Which service has the highest patient satisfaction?",
        key="chat_input"
    )
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        # Use compatible message display for older Streamlit versions
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**AI:** {message['content']}")
            if "confidence" in message:
                st.caption(f"Confidence: {message['confidence']:.2f}")
    
    # Process question
    if user_question and st.button("Ask AI", type="primary"):
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Get data context (this would be passed from the main app)
        if 'data_context' in st.session_state:
            context = st.session_state.data_context
        else:
            context = "No data context available. Please run the analysis first."
        
        # Get AI response
        with st.spinner("🤖 AI is thinking..."):
            response = chatbot.answer_question(user_question, context)
        
        # Add AI response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "confidence": response["confidence"]
        })
        
        # Add to chatbot history
        chatbot.add_to_history(user_question, response["answer"], response["confidence"])
        
        # Show follow-up suggestions
        if response.get("follow_up_suggestions"):
            st.markdown("**💡 Suggested follow-up questions:**")
            for suggestion in response["follow_up_suggestions"]:
                if st.button(suggestion, key=f"suggestion_{len(st.session_state.chat_history)}"):
                    st.session_state.chat_input = suggestion
                    st.rerun()
        
        # Rerun to show new messages
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def create_ai_insights_panel(analyzer_results: Dict[str, Any], data: List[Dict]):
    """Create an AI-powered insights panel."""
    st.markdown("### 🧠 AI-Powered Insights")
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HealthcareSentimentChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Generate insights
    insights = chatbot.generate_insights(analyzer_results, data)
    
    for insight in insights:
        st.info(insight)
    
    # Quick question buttons
    st.markdown("#### 🚀 Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Overall Performance"):
            st.session_state.chat_input = "What is the overall performance of our healthcare services?"
            st.rerun()
    
    with col2:
        if st.button("🏆 Best Services"):
            st.session_state.chat_input = "Which services are performing the best?"
            st.rerun()
    
    with col3:
        if st.button("🔧 Areas for Improvement"):
            st.session_state.chat_input = "What areas need improvement based on patient feedback?"
            st.rerun()
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("👥 Demographic Insights"):
            st.session_state.chat_input = "What demographic insights can you share?"
            st.rerun()
    
    with col5:
        if st.button("🔗 Service Combinations"):
            st.session_state.chat_input = "Tell me about service combinations and their impact."
            st.rerun()
    
    with col6:
        if st.button("💡 Recommendations"):
            st.session_state.chat_input = "What recommendations do you have for improving patient satisfaction?"
            st.rerun()
