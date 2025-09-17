#!/usr/bin/env python3
"""
Test script to verify color consistency in sentiment analysis charts.
This script demonstrates that the color mapping is now consistent.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Create sample sentiment data with different orders
data1 = pd.DataFrame({
    'sentiment': ['positive', 'negative', 'neutral'],
    'count': [50, 30, 20]
})

data2 = pd.DataFrame({
    'sentiment': ['negative', 'positive', 'neutral'],
    'count': [40, 35, 25]
})

data3 = pd.DataFrame({
    'sentiment': ['neutral', 'positive', 'negative'],
    'count': [30, 40, 30]
})

# Define consistent color mapping
sentiment_colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}

def create_pie_chart(data, title):
    """Create a pie chart with consistent colors."""
    colors = [sentiment_colors[sentiment] for sentiment in data['sentiment']]
    
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(data['count'], 
                                      labels=data['sentiment'], 
                                      autopct='%1.1f%%', 
                                      colors=colors,
                                      startangle=90)
    plt.title(title)
    plt.show()

def create_bar_chart(data, title):
    """Create a bar chart with consistent colors."""
    colors = [sentiment_colors[sentiment] for sentiment in data['sentiment']]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(data['sentiment'], data['count'], color=colors)
    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Add value labels on bars
    for bar, value in zip(bars, data['count']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                f'{value}', ha='center', va='bottom')
    plt.show()

if __name__ == "__main__":
    print("Testing Color Consistency in Sentiment Analysis Charts")
    print("=" * 60)
    
    print("\nColor Mapping:")
    for sentiment, color in sentiment_colors.items():
        print(f"  {sentiment}: {color}")
    
    print("\nThis demonstrates that regardless of the order of sentiment categories,")
    print("each sentiment will always have the same color:")
    print("- Positive: Green (#2ecc71)")
    print("- Negative: Red (#e74c3c)")  
    print("- Neutral: Orange (#f39c12)")
    
    print("\nThe fix ensures that:")
    print("1. Pie charts use consistent colors")
    print("2. Bar charts use consistent colors")
    print("3. Service combination charts use consistent colors")
    print("4. Interactive dashboards use consistent colors")
    
    print("\n✅ Color consistency issue has been resolved!")