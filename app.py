import nltk
nltk.download('vader_lexicon')
# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- NLTK Sentiment Analyzer Setup ---
# Initialize the VADER sentiment intensity analyzer
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically attuned to sentiments expressed in social media, and works well on short texts.
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

# --- App Title and Description ---
st.set_page_config(layout="wide")
st.title('Financial Sentiment Analysis Dashboard')
st.markdown("This dashboard retrieves real-time stock prices and the latest news headlines, performing sentiment analysis to gauge market mood.")
st.markdown("---")

# --- User Input for Stock Ticker in the Sidebar ---
st.sidebar.header('User Input')
ticker_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
ticker_input = ticker_input.upper() # Standardize to uppercase

# --- Data Fetching Functions (with Caching) ---

# Use Streamlit's caching to avoid re-fetching data on every interaction
@st.cache_data
def get_stock_data(ticker):
    """Fetches historical stock data for the given ticker."""
    stock = yf.Ticker(ticker)
    # Fetch data for the last year
    hist = stock.history(period="1y")
    return hist

@st.cache_data
def get_news_headlines(ticker):
    """Fetches news headlines for the given ticker."""
    stock = yf.Ticker(ticker)
    news = stock.news
    if not news:
        return pd.DataFrame() # Return empty DataFrame if no news
    # Create a DataFrame from the list of news dictionaries
    news_df = pd.DataFrame(news)
    # Ensure necessary columns exist
    if 'title' in news_df.columns and 'link' in news_df.columns and 'publisher' in news_df.columns:
        return news_df[['title', 'publisher', 'link']]
    return pd.DataFrame()

# --- Main Application Logic ---
if ticker_input:
    # --- Fetch Data ---
    stock_data = get_stock_data(ticker_input)
    news_df = get_news_headlines(ticker_input)

    if not stock_data.empty:
        # --- Display Stock Price Chart ---
        st.subheader(f'Stock Price Trend for {ticker_input}')
        fig_price = px.line(stock_data, x=stock_data.index, y='Close', title=f'{ticker_input} Closing Price Over Last Year')
        fig_price.update_layout(xaxis_title='Date', yaxis_title='Stock Price (USD)')
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.error(f"Could not retrieve stock data for {ticker_input}. Please check the ticker symbol.")

    if not news_df.empty:
        # --- Perform and Display Sentiment Analysis ---
        st.subheader(f'Sentiment Analysis of Recent News for {ticker_input}')

        # Apply sentiment analysis to each news headline
        # The 'compound' score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1 (most extreme negative) and +1 (most extreme positive).
        sentiments = news_df['title'].apply(lambda title: sia.polarity_scores(title)['compound'])
        news_df['sentiment_score'] = sentiments

        # Function to classify sentiment based on score
        def classify_sentiment(score):
            if score >= 0.05:
                return 'Positive'
            elif score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'

        news_df['sentiment'] = news_df['sentiment_score'].apply(classify_sentiment)

        # Display the news with sentiment in a table
        st.dataframe(news_df[['title', 'publisher', 'sentiment', 'sentiment_score', 'link']], use_container_width=True)

        # --- Display Sentiment Distribution ---
        sentiment_counts = news_df['sentiment'].value_counts()
        fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution of News Headlines')
        st.plotly_chart(fig_sentiment, use_container_width=True)

    else:
        st.warning(f"No recent news found for {ticker_input}.")