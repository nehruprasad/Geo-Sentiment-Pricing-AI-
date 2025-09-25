import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')

# ----------------- Helper Functions -----------------

def analyze_sentiment(reviews):
    """
    Analyze sentiment for each review.
    Returns polarity (positive/negative) and subjectivity.
    """
    sentiments = []
    for r in reviews:
        blob = TextBlob(r)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        sentiments.append({
            "review": r,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": "Positive" if polarity > 0 else ("Negative" if polarity < 0 else "Neutral")
        })
    return pd.DataFrame(sentiments)

def aggregate_sentiment(sentiment_df):
    """
    Aggregate sentiment by polarity to generate regional sentiment score.
    """
    avg_polarity = sentiment_df["polarity"].mean()
    avg_subjectivity = sentiment_df["subjectivity"].mean()
    if avg_polarity > 0.2:
        regional_sentiment = "Positive"
    elif avg_polarity < -0.2:
        regional_sentiment = "Negative"
    else:
        regional_sentiment = "Neutral"
    return {
        "average_polarity": round(avg_polarity, 2),
        "average_subjectivity": round(avg_subjectivity, 2),
        "regional_sentiment": regional_sentiment
    }

def pricing_recommendation(sentiment_score):
    """
    Suggest pricing adjustments based on sentiment.
    """
    base_price = 100  # Example base price
    if sentiment_score["regional_sentiment"] == "Positive":
        recommended_price = base_price * 1.1
    elif sentiment_score["regional_sentiment"] == "Negative":
        recommended_price = base_price * 0.9
    else:
        recommended_price = base_price
    return round(recommended_price, 2)

def demand_forecast(sentiment_score):
    """
    Predict demand trend based on sentiment.
    """
    if sentiment_score["regional_sentiment"] == "Positive":
        return "High demand expected 📈"
    elif sentiment_score["regional_sentiment"] == "Negative":
        return "Low demand expected 📉"
    else:
        return "Stable demand expected ⚖️"

def market_insights(sentiment_df):
    """
    Provide insights on common feedback themes.
    """
    positive_reviews = sentiment_df[sentiment_df["polarity"] > 0]["review"].tolist()
    negative_reviews = sentiment_df[sentiment_df["polarity"] < 0]["review"].tolist()
    return {
        "positive_feedback_count": len(positive_reviews),
        "negative_feedback_count": len(negative_reviews),
        "common_positive_review": positive_reviews[0] if positive_reviews else None,
        "common_negative_review": negative_reviews[0] if negative_reviews else None
    }

# ----------------- Streamlit App -----------------

st.set_page_config(page_title="Geo-Sentiment Pricing AI", layout="wide")
st.title("📍 Geo-Sentiment Pricing AI (NLP Powered)")

st.markdown("Enter a location and its reviews to get **regional sentiment**, **pricing recommendations**, **market insights**, and **demand forecasts**.")

# Input
location = st.text_input("Enter Location (City/Region):")
reviews_text = st.text_area("Enter customer reviews (one per line):", height=200)

if st.button("Analyze Reviews"):
    if not location.strip() or not reviews_text.strip():
        st.warning("Please provide both location and reviews.")
    else:
        st.success(f"Analyzing reviews for {location}...")

        # Split reviews
        reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]
        
        # Sentiment analysis
        sentiment_df = analyze_sentiment(reviews)
        st.subheader("💬 Individual Review Sentiment")
        st.table(sentiment_df)

        # Regional sentiment
        sentiment_score = aggregate_sentiment(sentiment_df)
        st.subheader("🌐 Regional Sentiment")
        st.json(sentiment_score)

        # Pricing recommendation
        price = pricing_recommendation(sentiment_score)
        st.subheader("💰 Pricing Recommendation")
        st.write(f"Recommended Price: ${price}")

        # Demand forecast
        forecast = demand_forecast(sentiment_score)
        st.subheader("📊 Demand Forecast")
        st.write(forecast)

        # Market insights
        insights = market_insights(sentiment_df)
        st.subheader("📈 Market Insights")
        st.json(insights)

        # Optional: Sentiment distribution plot
        st.subheader("📊 Sentiment Distribution")
        plt.figure(figsize=(6,3))
        sns.countplot(x="sentiment", data=sentiment_df)
        st.pyplot(plt)
