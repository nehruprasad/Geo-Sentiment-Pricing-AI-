"""
Geo-Sentiment Pricing AI - Streamlit App (single-file)

Features:
- Input location and reviews
- Analyze regional sentiment
- Generate pricing recommendations
- Provide market insights
- Produce demand forecasts

How to run:
1. Create and activate a virtualenv
2. pip install -r requirements.txt
3. streamlit run geo_sentiment_pricing_ai_app.py

Minimum requirements.txt (put alongside this file):
streamlit
pandas
numpy
nltk
scikit-learn
plotly
transformers
sentence-transformers
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
from sentence_transformers import SentenceTransformer

# Ensure vader lexicon is downloaded safely
nltk_data_path = 'nltk_data'
nltk.data.path.append(nltk_data_path)
nltk.download('vader_lexicon', download_dir=nltk_data_path, quiet=True)

st.set_page_config(page_title="Geo-Sentiment Pricing AI", layout="wide")

st.title("🌍 Geo-Sentiment Pricing AI")
st.markdown("Analyze regional sentiment from reviews to get pricing recommendations, market insights, and demand forecasts.")

# ------------------------- Input Section -------------------------
st.sidebar.header("Input Configuration")
location_input = st.sidebar.text_input("Enter Location/Region", "New York")
reviews_input = st.sidebar.text_area("Enter Reviews (one per line)")
num_reviews = st.sidebar.slider("Number of Reviews to Consider", 1, 1000, 100)

if st.button("Analyze" ):
    if not reviews_input.strip():
        st.error("Please enter at least one review.")
    else:
        reviews = [r.strip() for r in reviews_input.strip().split('\n') if r.strip()]
        if len(reviews) > num_reviews:
            reviews = reviews[:num_reviews]

        st.subheader(f"Regional Sentiment Analysis for {location_input}")

        # ------------------------- Sentiment Analysis -------------------------
        try:
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = [sia.polarity_scores(r)['compound'] for r in reviews]
        except Exception as e:
            st.warning(f"VADER failed: {e}. Using transformer-based sentiment.")
            from transformers import pipeline
            sentiment_pipe = pipeline("sentiment-analysis")
            sentiment_scores = [sentiment_pipe(r)[0]['score'] if sentiment_pipe(r)[0]['label']=='POSITIVE' else -sentiment_pipe(r)[0]['score'] for r in reviews]

        sentiment_df = pd.DataFrame({'Review': reviews, 'SentimentScore': sentiment_scores})

        # Average sentiment
        avg_sentiment = sentiment_df['SentimentScore'].mean()
        st.metric("Average Sentiment Score", round(avg_sentiment, 3))

        # Sentiment Distribution
        fig = px.histogram(sentiment_df, x='SentimentScore', nbins=20, title='Sentiment Score Distribution')
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------- Market Insights -------------------------
        st.subheader("Market Insights")
        # Placeholder: top 3 positive and negative words using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(reviews)
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sum = tfidf_matrix.sum(axis=0).A1
        tfidf_df = pd.DataFrame({'word': feature_array, 'tfidf': tfidf_sum})
        top_positive = tfidf_df.sort_values('tfidf', ascending=False).head(3)
        st.write("Top Influential Words:")
        st.table(top_positive)

        # ------------------------- Pricing Recommendation -------------------------
        st.subheader("Pricing Recommendation")
        # Simple heuristic: higher sentiment => higher recommended price
        base_price = 100  # placeholder base price
        recommended_price = base_price * (1 + avg_sentiment*0.5)
        st.metric("Recommended Price", f"${recommended_price:.2f}")

        # ------------------------- Demand Forecast -------------------------
        st.subheader("Demand Forecast")
        # Placeholder: sentiment and reviews count to simulate demand
        X = np.array([[avg_sentiment, len(reviews)]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Dummy linear regression model
        demand_model = LinearRegression()
        demand_model.coef_ = np.array([50, 0.1])
        demand_model.intercept_ = 200
        predicted_demand = demand_model.predict(X_scaled)[0]
        st.metric("Predicted Demand (Units)", int(predicted_demand))

        # ------------------------- Geo-Visualization -------------------------
        st.subheader("Geo Sentiment Map")
        # For demonstration, generate random nearby points colored by sentiment
        np.random.seed(42)
        lats = 40.7 + np.random.randn(len(reviews))*0.01
        lons = -74 + np.random.randn(len(reviews))*0.01
        map_df = pd.DataFrame({'lat': lats, 'lon': lons, 'Sentiment': sentiment_scores})
        st.map(map_df)

        # ------------------------- Summary -------------------------
        st.subheader("Summary")
        st.markdown(f"- Number of Reviews Analyzed: {len(reviews)}")
        st.markdown(f"- Average Sentiment Score: {avg_sentiment:.3f}")
        st.markdown(f"- Recommended Price: ${recommended_price:.2f}")
        st.markdown(f"- Predicted Demand: {int(predicted_demand)} units")
        st.markdown("- Influential words and sentiment trends shown above.")

else:
    st.info("Enter location and reviews, then click 'Analyze' to generate insights.")
