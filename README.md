# Geo-Sentiment-Pricing-AI-
# Geo-Sentiment Pricing AI 📍

A **Streamlit-based Python application** that uses **NLP** to analyze location-based customer reviews and provide **regional sentiment analysis, pricing recommendations, market insights, and demand forecasts**.

---

## **Features**

1. **Input Location + Reviews**  
   - Enter a city/region and multiple customer reviews (one per line).

2. **Regional Sentiment Analysis**  
   - Detects positive, negative, and neutral sentiment from reviews.  
   - Aggregates to provide a regional sentiment score.

3. **Pricing Recommendations**  
   - Suggests price adjustments based on regional sentiment.  
   - Example: positive sentiment → increase price slightly.

4. **Market Insights**  
   - Counts positive and negative reviews.  
   - Highlights common feedback examples.

5. **Demand Forecasts**  
   - Predicts demand trend (high, low, or stable) based on sentiment.

6. **Visualization**  
   - Optional plot showing sentiment distribution across reviews.

---

## **Requirements**

- Python 3.8+  
- Streamlit  
- pandas  
- NLTK  
- TextBlob  
- matplotlib  
- seaborn  

---

## **Setup Instructions**

1. **Clone the repository**:

```bash
git clone <repository-url>
cd geo_sentiment_pricing_ai
Create a virtual environment (recommended):

bash
Copy code
python -m venv venv
Activate the virtual environment:

Windows

bash
Copy code
venv\Scripts\activate
Linux / Mac

bash
Copy code
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install streamlit pandas nltk textblob matplotlib seaborn
Download NLTK punkt tokenizer:

python
Copy code
import nltk
nltk.download('punkt')
Download TextBlob corpora:

bash
Copy code
python -m textblob.download_corpora
Running the App
Ensure your virtual environment is active.

Run the Streamlit app:

bash
Copy code
streamlit run app.py
Open the URL displayed in the terminal (usually http://localhost:8501).

Enter a location and paste customer reviews (one per line).

Click "Analyze Reviews" to see sentiment analysis, pricing recommendations, market insights, and demand forecasts.

Sample Input
Location:

nginx
Copy code
San Francisco
Reviews (one per line):

sql
Copy code
Great experience with the product, very satisfied.
Service was slow but overall good.
Prices are too high for the quality offered.
Amazing quality and fast delivery.
Customer support was unhelpful.
Expected Output
Individual Review Sentiment

review	polarity	subjectivity	sentiment
Great experience with the product, very satisfied.	0.8	0.9	Positive
Service was slow but overall good.	0.1	0.4	Positive
Prices are too high for the quality offered.	-0.3	0.6	Negative
Amazing quality and fast delivery.	0.7	0.8	Positive
Customer support was unhelpful.	-0.5	0.7	Negative

Regional Sentiment

json
Copy code
{
  "average_polarity": 0.16,
  "average_subjectivity": 0.68,
  "regional_sentiment": "Positive"
}
Pricing Recommendation

nginx
Copy code
Recommended Price: $110.0
Demand Forecast

nginx
Copy code
High demand expected 📈
Market Insights

json
Copy code
{
  "positive_feedback_count": 3,
  "negative_feedback_count": 2,
  "common_positive_review": "Great experience with the product, very satisfied.",
  "common_negative_review": "Prices are too high for the quality offered."
}
Sentiment Distribution Plot

Visualizes counts of Positive, Negative, and Neutral reviews.

Notes
Works best with text-based reviews.

Pricing and demand predictions are rule-based and simplified.

Can be enhanced with ML models for accurate predictions.

Can add interactive visualizations for market trends.
