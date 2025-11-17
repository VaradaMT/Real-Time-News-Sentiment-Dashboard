#  Real-Time News Sentiment Dashboard with PySpark (2025)

**Technologies:** PySpark, Streamlit, API Integration, ML Pipeline

Created a real-time sentiment classification pipeline for live news headlines using **Spark Structured Streaming** and a **PySpark ML model**.  
Integrated automatic ingestion from news APIs and built a dashboard (Streamlit) to visualize sentiment trends and predictions.

---

## Features
- Fetches live news using **NewsData.io API**
- Performs NLP preprocessing with **PySpark MLlib**
- Trains a logistic regression model for sentiment detection
- Saves classified results as CSV

---

## Tech Stack
- PySpark (MLlib, DataFrames)
- Requests API
- Pandas

---

##  Run the Project
```bash
pip install -r requirements.txt
python news_sentiment_pyspark.py
