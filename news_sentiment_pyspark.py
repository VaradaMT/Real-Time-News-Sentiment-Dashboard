# Real-Time News Sentiment Dashboard with PySpark

import requests
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# --- Step 1: Fetch News Data ---
API_KEY = 'pub_ad5b62b3db184cba95049d7a94e644a4'
URL = f'https://newsdata.io/api/1/news?apikey={API_KEY}&language=en'

response = requests.get(URL)
data = response.json()
articles = data.get('results', [])

# Combine title + description
texts = [(article['title'] + " " + (article.get('description') or "")) for article in articles]

# --- Step 2: Create Spark DataFrame ---
spark = SparkSession.builder.appName("NewsSentimentML").getOrCreate()
df_news = spark.createDataFrame([(text,) for text in texts], ["text"])
df_news.show(5, truncate=False)

# --- Step 3: Create Training Data ---
train_data = [
    ("I love the new product launch", "Positive"),
    ("The stock market crashed today", "Negative"),
    ("The movie was fantastic", "Positive"),
    ("I am very disappointed by the service", "Negative"),
    ("Elections bring uncertainty to the market", "Negative"),
    ("This sports event is amazing", "Positive")
]
df_train = spark.createDataFrame(train_data, ["text", "label"])

# --- Step 4: Define ML Pipeline ---
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
lr = LogisticRegression(featuresCol="features", labelCol="labelIndex")

pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, label_indexer, lr])

# --- Step 5: Train Model ---
model = pipeline.fit(df_train)

# --- Step 6: Predict on Live News Data ---
predictions = model.transform(df_news)
label_converter = IndexToString(inputCol="prediction", outputCol="predicted_label",
                                labels=model.stages[3].labels)
predictions = label_converter.transform(predictions)
predictions.select("text", "predicted_label").show(truncate=False)

# --- Step 7: Save Results ---
final_results = predictions.select("text", "predicted_label")
pandas_df = final_results.toPandas()
pandas_df.to_csv("news_results.csv", index=False)
print("✅ Results saved to news_results.csv")
