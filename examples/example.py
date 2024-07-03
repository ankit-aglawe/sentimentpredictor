# example.py
# Author: Ankit Aglawe

import pandas as pd
from sentimentclassifier import DataFrameSentimentClassifier, SentimentClassifier
from sentimentclassifier.trends import SentimentAnalysisTrends
from sentimentclassifier.visualization import plot_sentiment_distribution


def main():
    classifier = SentimentClassifier()

    text = "I am very happy today!"
    result = classifier.predict(text)
    print("Single Prediction:")
    print("Sentiment:", result["label"])
    print("Confidence:", result["confidence"])

    texts = ["I love this!", "I hate this."]
    batch_results = classifier.predict_batch(texts)
    print("\nBatch Prediction:")
    for i, res in enumerate(batch_results):
        print(f"Text {i+1}:")
        print("Sentiment:", res["label"])
        print("Confidence:", res["confidence"])

    plot_sentiment_distribution(result["probabilities"], classifier.labels.values())

    df = pd.DataFrame({"text": ["I am very happy today!", "I hate this."]})
    df_classifier = DataFrameSentimentClassifier()
    df = df_classifier.classify_dataframe(df, "text")
    print("\nDataFrame Sentiment Classification:")
    print(df)

    trend_analyzer = SentimentAnalysisTrends()
    sentiments = trend_analyzer.analyze_trends(
        ["I am very happy today!", "I am sad.", "I am excited.", "I am worried."]
    )
    print("\nSentiment Trends:")
    print(sentiments)
    trend_analyzer.plot_trends(sentiments)


if __name__ == "__main__":
    main()
