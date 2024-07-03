# tests/test.py
# Author: Ankit

import pandas as pd
import pytest  # type: ignore
from sentimentclassifier import DataFrameSentimentClassifier, SentimentClassifier


@pytest.fixture
def classifier():
    return SentimentClassifier()


def test_single_prediction(classifier):
    result = classifier.predict("I am very happy today!")
    assert result["label"] in classifier.labels.values()
    assert 0 <= result["confidence"] <= 1


def test_batch_prediction(classifier):
    texts = ["I love this!", "I hate this."]
    results = classifier.predict_batch(texts)
    assert len(results) == 2
    for result in results:
        assert result["label"] in classifier.labels.values()
        assert 0 <= result["confidence"] <= 1


def test_dataframe_integration():
    df = pd.DataFrame({"text": ["I am very happy today!", "I hate this."]})
    df_classifier = DataFrameSentimentClassifier()
    df = df_classifier.classify_dataframe(df, "text")
    assert "sentiment" in df.columns
    assert df["sentiment"].iloc[0] in df_classifier.classifier.labels.values()
    assert df["sentiment"].iloc[1] in df_classifier.classifier.labels.values()
