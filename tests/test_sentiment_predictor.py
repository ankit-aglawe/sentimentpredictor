# tests/test.py
# Author: Ankit

import pandas as pd
import pytest  # type: ignore
from sentimentpredictor import DataFrameSentimentPredictor, SentimentPredictor


@pytest.fixture
def predictor():
    return SentimentPredictor()


def test_single_prediction(predictor):
    result = predictor.predict("I am very happy today!")
    assert result["label"] in predictor.labels.values()
    assert 0 <= result["confidence"] <= 1


def test_batch_prediction(predictor):
    texts = ["I love this!", "I hate this."]
    results = predictor.predict_batch(texts)
    assert len(results) == 2
    for result in results:
        assert result["label"] in predictor.labels.values()
        assert 0 <= result["confidence"] <= 1


def test_dataframe_integration():
    df = pd.DataFrame({"text": ["I am very happy today!", "I hate this."]})
    df_predictor = DataFrameSentimentPredictor()
    df = df_predictor.classify_dataframe(df, "text")
    assert "sentiment" in df.columns
    assert df["sentiment"].iloc[0] in df_predictor.predictor.labels.values()
    assert df["sentiment"].iloc[1] in df_predictor.predictor.labels.values()
