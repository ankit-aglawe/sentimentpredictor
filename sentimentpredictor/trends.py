# sentimentpredictor/trends.py
# Author: Ankit Aglawe


import matplotlib.pyplot as plt

from sentimentpredictor.logger import get_logger
from sentimentpredictor.predictor import SentimentPredictor

logger = get_logger(__name__)


class SentimentAnalysisTrends:
    """Analyzes and plots sentiment trends over time.

    Attributes:
        classifier: An instance of SentimentClassifier.
    """

    def __init__(self, model_name="roberta"):
        """Initializes the SentimentAnalysisTrends with a specified model.

        Args:
            model_name (str): The name of the model to use.
        """
        try:
            self.classifier = SentimentPredictor(model_name=model_name)
            logger.info(f"Initialized SentimentAnalysisTrends with model {model_name}")
        except Exception as e:
            logger.error(f"Error initializing SentimentAnalysisTrends: {e}")
            raise

    def analyze_trends(self, texts):
        """Analyzes sentiment trends in the provided texts.

        Args:
            texts (list): A list of texts to analyze.

        Returns:
            list: A list of sentiment labels for the texts.
        """
        try:
            sentiments = [self.classifier.predict(text)["label"] for text in texts]
            logger.info(f"Analyzed sentiment trends: {sentiments}")
            return sentiments
        except Exception as e:
            logger.error(f"Error analyzing sentiment trends: {e}")
            raise

    def plot_trends(self, sentiments):
        """Plots sentiment trends over time.

        Args:
            sentiments (list): A list of sentiment labels.
        """
        try:
            plt.plot(range(len(sentiments)), sentiments)
            plt.xlabel("Text Index")
            plt.ylabel("Sentiment")
            plt.title("Sentiment Trends Over Time")
            plt.xticks(rotation="vertical")
            plt.show()
            logger.info("Plotted sentiment trends.")
        except Exception as e:
            logger.error(f"Error plotting sentiment trends: {e}")
            raise
