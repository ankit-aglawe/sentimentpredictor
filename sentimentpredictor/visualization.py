# sentimentpredictor/visualization.py
# Author: Ankit Aglawe


import matplotlib.pyplot as plt

from sentimentpredictor.logger import get_logger

logger = get_logger(__name__)


def plot_sentiment_distribution(sentiment_probs, labels):
    """Plots the distribution of sentiment probabilities.

    Args:
        sentiment_probs (list): A list of sentiment probabilities.
        labels (list): A list of sentiment labels.
    """
    try:
        if isinstance(sentiment_probs[0], list):
            sentiment_probs = sentiment_probs[0]
        labels = list(labels)
        sentiment_probs = [float(prob) for prob in sentiment_probs]
        plt.bar(labels, sentiment_probs)
        plt.xlabel("Sentiments")
        plt.ylabel("Probability")
        plt.title("Sentiment Distribution")
        plt.show()
        logger.info("Plotted sentiment distribution.")
    except Exception as e:
        logger.error(f"Error plotting sentiment distribution: {e}")
        raise
