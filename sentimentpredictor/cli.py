# sentimentclassifier/cli.py
# Author: Ankit Aglawe


import click

from sentimentpredictor.logger import get_logger
from sentimentpredictor.predictor import SentimentPredictor

logger = get_logger(__name__)


@click.command()
@click.option("--model", default="roberta", help="Model name")
@click.option(
    "--text",
    prompt="Text to classify",
    help="Text for sentiment analysis classification",
)
def classify_text(model, text):
    """Classifies the sentiment of the input text using the specified model.

    Args:
        model (str): The name of the model to use.
        text (str): The text to classify.
    """
    try:
        predictor = SentimentPredictor(model_name=model)
        result = predictor.predict(text)
        click.echo(f"Sentiment: {result['label']}")
        click.echo(f"Confidence: {result['confidence']}")
    except Exception as e:
        logger.error(f"Error in CLI sentiment classification: {e}")
        click.echo(f"Error: {e}")


if __name__ == "__main__":
    classify_text()
