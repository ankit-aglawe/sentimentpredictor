# sentimentclassifier/integration.py
# Author: Ankit Aglawe


from sentimentclassifier.logger import get_logger

from .classifier import SentimentClassifier

logger = get_logger(__name__)


class DataFrameSentimentClassifier:
    """A DataFrame sentiment classifier.

    Attributes:
        classifier: An instance of SentimentClassifier.
    """

    def __init__(self, model_name="roberta"):
        """Initializes the DataFrameSentimentClassifier with a specified model.

        Args:
            model_name (str): The name of the model to use.
        """
        try:
            self.classifier = SentimentClassifier(model_name=model_name)
            logger.info(
                f"Initialized DataFrameSentimentClassifier with model {model_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing DataFrameSentimentClassifier: {e}")
            raise

    def classify_dataframe(self, df, text_column):
        """Classifies the sentiment of texts in a DataFrame column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The column containing the texts to classify.

        Returns:
            pd.DataFrame: The DataFrame with an added 'sentiment' column.
        """
        try:
            df["sentiment"] = df[text_column].apply(
                lambda x: self.classifier.predict(x)["label"]
            )
            logger.info("DataFrame sentiment classification complete")
            return df
        except Exception as e:
            logger.error(f"Error classifying DataFrame: {e}")
            raise
