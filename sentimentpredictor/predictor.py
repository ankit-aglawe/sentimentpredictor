# sentimentpredictor/predictor.py
# Author: Ankit Aglawe

import torch

from sentimentpredictor.logger import get_logger
from sentimentpredictor.model_loader import load_model_and_tokenizer
from sentimentpredictor.preprocess import Preprocessor

logger = get_logger(__name__)


class SentimentPredictor:
    """A sentiment predictor using pre-trained models.

    Attributes:
        model: The pre-trained model for sentiment analysis.
        tokenizer: The tokenizer for the model.
        preprocessor: An instance of Preprocessor for text cleaning.
        labels: A dictionary mapping label ids to label names.
    """

    def __init__(self, model_name="roberta", suppress_output=True):
        """Initializes the SentimentPredictor with a specified model.

        Args:
            model_name (str): The name of the model to use.
            suppress_output (bool): Whether to suppress the model download output.
        """
        try:
            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name, suppress_output
            )
            self.preprocessor = Preprocessor()
            self.labels = self.model.config.id2label
            logger.info(f"Initialized SentimentPredictor with model {model_name}")

        except Exception as e:
            logger.error(f"Error initializing SentimentPredictor: {e}")
            raise

    def preprocess(self, text):
        """Preprocesses the input text.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The cleaned text.
        """
        try:
            cleaned_text = self.preprocessor.clean(text)
            logger.debug(f"Preprocessed text: {cleaned_text}")
            return cleaned_text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            raise

    def predict(self, text):
        """Predicts the sentiment of a single text.

        Args:
            text (str): The input text.

        Returns:
            dict: A dictionary containing the label, confidence, and probabilities.
        """
        try:
            text = self.preprocess(text)
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            label = self.labels[probabilities.argmax().item()]
            confidence = probabilities.max().item()
            result = {
                "label": label,
                "confidence": confidence,
                "probabilities": probabilities.tolist(),
            }
            logger.info(f"Predicted sentiment: {result}")
            return result
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            raise

    def predict_batch(self, texts):
        """Predicts the sentiment of a batch of texts.

        Args:
            texts (list): A list of input texts.

        Returns:
            list: A list of dictionaries containing the label, confidence, and probabilities for each text.
        """
        try:
            texts = [self.preprocess(text) for text in texts]
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            results = [
                {
                    "label": self.labels[prob.argmax().item()],
                    "confidence": prob.max().item(),
                    "probabilities": prob.tolist(),
                }
                for prob in probabilities
            ]
            logger.info(f"Predicted batch sentiments: {results}")
            return results
        except Exception as e:
            logger.error(f"Error predicting batch sentiments: {e}")
            raise
