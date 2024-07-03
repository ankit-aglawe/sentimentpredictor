# sentimentpredictor/preprocess.py
# Author: Ankit Aglawe

import re

from sentimentpredictor.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """A text preprocessor for cleaning text data."""

    def clean(self, text):
        """Cleans the input text.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        try:
            text = text.lower()
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"@\w+", "", text)
            text = re.sub(r"#\w+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            logger.debug(f"Cleaned text: {text}")
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            raise
