# sentimentpredictor/model_loader.py
# Author: Ankit Aglawe


from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentimentpredictor.logger import get_logger
from sentimentpredictor.suppress_tqdm import suppress_tqdm

logger = get_logger(__name__)


def load_model_and_tokenizer(model_name, suppress_output=True):
    """Loads the model and tokenizer for the specified model name.

    Args:
        model_name (str): The name of the model to load.
        suppress_output (bool): Whether to suppress the output of the model download. # ! Test if this is working

    Returns:
        tuple: The model and tokenizer.
    """
    try:
        with suppress_tqdm(suppress_output):
            logger.info(f"Downloading model and tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                f"cardiffnlp/twitter-{model_name}-base-sentiment-latest"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                f"cardiffnlp/twitter-{model_name}-base-sentiment-latest"
            )
        logger.info("Download complete.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        raise
