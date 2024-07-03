# sentimentpredictor/utils.py
# Author: Ankit Aglawe


from sentimentpredictor.logger import get_logger

logger = get_logger(__name__)


def get_label_with_threshold(probabilities, labels, threshold=0.5):
    """Gets the label based on the given threshold.

    Args:
        probabilities (torch.Tensor): The probabilities for each label.
        labels (dict): A dictionary mapping label ids to label names.
        threshold (float): The threshold for deciding the label.

    Returns:
        str: The predicted label or 'Uncertain' if the maximum probability is below the threshold.
    """
    try:
        max_prob = probabilities.max().item()
        if max_prob > threshold:
            label = labels[probabilities.argmax().item()]
        else:
            label = "Uncertain"
        logger.debug(f"Label determined: {label} with threshold: {threshold}")
        return label
    except Exception as e:
        logger.error(f"Error getting label with threshold: {e}")
        raise
