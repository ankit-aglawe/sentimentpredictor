# sentimentpredictor/suppress_tqdm.py
# Author: Ankit Aglawe


import os
import sys
from contextlib import contextmanager

from sentimentpredictor.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def suppress_tqdm(enable=True):
    """Suppresses the tqdm output if enabled.

    Args:
        enable (bool): Whether to suppress the tqdm output.
    """
    if enable:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        logger.info("Suppressed tqdm output.")
    try:
        yield
    finally:
        if enable:
            sys.stdout = original_stdout
            logger.info("Restored stdout.")
