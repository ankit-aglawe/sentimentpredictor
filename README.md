[![python](https://img.shields.io/badge/Python-3.9|3.10|3.11|3.12|3.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![PyPI version](https://badge.fury.io/py/emotionclassifier.svg)](https://badge.fury.io/py/emotionclassifier) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

# Sentiment Analysis Classifier


A flexible sentiment analysis classifier package supporting multiple pre-trained models, customizable preprocessing, visualization tools, fine-tuning capabilities, and seamless integration with pandas DataFrames.

## Overview

`sentimentclassifier` is a Python package designed to classify sentiments in text using various pre-trained models from Hugging Face's Transformers library. This package provides a user-friendly interface for sentiment classification, along with tools for data preprocessing, visualization, fine-tuning, and integration with popular data platforms.

## Features

- **Multiple Model Support**: Easily switch between different pre-trained models.
- **Customizable Preprocessing**: Clean and preprocess text data with customizable functions.
- **Visualization Tools**: Visualize sentiment distributions and trends over time.
- **Fine-tuning Capability**: Fine-tune models on your own datasets.
- **User-friendly CLI**: Command-line interface for quick sentiment classification.
- **Integration with Data Platforms**: Seamless integration with pandas DataFrames.
- **Extended Post-processing**: Additional utilities for detailed sentiment analysis.

## Installation

You can install the package using pip:

```bash
pip install sentimentclassifier
```

## Usage

### Basic Usage

Here's an example of how to use the `SentimentClassifier` to classify a single text:

```python
from sentiment_classifier import SentimentClassifier

# Initialize the classifier with the default model
classifier = SentimentClassifier()

# Classify a single text
text = "I am very happy today!"
result = classifier.predict(text)
print("Sentiment:", result['label'])
print("Confidence:", result['confidence'])
```

### Batch Processing

You can classify multiple texts at once using the `predict_batch` method:

```python
texts = ["I am very happy today!", "I am so sad."]
results = classifier.predict_batch(texts)
print("Batch processing results:", results)
```

### Visualization

To visualize the sentiment distribution of a text:

```python
from sentiment_classifier.visualization import plot_sentiment_distribution

result = classifier.predict("I am very happy today!")
plot_sentiment_distribution(result['probabilities'], classifier.labels.values())
```

### CLI Usage

You can also use the package from the command line:

```bash
sentimentclassifier --model roberta --text "I am very happy today!"
```

### DataFrame Integration

Integrate with pandas DataFrames to classify text columns:

```python
import pandas as pd
from sentiment_classifier.integration import DataFrameSentimentClassifier

df = pd.DataFrame({
    'text': ["I am very happy today!", "I am so sad."]
})

classifier = DataFrameSentimentClassifier()
df = classifier.classify_dataframe(df, 'text')
print(df)
```

### Sentiment Trends Over Time

Analyze and plot sentiment trends over time:

```python
from sentiment_classifier.trends import SentimentAnalysisTrends

texts = ["I am very happy today!", "I am feeling okay.", "I am very sad."]
trends = SentimentAnalysisTrends()
sentiments = trends.analyze_trends(texts)
trends.plot_trends(sentiments)
```

### Fine-tuning

Fine-tune a pre-trained model on your own dataset:

```python
from sentiment_classifier.fine_tune import fine_tune_model

# Define your train and validation datasets
train_dataset = ...
val_dataset = ...

# Fine-tune the model
fine_tune_model(classifier.model, classifier.tokenizer, train_dataset, val_dataset, output_dir='fine_tuned_model')
```

### Logging Configuration

By default, the `sentimentclassifier` package logs messages at the `WARNING` level and above. If you need more detailed logging (e.g., for debugging), you can set the logging level to `INFO` or `DEBUG`:

```python
from sentiment_classifier.logger import set_logging_level

# Set logging level to INFO
set_logging_level('INFO')

# Set logging level to DEBUG
set_logging_level('DEBUG')
```

You can set the logging level to one of the following: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.


### Running Tests

Run the tests using pytest:

```bash
poetry run pytest
```


### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This package uses pre-trained models from the [Hugging Face Transformers library](https://github.com/huggingface/transformers).