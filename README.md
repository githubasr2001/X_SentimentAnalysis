
# Twitter Sentiment Analysis on Vaccination Tweets

## Project Overview

This project focuses on analyzing public sentiment towards vaccinations through tweets collected from Twitter. Using Python, NLP techniques, and machine learning, we aim to uncover the general public's opinions, concerns, and attitudes towards vaccination efforts globally.

## Motivation

In the wake of global health challenges, understanding public sentiment towards vaccinations is crucial for policymakers, health organizations, and the general public. This project seeks to provide insights into these sentiments by analyzing tweets related to vaccinations.

## Data Source

The dataset `vaccination_all_tweets.csv` contains tweets related to vaccinations, including user information and tweet content. This data serves as the foundation for our sentiment analysis.

## Requirements

- Python 3.x
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- TextBlob

## Installation and Setup

Ensure you have Python installed on your system. You can then install the necessary libraries using pip:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn textblob
```

## Data Preprocessing

The preprocessing steps include:

- Cleaning text data by removing URLs, mentions, and special characters.
- Lowercasing all text for consistency.
- Tokenizing the text and removing stopwords.
- Applying lemmatization to get the root form of words.

## Sentiment Analysis

We used TextBlob to assign sentiment polarity to each tweet, categorizing them into positive, negative, or neutral sentiments.

## Machine Learning Model

A Logistic Regression model was trained on TF-IDF vectorized tweet texts to predict the sentiment of tweets. We used `CountVectorizer` to convert text data into a matrix of token counts and `TfidfTransformer` to compute TF-IDF values.

## Evaluation

The model's performance was evaluated using accuracy, precision, recall, and F1-score metrics.

## Visualization

We visualized the distribution of sentiments and created word clouds to display the most frequent words in positive, negative, and neutral tweets.

## How to Use

1. Clone the repository to your local machine.
2. Load your dataset or use the provided `vaccination_all_tweets.csv`.
3. Run the preprocessing script to clean and prepare the data.
4. Execute the sentiment analysis notebook to analyze and visualize sentiments.
5. Train the machine learning model using the training script.

## Future Work

- Explore more advanced models like neural networks for better accuracy.
- Implement other vectorization techniques like Word2Vec or GloVe.
- Expand the dataset to include tweets in different languages.

 
