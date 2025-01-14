# Automatic Sentiment Analysis Tool for Urdu Text on Social Media Platforms

## Overview
This project develops a **Natural Language Processing (NLP) pipeline** for sentiment analysis of Urdu text extracted from various social media platforms like Twitter, Facebook, Instagram, and YouTube. The tool classifies the posts into **positive**, **negative**, or **neutral** sentiments to assist brands, influencers, and businesses in understanding Urdu-speaking users' sentiments.

## Scenario
As a data scientist working in a firm specializing in sentiment analysis, the goal is to cater to Urdu-speaking users by addressing the complexities of the Urdu language and noisy data from social media. The key task is to preprocess and classify sentiments from Urdu social media posts using a custom NLP pipeline.

## Key Features
1. **Text Preprocessing**:
   - **Stopword Removal**: Custom stopword list tailored for Urdu.
   - **Punctuation, Emoji, and Hashtag Removal**: Filtering non-informative tokens.
   - **Diacritics Removal**: Removing diacritics like Zabar, Zer, Pesh.

2. **Stemming & Lemmatization**:
   - Implementation of Urdu-specific stemming and lemmatization techniques.

3. **Feature Extraction**:
   - **Tokenization**: Properly segmenting Urdu text.
   - **TF-IDF Analysis**: Extracting relevant terms for sentiment classification.
   - **Word2Vec**: Capturing word relationships based on context.

4. **N-grams Analysis**:
   - Creation of unigrams, bigrams, and trigrams to identify common word patterns in Urdu text.

5. **Sentiment Classification Model**:
   - Machine learning models (e.g., Logistic Regression, SVM) to classify sentiment.
   - **Evaluation Metrics**: Performance metrics include accuracy, precision, recall, and F1-score.

## Challenges Addressed
- **Urdu Text Complexity**: Handling the grammatical structure, morphology, and script challenges.
- **Noisy Social Media Data**: Dealing with emojis, spelling variations, URLs, and incomplete sentences.
- **Limited Language Resources**: Development of custom Urdu NLP resources for stemming, tokenization, and sentiment lexicons.

## Tools & Libraries
- **Python** for core development
- **NLTK, spaCy, Urduhack** for text processing
- **Scikit-learn** for machine learning models
- **Gensim** for Word2Vec implementation
- **pandas, matplotlib** for data analysis and visualization

## Dataset
A publicly available **Urdu social media dataset** from platforms such as **Twitter** or **YouTube comments**, consisting of raw social media posts and their sentiment labels (positive, negative, neutral).

## Final Deliverables
1. **Text Preprocessing Results**: Cleaned Urdu text after preprocessing.
2. **Feature Extraction Results**: Tokenized text, TF-IDF scores, and Word2Vec outputs.
3. **N-gram Analysis**: Top unigrams, bigrams, and trigrams.
4. **Sentiment Classification Model**: Model performance summary (accuracy, precision, recall, F1-score).
5. **Reflection**: Challenges encountered and future optimization possibilities.

## Reflection
This project highlights the challenges of performing sentiment analysis on **Urdu text**, including the handling of complex morphology, noisy data, and limited NLP resources for Urdu. Future improvements could involve incorporating **deep learning models** like BERT fine-tuned for Urdu, or leveraging additional datasets to improve performance.
