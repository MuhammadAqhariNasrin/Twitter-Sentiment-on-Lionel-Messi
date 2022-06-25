# Twitter-Sentiment-on-Lionel-Messi
A twitter sentiment analysis on Lionel Messi keyword by classifying the Positive, Negative, and Neutral tweets by machine learning models for classification, text mining, text analysis, data analysis, and data visualization.

# INTRODUCTION
Natural Language Processing (NLP) is a hotbed of research in data science these days and one of the most common applications of NLP is sentiment analysis. From opinion polls to creating entire marketing strategies, this domain has completely reshaped the way businesses work, which is why this is an area every data scientist must be familiar with.

Thousands of text documents can be processed for sentiment (and other features including named entities, topics, themes, etc.) in seconds, compared to the hours it would take a team of people to manually complete the same task.

I will do so by following a sequence of steps needed to solve a general sentiment analysis problem. I will start with preprocessing and cleaning of the raw text of the tweets. Then we will show how  the cleaned text and try to get some intuition about the context of the tweets. After that, I will extract numerical features from the data and finally use these feature sets to train models and identify the sentiments of the tweets.

# Tweets Preprocessing and Cleaning

The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text. The preprocessing of the text data is an essential step as it makes the raw text ready for mining, for example, it becomes easier to extract information from the text and apply machine learning algorithms to it.
