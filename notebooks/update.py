# %%
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# part 1: Descriptive Statistics
data = pd.read_csv('C:/Users/hp/Desktop/week1/data/raw_analyst_ratings.csv')

# %%
headlines = data["headline"]

# %%
length_stat = headlines.str.len()
print("Statistics for Headline Lengths:")
print(length_stat.describe())

# %%
publisher_counts = data["publisher"].value_counts()
publisher_counts = publisher_counts.sort_values(ascending=False)
print("Top Publishers in descending order:",publisher_counts)

# %%
data["publication_date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S")

# Group articles by day and count the number of articles published
daily_counts = data.groupby(data["date"].dt.date).size()

# Plot the daily publication counts (same as before)
plt.figure(figsize=(12, 6))
plt.plot(daily_counts)
plt.title("Daily Publication Frequency")
plt.xlabel("Publication Date")
plt.ylabel("Number of Articles")
plt.show()

# %%
# part 2: Text Analysis(Sentiment analysis & Topic Modeling)

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


