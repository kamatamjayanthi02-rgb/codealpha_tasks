import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("reviews.csv")

# Function to find sentiment
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
data["Sentiment"] = data["Review"].apply(get_sentiment)

# Count sentiments
sentiment_counts = data["Sentiment"].value_counts()

print(sentiment_counts)

# Plot chart
sentiment_counts.plot(kind="bar")

plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.savefig("sentiment_chart.png")
plt.show()