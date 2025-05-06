import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the JSON file containing tweets into a DataFrame
df = pd.read_json("tweets_ndjson.json", lines=True)

# Check if the 'text' column is in the DataFrame
print(df.head())

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Function for sentiment using VADER
def get_vader_sentiment(text):
    score = sia.polarity_scores(text)
    return 'positive' if score['compound'] >= 0.05 else 'negative' if score['compound'] <= -0.05 else 'neutral'

# Apply sentiment analysis to the 'text' column (replace 'text' with the correct column name if needed)
df['vader_sentiment'] = df['text'].apply(get_vader_sentiment)

# Count the occurrences of each sentiment
vader_sentiment_counts = df['vader_sentiment'].value_counts()

# Plot the sentiment counts using matplotlib
plt.figure(figsize=(8, 6))
vader_sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])

# Add titles and labels
plt.title('Sentiment Analysis Results with VADER', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.show()

# Display the sentiment counts
print(vader_sentiment_counts)
