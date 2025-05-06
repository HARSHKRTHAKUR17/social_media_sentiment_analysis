import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load the JSON file containing tweets into a DataFrame
df = pd.read_json("tweets_ndjson.json", lines=True)

# Check if the 'text' column is in the DataFrame
print(df.head())

# Function for sentiment using TextBlob
def get_textblob_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    return 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'

# Apply sentiment analysis to the 'text' column (replace 'text' with the correct column name if needed)
df['textblob_sentiment'] = df['text'].apply(get_textblob_sentiment)

# Count the occurrences of each sentiment
sentiment_counts = df['textblob_sentiment'].value_counts()

# Plot the sentiment counts using matplotlib
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])

# Add titles and labels
plt.title('Sentiment Analysis Results with TextBlob', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.show()

# Display the sentiment counts
print(sentiment_counts)
