from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Function for sentiment
def get_vader_sentiment(text):
    score = sia.polarity_scores(text)
    return 'positive' if score['compound'] >= 0.05 else 'negative' if score['compound'] <= -0.05 else 'neutral'

# Apply function to tweets
df['vader_sentiment'] = df['text'].apply(get_vader_sentiment)
print(df[['text', 'vader_sentiment']].to_string(index=False))