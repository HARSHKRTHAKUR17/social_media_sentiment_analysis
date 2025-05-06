from textblob import TextBlob

# Function for sentiment
def get_textblob_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    return 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'

df['textblob_sentiment'] = df['text'].apply(get_textblob_sentiment)
#print(df[['text', 'textblob_sentiment']].head())
print(df[['text', 'textblob_sentiment']].to_string(index=False))