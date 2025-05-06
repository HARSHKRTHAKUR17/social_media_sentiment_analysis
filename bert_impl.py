from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Apply model to each tweet in the DataFrame
df['bert_sentiment'] = df['text'].apply(lambda text: sentiment_model(text)[0]['label'])

# Display the first few rows with the sentiment labels
#print(df[['text', '']].head())
print(df[['text', 'bert_sentiment']].to_string(index=False))
