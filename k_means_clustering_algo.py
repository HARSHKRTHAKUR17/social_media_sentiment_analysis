#k-means clustering
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load NDJSON file properly
df = pd.read_json('tweets_ndjson.json', lines=True)
df = df[['text']].dropna()

# Clean tweet text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)            # remove URLs
    text = re.sub(r"@\w+", "", text)               # remove mentions
    text = re.sub(r"#\w+", "", text)               # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)        # remove special characters
    return text.lower().strip()

df['clean_text'] = df['text'].apply(clean_text)

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])

# KMeans clustering (2 clusters for bot/human)
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Show sample tweets from both clusters
print("Cluster 0 Sample Tweets:\n", df[df['cluster'] == 0]['text'].head(), "\n")
print("Cluster 1 Sample Tweets:\n", df[df['cluster'] == 1]['text'].head())

# Manually map clusters to labels after seeing samples
df['label'] = df['cluster'].map({0: 'bot', 1: 'human'})  # flip if needed

# Save labeled tweets
df.to_csv('labeled_tweets.csv', index=False)
print("\nSaved labeled tweets to labeled_tweets.csv")
