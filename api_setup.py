import tweepy
import json

# Authenticate with Twitter API v2
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAKzQ0AEAAAAAmkpHMsFs3%2B5tSdWfXseEH4ObFt0%3DOFlmfOqqa5CTZS8zNf5DDAq87i3AySMZWusonvV5503ifeX3Te"

client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Fetch latest 100 tweets with keyword "AI"
query = "AI lang:en -is:retweet"  # Exclude retweets
tweets = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["text"])

# Save to JSON
tweets_data = [{"text": tweet.text} for tweet in tweets.data] if tweets.data else []

with open("tweets.json", "w") as file:
    json.dump(tweets_data, file, indent=4)

print("Tweets saved to tweets.json")