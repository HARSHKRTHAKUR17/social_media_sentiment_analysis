#people and hashtags identifier
import re
import pandas as pd
from collections import Counter

# Path to your delimited JSON file
file_path = 'tweets_ndjson.json'

# Load JSON data into a DataFrame (one tweet per line)
df = pd.read_json(file_path, lines=True)

# Extract hashtags and mentions
def extract_hashtags_and_mentions(text):
    hashtags = re.findall(r'#\w+', text)  # Find all hashtags
    mentions = re.findall(r'@\w+', text)  # Find all mentions
    return hashtags, mentions

# Extract hashtags and mentions from all tweets
hashtags_and_mentions = [extract_hashtags_and_mentions(tweet) for tweet in df['text']]

# Flatten the lists
flat_hashtags = [hashtag for sublist in hashtags_and_mentions for hashtag in sublist[0]]
flat_mentions = [mention for sublist in hashtags_and_mentions for mention in sublist[1]]

# Count frequencies
hashtag_freq = Counter(flat_hashtags)
mention_freq = Counter(flat_mentions)

# Display top 10 hashtags and mentions
print("Top 10 Hashtags:", hashtag_freq.most_common(10))
print("Top 10 Mentions:", mention_freq.most_common(10))
