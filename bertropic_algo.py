from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load your data (replace 'path_to_your_tweets.json' with the actual file path)
df = pd.read_json('tweets_ndjson.json', lines=True)

# Clean the text
df['cleaned_text'] = df['text'].apply(lambda x: x.lower())  # Simple lowercase preprocessing

# Initialize SentenceTransformer to get embeddings for the tweets
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Get the embeddings for the tweets
embeddings = embedding_model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)

# Initialize BERTopic
topic_model = BERTopic(language="english", nr_topics=5)

# Fit the model
topics, _ = topic_model.fit_transform(df['cleaned_text'], embeddings)

# Print the topics
topic_info = topic_model.get_topic_info()
print(topic_info)

# Get top words for each topic
for topic_num in range(len(topic_info)):
    print(f"Topic {topic_num}: {topic_model.get_topic(topic_num)}")
