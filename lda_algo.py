#LDA
import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download stopwords if not already available
nltk.download('stopwords')
from nltk.corpus import stopwords

# Path to your delimited JSON file
file_path = 'tweets_ndjson.json'

# Load JSON data into a DataFrame (one tweet per line)
df = pd.read_json(file_path, lines=True)

# Preprocess the text data
def preprocess_text(text):
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Apply preprocessing to all tweets
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Tokenization and removing stopwords
stop_words = stopwords.words('english')

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words)

# Convert the cleaned text into a matrix of TF-IDF features
X = vectorizer.fit_transform(df['cleaned_text'])

# Apply LDA (Latent Dirichlet Allocation)
num_topics = 5  # Number of topics to extract
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Display the top words for each topic
n_words = 10
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx + 1}:")
    print([feature_names[i] for i in topic.argsort()[-n_words:]])
    print()
