import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Optional
from collections import Counter
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Use NLTK's stopwords plus some custom ones
STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = {'would', 'could', 'should', 'might', 'must', 'need', 'shall', 'will', 'also', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}
STOPWORDS.update(CUSTOM_STOPWORDS)

def preprocess_text(text: str,
                   use_stemming: bool = True,  # Changed default to True
                   use_lemmatization: bool = False,  # Changed default to False
                   min_length: int = 2,
                   remove_numbers: bool = True) -> List[str]:
   
    # Initialize stemmers/lemmatizers if needed
    stemmer = PorterStemmer() if use_stemming else None
    lemmatizer = WordNetLemmatizer() if use_lemmatization else None

    # Clean text
    text = text.lower()
    # Remove special characters but keep hyphens in compound words
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Remove URLs, emails, and references
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+|\[.*?\]|\(.*?\)', '', text)
    if remove_numbers:
        text = re.sub(r'\b\d+\b', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and process
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if (
        len(token) >= min_length and  # Length check
        token.isalnum() and           # Alphanumeric only
        token not in STOPWORDS and    # Not a stopword
        not token.startswith('-') and # Not starting with hyphen
        not token.endswith('-')       # Not ending with hyphen
    )]

    # Apply stemming/lemmatization
    if stemmer:
        tokens = [stemmer.stem(token) for token in tokens]
    elif lemmatizer:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def get_term_frequencies(tokens: List[str]) -> Counter:
    """Calculate term frequencies."""
    return Counter(tokens) 