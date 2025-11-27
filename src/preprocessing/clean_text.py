"""
clean_text.py

1. Normalize text
2. Lemmatize & tokenize
3. Apply bigrams and trigrams

Use examples:
  cleaned = normalize_texts(texts)  - returns cleaned texts list 
  corpus = lemmatize_and_tokenize(cleaned) - returns a list of lists of lemmatized tokens
  ngrammed = apply_ngrams(corpus)  - returns list of lists of tokens with n-grams
"""

import re
import spacy
from nltk.corpus import stopwords
from nltk import download
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from typing import List, Iterable

# --------------------------
# Basic text normalization
# --------------------------

def normalize_texts(texts: List[str]) -> List[str]:
    """Remove URLs, non-letters, extra spaces and return a list of lowercased strings."""
    corpus = []
    for text in texts:
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^a-zA-Z ]", " ", text)
        text = re.sub(r"\s+", " ", text)
        corpus.append(text.strip().lower())
    return corpus

# ---------------------------------
# Lemmatization and tokenization
# ---------------------------------

# Qietly download stopwords
download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))        # 198 words not helping to identify topic, for example: 'while', 'might', "he'd", 'his' ... 

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])   #NER will add entity tokens which will not help LSA/LDA
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

domain_stopwords = [    
    "make", "need", "hard", "difficult", "wish", "really", "feel", "experience",
    "get", "feel_like", "often", "student", "university", "lack", "take", "enough",
    "challenge", "important", "try", "keep", "high", "way", "always", "due", 
    "would", "would_great", "like", "able", "different", "seem", "especially", 
    "could", "frustrating", "easy", "worried", "sure", "tough", "interested", "nice",
    "disappointing", "frustrated", "good", "stressful", "unfair", "outrageous",
    "stressed", "unacceptable", "discouraging", "unhelpful", "terrible", "unhealthy", 
    "harsh", "scared", "subjective", "glitchy", "disconnected", "unprepared", "thorough",
    "scary", "unfocused", "harmful", "dead", "unbearable", "crippling",
    "negative", "impatient", "dismissive", "embarrassing", "disrespectful",
    "cross", "discouraged", "incomplete", "irrelevant", "inefficient", "ridiculous", 
    "unclear", "exorbitant", "bad", "uncomfortable", "unacceptable", "unhelpful", "unreliable",
    "unsupported", "unnecessary"
]

def lemmatize_and_tokenize(texts: List[str], domain_stopwords: List[str] = domain_stopwords) -> List[List[str]]:
    """Convert texts into lemmatized tokens."""
    corpus = []
    for text in texts:
        doc = nlp(text)                    # lemmatization by spaCy
        tokens = [
            token.lemma_
            for token in doc
            if token.lemma_ not in STOPWORDS
            and token.lemma_ not in domain_stopwords
            and len(token.lemma_) > 2     # removing one or two char words
        ]
        corpus.append(tokens)
    return corpus


# ----------------------------------------------------
# Implement bi-grams: ["new", "york"] -> ["new_york"]
# ----------------------------------------------------

def apply_bigrams(texts: List[List[str]],
                 min_count: int = 5,                                           # min times bi-gram should be found in the corpus    
                 threshold: float = 20.0) -> List[List[str]]:                  # how words in bi-gram are found togerther more often than separately  
    """
    Builds bi-gram model using gensim Phrases.
    Applies model to tokenized texts. 
    """
    # 1. Build bigram model
    bigram = Phrases(texts, min_count=min_count, threshold=threshold)
    bigram_model = Phraser(bigram)

    # 2. Apply model to get ngrammed texts
    birammed_texts = [bigram_model[text] for text in texts]

    return birammed_texts

