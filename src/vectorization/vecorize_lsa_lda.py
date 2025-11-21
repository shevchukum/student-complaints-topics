"""
vectorize_lsa_lda.py

1. Build dictionary and corpus for LSA/LDA
2. Convert tokenized texts to BoW or TF-IDF

Use examples:
  dictionary = build_dictionary(tokenized_texts, no_below=1, no_above=0.9)  - build Gensim dictionary (better than standard Python dictionary) from tokenized_texts, too rare and too frequent tokens are removed
  corpus_bow, _ = build_corpus(tokenized_texts, dictionary, vector_type="bow") - returns list of BoW vecotors (word index in dictionary, word count in the doc), for example: [(0, 1), (1, 1), (2, 1), (3, 1)] 
  corpus_tfidf, tfidf_model = build_corpus(tokenized_texts, dictionary, vector_type="tfidf") - returns list of TF-IDF vectors (word index in dictionary, TF-IDF value), for example: [(0, np.float64(0.5646732768699807)), (1, np.float64(0.2525147628886298))]

"""

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from typing import List, Tuple

def build_dictionary(tokenized_texts: List[List[str]],
                     no_below: int = 5,
                     no_above: float = 0.5) -> Dictionary:
    """
    Build a Gensim dictionary from tokenized texts.

    Parameters:
    - no_below: keep tokens appearing in at least no_below documents
    - no_above: remove tokens appearing in more than no_above fraction of documents

    Returns:
    - Gensim Dictionary
    """
    dictionary = Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    return dictionary

def build_corpus(tokenized_texts: List[List[str]],
                 dictionary: Dictionary,
                 vector_type: str = "bow") -> List[List[Tuple]]:
    """
    Convert tokenized texts to a corpus for LSA/LDA.

    Parameters:
    - tokenized_texts: list of token lists
    - dictionary: Gensim Dictionary
    - vector_type: "bow" or "tfidf"

    Returns:
    - corpus: list of doc vectors
    - model: TfidfModel if vector_type=="tfidf", else None
    """
    # Bag-of-Words corpus
    corpus_bow = [dictionary.doc2bow(text) for text in tokenized_texts]

    if vector_type.lower() == "tfidf":
        tfidf_model = TfidfModel(corpus_bow)
        corpus_tfidf = tfidf_model[corpus_bow]
        return corpus_tfidf, tfidf_model
    else:
        return corpus_bow, None