"""
lsa_lda_models.py

1. Trains LSA model on corpus: TF-IDF prefered over BoW (less noise)
2. Trains LDA model on coprus: BoW prefered over TF-IDF (real frequences needed)
3. Print top words for each topic in LSA or LDA model
4. Save and load models

Use example:
    lsa_model = train_lsa(tfidf_corpus, dictionary, num_topics=11)
    lda_model = train_lda(corpus, dictionary, num_topics=11, passes=15)
    print_topics(lsa_model, num_words=8) - example: Topic 0: 0.447*"fail" + 0.447*"repair" + 0.447*"road" + 0.447*"council" + 0.447*"city" ...
     - numbers before the words - for LSA significance of the word in the topic, for LDA - probability of the word in the topic
    save_model(lsa_model, "models/lsa/lsa_1.lsi") - ".lsi" is standard for LSA, ".lda" - for LDA. 

"""

from gensim.models import LsiModel, LdaModel
from gensim.corpora import Dictionary
from typing import List, Tuple

def train_lsa(corpus: List[List[Tuple]], dictionary: Dictionary, num_topics=11) -> LsiModel:
    """
    Train LSA (Latent Semantic Analysis) model
    """
    lsa_model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    return lsa_model

def train_lda(corpus: List[List[Tuple]], dictionary: Dictionary, num_topics=11, passes=10, random_state=42) -> LdaModel:
    """
    Train LDA (Latent Dirichlet Allocation) model
    """
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state
    )
    return lda_model

def print_topics(model, num_words=10):
    """
    Print top-10 words for each topic in LSA or LDA model
    """
    for idx, topic in model.print_topics(num_words=num_words):
        print(f"Topic {idx}: {topic}")

def save_model(model, path):
    model.save(path)

def load_model(path):
    from gensim.models import LsiModel, LdaModel
    if path.endswith(".lda"):
        return LdaModel.load(path)
    else:
        return LsiModel.load(path)