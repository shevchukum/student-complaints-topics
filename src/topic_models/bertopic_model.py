"""
bertopic_model.py

1. Trains BERTopic model on texts.
2. Can specify number of topics or let BERTopic автоматически определить.
3. Print top words for each topic.
4. Save and load BERTopic model.

Use example:
    bert_model = train_bertopic(texts_cleaned, nr_topics=11)
    print_topics(bert_model, top_n_words=10)
    save_model(bert_model, "models/bertopic/bert_1")  - bert_1 is a folder for config and other files
"""

from bertopic import BERTopic
from typing import List, Any
from umap import UMAP
from hdbscan import HDBSCAN

umap_model = UMAP(
    n_neighbors=50,
    n_components=10,
    random_state=42,
    metric="cosine"
)
min_topic_size = 10
hdbscan_model = HDBSCAN(
  min_cluster_size=min_topic_size, 
  prediction_data=True,
  gen_min_span_tree=True
)


def train_bertopic(texts: List[str],
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    nr_topics: int = None,
                    min_topic_size: int = min_topic_size,
                    top_n_words: int = 20,
                    vectorizer: Any = None,
                    verbose: bool = True,
                    calculate_probabilities: bool = False) -> BERTopic:
    """
    Train BERTopic model.
    
    Parameters:
        texts: List of strings (preprocessed or raw).
        nr_topics: Number of topics to reduce to. If None, model chooses itself.
        min_topic_size: Minimum number of documents per topic.
        verbose: Show progress bar.
        calculate_probabilities: True for soft topic labeling
    """
    model = BERTopic(
      umap_model=umap_model,
      hdbscan_model=hdbscan_model,
      vectorizer_model=vectorizer, 
      nr_topics=nr_topics, 
      top_n_words=top_n_words, 
      min_topic_size=min_topic_size, 
      verbose=verbose, 
      calculate_probabilities=calculate_probabilities
    )
    
    topics, _ = model.fit_transform(texts)
    return model, topics

def print_topics(model: BERTopic, top_n_words: int = 10):
    """
    Print top words for each topic in BERTopic model.
    """
    topics_info = model.get_topic_info()
    for topic_id in topics_info.Topic.unique():
        if topic_id == -1:  # outliers
            continue
        topic_words = model.get_topic(topic_id)
        words_str = ", ".join([word for word, _ in topic_words[:top_n_words]])
        print(f"Topic {topic_id}: {words_str}")

def save_model(model: BERTopic, path: str):
    """
    Save BERTopic model.
    """
    model.save(path)

def load_model(path: str) -> BERTopic:
    """
    Load BERTopic model.
    """
    return BERTopic.load(path)