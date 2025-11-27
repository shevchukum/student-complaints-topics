"""
wetc.py

Calculates WETC centroid - average cos similarity between
    topic words embeddings and their centroid.

Use examples:
    wc = wetc(topics, model)
    
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

model = SentenceTransformer("all-MiniLM-L6-v2")

def wetc(topics: List[List[str]], model=model) -> float:
    similarities = []
    for topic_words in topics:
      embeddings = model.encode(topic_words)
      norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
      embeddings = embeddings / norms 
  
      centroid = np.mean(embeddings, axis=0)
      centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
  
      similarities.append(np.mean(embeddings.dot(centroid)))
  
    return float(np.mean(similarities))