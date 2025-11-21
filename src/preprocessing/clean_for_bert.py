"""
clean_for_bert.py

Munimal clearning for BERT-topic: URL's, multiple spaces and simbols, leading/trailing spaces.

Use example:
  cleaned = clean_texts_for_bert(texts) - returns cleaned list of strings
"""

import re
from typing import List

def clean_texts_bert(texts: List[str]) -> List[str]:
    """
    Removes URLs, multiple spaces and simbols, strips leading/trailing spaces.
    """
    corpus = []
    for text in texts:
        # Remove URLs
        text = re.sub(r"http\S+", " ", text)
    
        # Remove repeated symbols (e.g., !!!!, ???)
        text = re.sub(r"([!?.,])\1+", r"\1", text)
    
        # Normalize spaces
        text = re.sub(r"\s+", " ", text)
    
        corpus.append(text.strip())
    return corpus