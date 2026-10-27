"""Baseline B4: pure-text Sentence-BERT embedding (no structure, no contrastive)."""
from ecd.embedding import SBertEmbedder

def make():
    return SBertEmbedder()
