---
layout: default
---

# BERT4Rec

Link: https://arxiv.org/abs/1904.06690

Sequential recommendation with bidirectional context.

Core shift:
user history as a full sequence  
not left-to-right prediction  

Training pattern:
- mask items in sequence
- predict masked positions
- use both past and future context

Effects:
better intent modeling  
less recency bias than autoregressive models  

System costs:
heavier models  
slower inference  
harder real-time serving  

Often used as:
offline ranker  
reranker on shortlists  

Conceptual bridge:
NLP pretraining ideas → recommender systems
