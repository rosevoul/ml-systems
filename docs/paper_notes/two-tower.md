---
layout: default
---

# Two-Tower Models (DSSM-style)

Link: https://arxiv.org/abs/1608.07428

Decoupled user and item representations.

Structure:
user encoder  
item encoder  
dot product / cosine similarity  

Key benefit:
embeddings precomputed  
ANN retrieval at scale  

Serving pattern:
- user embedding computed online
- item embeddings static or batched
- nearest-neighbor search

Tradeoffs:
expressiveness limited  
interaction features hard to model  

Typical placement:
candidate generation layer  
feeds rankers and rerankers  

This is the backbone:
large catalogs  
low-latency personalization
