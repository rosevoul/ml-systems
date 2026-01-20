---
layout: default
---

# Retrieval-Augmented Generation (RAG)

Link: https://arxiv.org/abs/2005.11401

Retrieval + generation split.

Core idea:
model does not need to store everything  
knowledge lives in an external index  

Typical setup:
- documents embedded offline
- ANN index built
- query encoded at inference
- top-k retrieved
- generation conditioned on retrieved context

System consequences:
retriever quality caps generator quality  
freshness handled via data, not weights  
training and serving paths diverge  

Common failure:
retrieval returns plausible but wrong context  
generator confidently elaborates  

Mitigations:
retrieval evaluation separate from generation  
log retrieved chunks  
treat retriever as first-class model  

Maps cleanly to recsys:
candidate generation → reranking → response
