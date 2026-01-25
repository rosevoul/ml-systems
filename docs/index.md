---
layout: default
title: Home
---

## About

I work on machine learning systems where modeling choices shape user experience, business outcomes, and long-term system behavior.

My focus is on:
- Translating abstract models into production-ready systems  
- Choosing the appropriate level of complexity for the problem  
- Making trade-offs, uncertainty, and failure modes explicit  

I’ve shipped end-to-end ML systems, from data pipelines and modeling to deployment, monitoring, and iteration, with direct ownership of production metrics. In recent work, I owned multiple production ML systems that drove measurable lift in engagement and tens of millions in incremental revenue.


<a id="selected-technical-notes"></a>
## Selected Technical Notes

1) **Recommender Embeddings: From Simple to Modern**
   - Co-occurrence and matrix factorization → item2vec → graph/network embeddings → multimodal embeddings.
   - Focus: when each method is appropriate, common failure modes, and how embeddings feed retrieval and ranking.
   - Notebook: [Embedding Methods for Recommenders](notebooks/recsys_embeddings.ipynb)

2) **Retrieval + Ranking, the Production Way**
   - Candidate generation (ANN/FAISS-style), two-tower retrieval, feature pipelines, pointwise vs pairwise losses, calibration, latency and throughput tradeoffs.
   - Focus: system boundaries, offline/online consistency, and safe iteration in production.
   - Notebook: [Production Retrieval & Ranking Pipeline](notebooks/retrieval_ranking_production.ipynb)

3) **Inference-Time Enhancements for Recommenders**
   - Reranking, query and item rewriting, hybrid scoring, structured outputs, caching, latency and cost controls, fallback logic.
   - Focus: inference-time pipeline steps that improve relevance without destabilizing the system.
   - Notebook: [Inference-Time Enhancements](notebooks/genai_recsys_inference.ipynb)

4) **Monitoring + Metrics for Recommenders**
   - Feature and target drift, segment health, ranking quality metrics, online monitoring, alerting, and incident triage.
   - Safeguards: policy checks, safety filters, and controlled fallback behavior.
   - Notebook: [Monitoring & Safeguards for Recommenders](notebooks/recsys_monitoring_guardrails.ipynb)



<a id="reference-papers"></a>
## Reference Papers

### GenAI & Multimodal Foundations

- **Attention Is All You Need (2017)**  
  [Paper](https://arxiv.org/abs/1706.03762) · [Notes](notes/transformers.md)

- **CLIP: Learning Transferable Visual Models from Natural Language Supervision (2021)**  
  [Paper](https://arxiv.org/abs/2103.00020) · [Notes](notes/clip.md)

- **Retrieval-Augmented Generation (RAG, 2020)**  
  [Paper](https://arxiv.org/abs/2005.11401) · [Notes](notes/rag.md)


### Recommender Systems

- **Deep Neural Networks for YouTube Recommendations (2016)**  
  [Paper](https://research.google/pubs/pub45530/) · [Notes](paper_notes/youtube-two-stage-retrieval.md)

- **Sampling-Bias-Corrected Neural Modeling (Two-Tower Retrieval, 2019)**  
  [Paper](https://dl.acm.org/doi/10.1145/3298689.3346996) · [Notes](paper_notes/two-tower.md)

- **Deep Learning Recommendation Model (DLRM, Meta, 2019)**  
  [Paper](https://arxiv.org/abs/1906.00091) · [Notes](paper_notes/dlrm.md)


### ML Foundations

- **XGBoost: A Scalable Tree Boosting System (2016)**  
  [Paper](https://arxiv.org/abs/1603.02754) · [Notes](paper_notes/xgboost.md)

- **A Few Useful Things to Know About Machine Learning (2012)**  
  [Paper](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) · [Notes](paper_notes/useful-things-ml.md)


