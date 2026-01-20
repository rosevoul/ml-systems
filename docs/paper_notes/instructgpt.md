---
layout: default
---

# InstructGPT

Link: https://arxiv.org/abs/2203.02155

Preference-driven training.

Pipeline shape:
- base model pretraining
- supervised fine-tuning
- preference comparisons
- reward modeling
- policy optimization

System implications:
multi-stage training  
human feedback as data source  
model updates tied to product loops  

Recommender parallel:
preferences > clicks  
reward design ≈ ranking loss design  

Operational risks:
reward misspecification  
feedback loops  
distribution shift  

Primary contribution:
formalize preference learning as a system component
