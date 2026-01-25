# Sampling-Bias-Corrected Neural Modeling (2019)

## Core Idea
Two-tower retrieval models often learn popularity rather than relevance because popular items appear too frequently as negatives during training.  
This paper adjusts the training loss so negative examples contribute proportionally to how often they are sampled.


## Model Structure
- A user tower maps user context and history to a fixed-dimensional embedding.
- An item tower maps item features to embeddings in the same space.
- Relevance is computed as a dot product between user and item embeddings.
- At serving time, user embeddings are matched against a pre-built item index using ANN search.


![Two-tower Model](images/two-tower.png)


## What Changes Compared to a Standard Two-Tower

- User and item encoders remain unchanged.
- Serving path and ANN-based retrieval remain unchanged.
- The training objective is modified:
  - Negative samples are reweighted based on their sampling probability.
  - Popular items no longer dominate gradient updates.

This is a training-time change only, not a serving or architecture change.

## Tradeoffs

- Bias correction improves coverage of tail items, but depends on reasonable estimates of sampling frequency.
- Training remains efficient, but gradient quality is still influenced by batch composition.
- Retrieval recall improves, but ranking behavior is unchanged.

##  Mental Note

A two-tower model is trained with in-batch negatives.  
Popular items appear in nearly every batch, while tail items appear rarely.

The model learns to strongly downscore popular items and largely ignore tail items.  
Embeddings become organized around popularity rather than relevance.

Reweighting negatives reduces this effect without changing indexing or serving infrastructure.

## Takeaway
If a two-tower model retrieves mostly popular items despite sufficient data and capacity, the issue may lie in the training objective.  
Bias-corrected negative sampling can improve recall without changing model architecture or serving latency.
