# Graph Neural Networks for Recommendation

## Context
Graph Neural Networks (GNNs) are used in recommendation when user–item interactions
are naturally modeled as graphs and neighborhood structure carries signal beyond
independent interactions.

Typical setting: a bipartite graph with users and items as nodes and interactions
(clicks, purchases, views) as edges.

## Core idea
Instead of learning user and item embeddings independently, GNNs iteratively
**propagate information along graph edges** so each node embedding incorporates
signals from its neighbors.

Message passing intuition:
- Users aggregate information from interacted items
- Items aggregate information from interacting users
- Repeated over multiple hops to capture collaborative structure

## LightGCN vs GAT
**LightGCN**
- Removes nonlinearities and feature transformations
- Pure neighborhood aggregation
- Optimized for recommendation
- Strong empirical performance with low complexity

**GAT (Graph Attention Networks)**
- Learns attention weights over neighbors
- More expressive
- Higher computational cost
- Often unnecessary for large-scale recommender graphs

In practice, LightGCN is preferred for large recommender systems due to simplicity
and scalability.

## When GNNs beat matrix factorization
GNNs outperform matrix factorization when:
- Interaction graph is sparse but locally informative
- Higher-order connectivity matters (friends-of-friends effects)
- Cold-start items can benefit from neighbor structure

They provide limited gains when:
- Data is dense
- User behavior is mostly independent
- Strong side features already dominate

## Scaling constraints
Key limitations:
- Memory footprint grows with edges
- Neighborhood sampling required at scale
- Training often offline with periodic refresh
- Online inference typically uses precomputed embeddings

This makes GNNs more suitable for candidate generation or offline embedding learning
than real-time end-to-end ranking.

## Applied example
Implemented a toy LightGCN-style link prediction example on synthetic user–item data
using PyTorch Geometric.

## Evaluation considerations
Offline:
- Recall@K
- NDCG
- Link prediction AUC

Online:
- CTR lift
- Conversion
- Latency impact (embedding refresh frequency)


