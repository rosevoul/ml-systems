# Graph Neural Networks for Recommendation

## Context
Graph Neural Networks (GNNs) are used in recommendation when user–item interactions
are best represented as a graph and neighborhood structure carries signal beyond
independent interactions.

The canonical setup is a **bipartite graph**:
- Users and items are nodes
- Interactions (views, clicks, purchases) are edges

---

## Synthetic graph used in examples

The notebooks linked below use the same synthetic graph throughout.

```
Users (0–4)                      Items (5–10)

U0 (0) ───────────────▶ I5 (5)
  │
  └──────────────▶     I6 (6)

U1 (1) ───────────────▶ I6 (6)

U2 (2) ───────────────▶ I7 (7)

U3 (3) ───────────────▶ I8 (8)

U4 (4) ───────────────▶ I9 (9)
  └──────────────▶     I10 (10)
```

The graph is made **undirected** for message passing so information flows:
```
User ⇄ Item
```

---

## Core idea
GNNs learn node embeddings by **iteratively aggregating information from neighbors**.

In recommendation:
- Users aggregate signals from interacted items
- Items aggregate signals from interacting users
- Repeating this process captures collaborative structure beyond direct interactions

---

## System design variants shown

### A) LightGCN (graph-only)

```
Inputs:
  - Graph edges
  - Trainable embedding table (one vector per node)

Propagation:
  Mean aggregation over neighbors
  No node features
  No nonlinear transformations

Output:
  Refined embeddings for users and items
```

Key properties:
- Pure collaborative filtering
- Low parameter count
- Scales well to large graphs

---

### B) Feature-aware GNN (GraphSAGE-style)

```
Inputs:
  - Graph edges
  - Node feature matrix X (text, image, metadata, etc.)

Propagation:
  Feature transformation + neighborhood aggregation
  Nonlinearities (e.g., ReLU)

Output:
  Feature-enriched node embeddings
```

---

### C) Hybrid design

```
Step 1: Graph encoder
  Graph edges ──LightGCN──▶ Graph embeddings Eg

Step 2: Feature pipeline
  Content / metadata ──▶ Feature vectors X

Step 3: Downstream fusion
  [Eg || X] ──MLP / ranker──▶ Final representation
```


Key properties:
- Keeps graph learning simple
- Allows independent refresh cycles
- Easier to debug and operate at scale

This pattern is common in large production recommender systems.

---

## When GNNs beat matrix factorization
GNNs tend to outperform MF when:
- Interaction graph is sparse but locally informative
- Higher-order connectivity matters
- Cold-start items benefit from neighborhood structure

Limited gains when:
- Data is dense
- User behavior is mostly independent
- Strong content models already dominate

---

## Scaling constraints
Practical limitations:
- Memory footprint grows with number of edges
- Neighborhood sampling required at scale
- Training is typically offline
- Online inference usually relies on precomputed embeddings

As a result, GNNs are often used **upstream of ranking**, not inside tight
real-time loops.

---

## Applied example
- [gnn_toy.ipynb](https://github.com/rosevoul/rec-ml-notes/blob/main/notebooks/gnn_toy.ipynb)

This notebook shows:
- Graph-only LightGCN
- Feature-aware GraphSAGE
- Hybrid fusion design
on the same synthetic graph.


---

## Evaluation considerations

Offline:
- Recall@K
- NDCG
- Link prediction AUC

Online:
- CTR / conversion lift
- Latency impact
- Embedding refresh cadence
