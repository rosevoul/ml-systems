# Behavioral Item Embeddings (for Retrieval)

## TLDR
This component builds item embeddings from user behavior for retrieval.  
For example, it learns which items are viewed or bought together.  
It runs offline, produces versioned embedding tables, and is used by ANN retrieval.  

**Methods**
- **Co-occurrence + SVD:** deterministic embeddings from item co-occurrence.
- **Item2Vec:** sequence-based embeddings that capture order and local context.
- **Graph embeddings:** random-walk embeddings from item graphs.
- **Behavior + content fusion:** combines behavior and content embeddings into a new version.  

---

## What this component builds
- Converts interaction sessions into fixed-size item embeddings.
- Supports multiple embedding methods with the same output format.
- Produces a single embedding table keyed by item_id and version.

## When this component is needed
- There is enough behavioral data to learn item relationships.
- Metadata alone is not enough for good retrieval.
- Retrieval quality depends on item-to-item similarity.
- Offline training jobs can run and publish new embedding versions.

## How this component fits in the retrieval flow

```

Interaction logs → Sessionize → Behavioral embedding job → Embedding store → ANN index → Retrieval
└─ optional fusion with content embeddings → writes a new embedding version

````

## What goes in and what comes out

Input:
```json
{ "session_id": "abc", "items": [12, 45, 98, 45, 311] }
````

Output:

```json
{ "item_id": 45, "embedding": [0.012, -0.081, ...], "version": "beh_v9" }
```

For this component, the output is processed into fixed-size vectors, L2-normalized, and written as a new versioned table.

## How item embeddings are built

### 1) Simple co-occurrence from sessions

**Method:** Co-occurrence + SVD.
Deterministic offline method that produces stable embeddings, for example from items frequently viewed together.

```python
def build_cooccurrence(
    sessions: list[list[int]],
    n_items: int,
    window: int = 5
) -> np.ndarray:
    C = np.zeros((n_items, n_items), dtype=np.float32)
    for seq in sessions:
        L = len(seq)
        for i, center in enumerate(seq):
            lo = max(0, i - window)
            hi = min(L, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                C[center, seq[j]] += 1.0
    return C
```

```python
def ppmi(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    total = C.sum() + eps
    row = C.sum(axis=1, keepdims=True) + eps
    col = C.sum(axis=0, keepdims=True) + eps
    pmi = np.log((C * total) / (row * col))
    return np.maximum(pmi, 0.0).astype(np.float32)
```

```python
def svd_embed(PPMI: np.ndarray, k: int = 128) -> np.ndarray:
    U, S, _ = np.linalg.svd(PPMI, full_matrices=False)
    E = U[:, :k] * np.sqrt(S[:k])[None, :]
    return l2_normalize(E.astype(np.float32))
```

### 2) Learning from item order in sessions

**Method:** Item2Vec.
Use when order and local context matter, for example when users view a phone and then a phone case in the same session.

```python
def make_skipgram_pairs(
    sessions: list[list[int]],
    window: int = 5
) -> np.ndarray:
    pairs = []
    for seq in sessions:
        for i, center in enumerate(seq):
            lo = max(0, i - window)
            hi = min(len(seq), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, seq[j]))
    return np.asarray(pairs, dtype=np.int64)
```

```python
def train_item2vec(
    pairs: np.ndarray,
    n_items: int,
    dim: int = 128,
    neg_k: int = 10,
    steps: int = 50_000
) -> np.ndarray:
    rng = np.random.default_rng(0)
    W_in = 0.01 * rng.normal(size=(n_items, dim)).astype(np.float32)
    W_out = 0.01 * rng.normal(size=(n_items, dim)).astype(np.float32)

    centers = pairs[:, 0]
    contexts = pairs[:, 1]

    for _ in range(steps):
        idx = rng.integers(0, len(pairs))
        c = int(centers[idx])
        o = int(contexts[idx])

        neg = rng.integers(0, n_items, size=neg_k)

        vc = W_in[c]
        vo = W_out[o]

        score_pos = 1 / (1 + np.exp(-np.dot(vc, vo)))
        grad = score_pos - 1.0

        W_in[c] -= 0.05 * grad * vo
        W_out[o] -= 0.05 * grad * vc

        for n in neg:
            vn = W_out[int(n)]
            score_neg = 1 / (1 + np.exp(-np.dot(vc, vn)))
            gradn = score_neg
            W_in[c] -= 0.05 * gradn * vn
            W_out[int(n)] -= 0.05 * gradn * vc

    return l2_normalize(W_in)
```

### 3) Learning from item graphs

**Method:** Graph embeddings.
Use when item relationships are better represented as a graph, for example co-purchase or co-cart data.

```python
def build_graph(
    C: np.ndarray,
    topk: int = 50
) -> list[list[int]]:
    neighbors = []
    for i in range(C.shape[0]):
        idx = np.argsort(-C[i])[:topk]
        neighbors.append([int(j) for j in idx if C[i, j] > 0])
    return neighbors
```

```python
def random_walks(
    neighbors: list[list[int]],
    n_walks: int = 10,
    walk_len: int = 20,
    seed: int = 0
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    walks = []
    for start in range(len(neighbors)):
        for _ in range(n_walks):
            cur = start
            seq = [cur]
            for _ in range(walk_len - 1):
                if not neighbors[cur]:
                    break
                cur = int(rng.choice(neighbors[cur]))
                seq.append(cur)
            walks.append(seq)
    return walks
```

### 4) Combining behavior with content embeddings

**Method:** Behavior + content fusion.
Use to handle cold-start items, for example new items with no interaction history.

```python
def fuse_behavior_content(
    behavior_embeddings: np.ndarray,
    content_embeddings: np.ndarray,
    behavior_weight: float = 0.7
) -> np.ndarray:
    if behavior_embeddings.shape != content_embeddings.shape:
        raise ValueError("embedding shape mismatch")
    fused = (
        behavior_weight * behavior_embeddings
        + (1.0 - behavior_weight) * content_embeddings
    )
    return l2_normalize(fused.astype(np.float32))
```

## What can go wrong and how to notice it

* **Popularity domination:** embeddings collapse toward popular items.
* **Sessionization errors:** cross-user leakage creates false co-occurrence.
* **Cold-start gaps:** new items have no behavior signal.
* **Index skew:** embeddings updated without rebuilding ANN.
* **Silent regressions:** vectors look valid but neighbors change unexpectedly.


## Things to note
* This component owns item embedding quality and stability.
* Exposure bias from historical data.
* Fast distribution shifts without retraining.
* Fusion weights are heuristic unless learned.
