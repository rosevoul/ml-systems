# User Embeddings (for Retrieval)

## TLDR
This component builds a user embedding from recent interactions for retrieval.  
For example, it turns a user’s last viewed items into a single query vector.  
It runs at request time or from cache and queries ANN over item embeddings.  

**Methods**
- **Recency-weighted average:** deterministic embedding from recent interactions.
- **Short-term + long-term mix:** combines stable preferences with recent intent.
- **Learned encoder:** trained model that maps interaction history to the item embedding space.
- **Fallback embedding:** safe default when history or embeddings are missing.

---

## What this component builds
- Produces one user embedding per request or cached user state.
- Uses recent item interactions as the main signal.
- Must fail open to a fallback retrieval strategy if history or embeddings are missing.

## When this component is needed
- Retrieval uses ANN over item embeddings.
- User history is sparse or noisy and needs a stable default, for example a new or infrequent user.
- Latency requires cheap request-time compute or caching.
- Retrieval must continue even if user embedding fails.

## How this component fits in the retrieval flow

```

User events → History buffer → User embedding → ANN retrieval → Candidate merge → Ranker

````

The user embedding is a query vector. 

## Inputs and Outputs

Input:
```json
{
  "user_id": 42,
  "history_item_ids": [311, 98, 712, 45, 45, 901],
  "item_embedding_version": "items_v42"
}
````

Output:

```json
{
  "user_embedding": [0.021, -0.044, ...],
  "user_embedding_version": "user_v7",
  "depends_on": "items_v42"
}
```

For this component, the output is processed into a fixed-size vector, L2-normalized, and versioned.

## How user embeddings are built

### 1) Simple baseline using recent interactions

**Method:** Recency-weighted average.
Deterministic method that produces stable embeddings, for example from the last items a user viewed.

```python
def user_embed_avg(
    history_item_ids: np.ndarray,
    item_emb: np.ndarray,
    decay: float = 0.05
) -> np.ndarray:
    L = int(history_item_ids.shape[0])
    if L == 0:
        raise ValueError("empty_history")

    t = np.arange(L, dtype=np.float32)
    w = np.exp(decay * (t - (L - 1))).astype(np.float32)
    w = w / (w.sum() + 1e-12)

    X = item_emb[history_item_ids]
    u = (X * w[:, None]).sum(axis=0)
    return l2_normalize(u)
```

### 2) Mixing long-term preferences with recent intent

**Method:** Short-term + long-term mix.
Use when both stable preferences and recent intent matter, for example a user who usually buys books but is currently browsing cameras.

```python
def user_embed_mix(
    history_item_ids: np.ndarray,
    item_emb: np.ndarray,
    alpha_short: float = 0.6
) -> np.ndarray:
    u_long = user_embed_avg(history_item_ids, item_emb, decay=0.01)
    u_short = user_embed_avg(history_item_ids, item_emb, decay=0.15)
    u = (1 - alpha_short) * u_long + alpha_short * u_short
    return l2_normalize(u)
```

### 3) Using a learned model instead of averaging

**Method:** Learned encoder.
Use when simple averaging is not expressive enough, for example when different interaction patterns should map to different intents.

```python
def make_user_features(
    history_item_ids: np.ndarray,
    item_emb: np.ndarray
) -> np.ndarray:
    X = item_emb[history_item_ids]
    mean = X.mean(axis=0)
    last = X[-1]
    short = X[-5:].mean(axis=0) if X.shape[0] >= 5 else mean
    return np.concatenate([mean, short, last], axis=0)

def user_embed_learned(
    history_item_ids: np.ndarray,
    item_emb: np.ndarray,
    encoder
) -> np.ndarray:
    feat = make_user_features(history_item_ids, item_emb)
    u = encoder(feat)
    return l2_normalize(u)
```

Output dimension must match item embeddings. The encoder must be versioned against the item embedding space.

### 4) What to do when history is empty or embeddings are unavailable

**Method:** Fallback embedding.
Retrieval must not block if user embedding fails, for example when history is empty or embeddings are unavailable.

```python
def fallback_user_embedding(d: int) -> np.ndarray:
    return l2_normalize(np.ones((d,), dtype=np.float32))

def build_user_embedding(
    user_id: int,
    history_item_ids: list[int],
    item_emb: np.ndarray
) -> np.ndarray:
    try:
        h = np.asarray(history_item_ids, dtype=np.int64)
        if h.size < 2:
            raise ValueError("short_history")
        return user_embed_mix(h, item_emb)
    except Exception:
        return fallback_user_embedding(item_emb.shape[1])
```

### 5) How caching is used in practice

**Method:** Cached user embeddings keyed by versions.
User embeddings should be cached. Cache keys must include item embedding version.

```python
def user_cache_key(
    user_id: int,
    item_emb_version: str,
    user_emb_version: str
) -> str:
    return f"uemb:{user_emb_version}:{item_emb_version}:{user_id}"
```

## What can go wrong and how to notice it

* **Embedding space mismatch:** user encoder not aligned with item embeddings, for example trained on an older item embedding version.
* **History leakage:** future or post-click signals leak into features.
* **Cache staleness:** user intent changes but cached embedding is reused too long.
* **Popularity collapse:** averaging overweights popular items.
* **Feedback loops:** embedding depends on retrieved items.


## Things to note
* This component owns user embedding correctness and stability.
* Averaging methods ignore fine-grained order.
* Learned encoders increase versioning and skew risk.
* Cold-start users still need non-embedding fallback retrieval.
