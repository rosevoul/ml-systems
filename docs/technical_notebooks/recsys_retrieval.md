# Candidate Retrieval (ANN)

## TLDR
This component retrieves a high-recall candidate set using ANN search. 
For example, it finds the 200 nearest items to a query or user vector. 
It runs on the request path and feeds ranking. 

**Methods**
- **ANN index build (HNSW):** offline index build for a specific embedding version.
- **ANN lookup:** request-time nearest neighbor lookup.
- **Fallback retrieval:** bounded default when ANN fails.
- **Multi-strategy merge:** union semantic and user-profile retrieval results.

---

## What this component builds
- Takes a query or user embedding and returns candidate item_ids plus similarity.
- Optimizes for recall under strict latency and memory constraints.
- Must fail open with a bounded fallback.

## When this component is needed
- Catalog is too large for brute-force scoring (10^6+ items).
- Retrieval is on the request path with tight p95/p99 budgets.
- Multiple retrieval strategies are unioned downstream (semantic + lexical + rules).
- Embeddings and indices are updated regularly and need clean rollback.

## How this component fits in the retrieval flow

```
Item embeddings (offline)
   ↓
ANN index build + publish (versioned)
   ↓
Serving: query/user embedding
   ↓
ANN lookup (k)
   ↓
Candidates (+ similarity)
   ↓
Merge/dedupe + post-filters
   ↓
Ranker
```

Retrieval emits candidates. It does not decide what is eligible or safe.

## What goes in and what comes out

Input (request-time):
```json
{
  "embedding": [0.02, -0.11, ...],
  "k": 200,
  "index_version": "items_v42"
}
```

Output:
```json
{
  "candidates": [
    {"item_id": 712, "sim": 0.83},
    {"item_id": 45,  "sim": 0.81}
  ],
  "index_version": "items_v42"
}
```

## How candidate retrieval works

### 1) Checking versions and compatibility
**Method:** Versioned artifacts and compatibility checks. 
Use to prevent querying the wrong index, for example when dimensions or versions mismatch.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class IndexMeta:
    index_version: str
    embedding_version: str
    dim: int
    space: str  # "cosine" or "l2"

def assert_compatible(meta: IndexMeta, query_vec):
    if meta.dim != int(query_vec.shape[-1]):
        raise ValueError(f"dim mismatch: index={meta.dim} query={query_vec.shape[-1]}")
```

### 2) Building the ANN index offline
**Method:** ANN index build (HNSW). 
Offline, deterministic index build for a specific embedding version.

```python
import hnswlib
import numpy as np

def build_hnsw_index(item_ids: np.ndarray,
                     item_vecs: np.ndarray,
                     meta: IndexMeta,
                     M: int = 16,
                     ef_construction: int = 200):
    index = hnswlib.Index(space=meta.space, dim=meta.dim)
    index.init_index(max_elements=len(item_ids), ef_construction=ef_construction, M=M)
    index.add_items(item_vecs, item_ids)
    return index
```

### 3) Publishing and rolling back safely
**Method:** Publish + rollback via pointer switch. 
Use to treat indices as immutable and roll back by switching versions.

```python
def publish_index(index, meta: IndexMeta):
    path = f"/indices/{meta.index_version}/index.bin"
    index.save_index(path)
    write_json(f"/indices/{meta.index_version}/meta.json", meta.__dict__)
    # Production: atomic pointer update (e.g., service discovery, config store)
```

### 4) Serving configuration
**Method:** Retrieval config (k, ef_runtime, timeouts). 
Use to control latency and recall without changing build settings.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 200
    ef_runtime: int = 100
    min_candidates: int = 50
    timeout_ms: int = 30  # retrieval must be fast
    hard_fail_open: bool = True

RCFG = RetrievalConfig()
```

### 5) Doing ANN lookup at request time
**Method:** ANN lookup. 
Use to retrieve candidates under tight latency budgets.

```python
def ann_lookup(index, meta: IndexMeta, query_vec, cfg: RetrievalConfig = RCFG):
    assert_compatible(meta, query_vec)
    index.set_ef(cfg.ef_runtime)

    ids, sims = index.knn_query(query_vec, k=cfg.k)
    ids = ids[0].tolist()
    sims = sims[0].tolist()

    if len(ids) == 0 or len(ids) < cfg.min_candidates:
        raise RuntimeError("too_few_candidates")

    return ids, sims
```

### 6) What to do when ANN lookup fails
**Method:** Fallback retrieval. 
Use when ANN errors or returns too few candidates.

```python
def fallback_candidates(limit: int = 200) -> list[int]:
    return popular_items(limit)

def retrieve_candidates(index, meta, query_vec, cfg: RetrievalConfig = RCFG) -> list[int]:
    try:
        ids, _sims = ann_lookup(index, meta, query_vec, cfg=cfg)
        return ids
    except Exception as e:
        log.warning("retrieval_fallback", extra={"err": str(e), "index_version": meta.index_version})
        return fallback_candidates(limit=cfg.k)
```

### 7) Combining multiple retrieval strategies
**Method:** Multi-strategy merge. 
Use when you union semantic and user-profile retrieval before ranking.

```python
def dedupe_keep_order(ids: list[int]) -> list[int]:
    seen = set()
    out = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def retrieve_multi(query_vec, user_vec, index_map, meta_map, cfg: RetrievalConfig = RCFG) -> list[int]:
    all_ids = []

    all_ids.extend(retrieve_candidates(index_map["query"], meta_map["query"], query_vec, cfg=cfg))

    if user_vec is not None:
        all_ids.extend(retrieve_candidates(index_map["user"], meta_map["user"], user_vec, cfg=cfg))

    return dedupe_keep_order(all_ids)[: cfg.k]
```

## What can go wrong and how to notice it
- **Index/version mismatch:** returns plausible but wrong neighbors. enforce meta checks.
- **Silent recall drift:** traffic mix changes. ANN recall degrades without errors.
- **Latency creep:** ef_runtime tuned too high. p99 blows up under load.
- **Partial index build:** missing shards/items. quality collapses in specific segments.
- **Over-reliance on similarity:** downstream treats sims as relevance. keep contracts explicit.


## Things to note
- This component owns candidate recall and retrieval stability.
- Approximate by design. exact nearest neighbors are not guaranteed.
- Requires workload-specific tuning (k, ef, M).
- Debugging is harder than brute-force baselines. keep a brute-force evaluator offline.
