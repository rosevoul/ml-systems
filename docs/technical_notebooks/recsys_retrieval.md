# Candidate Retrieval (ANN)
**Scope:** Retrieves a high-recall candidate set from a large item corpus using approximate nearest neighbor (ANN) search. Feeds ranking. Does not apply business rules, eligibility, or final ordering.

## What this component does (and does not do)
- Takes a query/user embedding and returns candidate item_ids (+ similarity).
- Optimizes for recall under strict latency and memory constraints.
- Does not calibrate relevance. Similarity is not a rank score.
- Does not gate serving; must fail open with a bounded fallback.

## When this component is used
- Catalog is too large for brute-force scoring (10^6+ items).
- Retrieval is on the request path with tight p95/p99 budgets.
- Multiple retrieval strategies are unioned downstream (semantic + lexical + rules).
- Embeddings and indices are updated regularly and need clean rollback.

## Integration points

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

Retrieval emits candidates. It does not decide what is “eligible” or “safe”. That happens downstream.

## Example input / output

Input (request-time):
```json
{
  "embedding": {"dim": 48, "norm": 1.0},
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

## Core implementation (handoff-grade)

### 1) Versioned artifacts and compatibility checks
The most common retrieval failure is “everything works, but it’s the wrong index.”

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

### 2) Offline build (HNSW example)
Build is expensive and should be deterministic and reproducible.

```python
import hnswlib
import numpy as np

def build_hnsw_index(item_ids: np.ndarray,
                     item_vecs: np.ndarray,
                     meta: IndexMeta,
                     *,
                     M: int = 16,
                     ef_construction: int = 200):
    index = hnswlib.Index(space=meta.space, dim=meta.dim)
    index.init_index(max_elements=len(item_ids), ef_construction=ef_construction, M=M)
    index.add_items(item_vecs, item_ids)
    return index
```

Key decisions encoded:
- `M` and `ef_construction` define memory and build time.
- Index is built for a specific embedding space and version.

### 3) Publish + rollback semantics
Index is immutable after publish. Rollback is a pointer switch.

```python
def publish_index(index, meta: IndexMeta):
    path = f"/indices/{meta.index_version}/index.bin"
    index.save_index(path)
    write_json(f"/indices/{meta.index_version}/meta.json", meta.__dict__)
    # Production: atomic pointer update (e.g., service discovery, config store)
```

### 4) Load and serve with explicit configuration
Serving config is separate from build config.

```python
@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 200
    ef_runtime: int = 100
    min_candidates: int = 50
    timeout_ms: int = 30  # retrieval must be fast
    hard_fail_open: bool = True

RCFG = RetrievalConfig()
```

### 5) Query-time lookup with bounded behavior
ANN is approximate; treat it as best-effort and validate outputs.

```python
def ann_lookup(index, meta: IndexMeta, query_vec, *, cfg: RetrievalConfig = RCFG):
    assert_compatible(meta, query_vec)
    index.set_ef(cfg.ef_runtime)

    ids, sims = index.knn_query(query_vec, k=cfg.k)
    ids = ids[0].tolist()
    sims = sims[0].tolist()

    # Basic sanity checks (silent failure protection)
    if len(ids) == 0 or len(ids) < cfg.min_candidates:
        raise RuntimeError("too_few_candidates")

    return ids, sims
```

### 6) Fail-open fallback
Fallback should be bounded and predictable. Never return “nothing” unless that is explicitly allowed.

```python
def fallback_candidates(*, limit: int = 200) -> list[int]:
    # Example: popularity or cached per-surface shortlist
    return popular_items(limit)

def retrieve_candidates(index, meta, query_vec, *, cfg: RetrievalConfig = RCFG) -> list[int]:
    try:
        ids, _sims = ann_lookup(index, meta, query_vec, cfg=cfg)
        return ids
    except Exception as e:
        log.warning("retrieval_fallback", extra={"err": str(e), "index_version": meta.index_version})
        return fallback_candidates(limit=cfg.k)
```

### 7) Wiring: multi-strategy retrieval + merge
This is typically where systems drift. Make merge behavior explicit.

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

def retrieve_multi(query_vec, user_vec, index_map, meta_map, *, cfg: RetrievalConfig = RCFG) -> list[int]:
    all_ids = []

    # semantic query retrieval
    all_ids.extend(retrieve_candidates(index_map["query"], meta_map["query"], query_vec, cfg=cfg))

    # user-profile retrieval (optional)
    if user_vec is not None:
        all_ids.extend(retrieve_candidates(index_map["user"], meta_map["user"], user_vec, cfg=cfg))

    return dedupe_keep_order(all_ids)[: cfg.k]
```

## Guardrails and failure modes (the ones that matter)
- **Index/version mismatch:** returns plausible but wrong neighbors; enforce meta checks.
- **Silent recall drift:** traffic mix changes; ANN recall degrades without errors.
- **Latency creep:** `ef_runtime` tuned too high; p99 blows up under load.
- **Partial index build:** missing shards/items; quality collapses in specific segments.
- **Over-reliance on similarity:** downstream treats sims as relevance; keep contracts explicit.

## Known limitations
- Approximate by design; exact nearest neighbors are not guaranteed.
- Requires workload-specific tuning (k, ef, M).
- Debugging is harder than brute-force baselines; keep a brute-force evaluator offline.
