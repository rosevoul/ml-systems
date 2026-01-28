# Candidate Ranking Model

## TLDR
This component scores and orders retrieved candidates. 
For example, it sorts 200 retrieved items into a final list. 
It runs after retrieval and owns relevance ordering. 

**Methods**
- **Feature join:** user × item × context features.
- **Feature availability checks:** detect missing features early.
- **Scoring model:** predicts relative relevance for ordering.
- **Deterministic sorting:** stable ordering and tie-breaking.
- **Fail-soft ranking:** bounded behavior under feature outages.

---

## What this component builds
- Takes candidate item_ids and request context, joins features, scores candidates, and sorts.
- Optimizes ordering, not calibration. Scores are for ranking.
- Must degrade safely when features are missing.

## When this component is needed
- Candidates are already limited (typically 200–2000).
- Feature store / join layer is available at request time.
- This is the main experimentation surface (A/B tests).
- The system needs predictable fallbacks under partial feature outages.

## How this component fits in the retrieval flow

```
Candidates from retrieval
   ↓
Feature join (user × item × context)
   ↓
Model inference (score)
   ↓
Sort + tie-breaking
   ↓
Post-processing (policy/diversity/UI)
```

Ranking emits an ordered list plus per-item scores and diagnostics for monitoring/debug.

## Inputs & Outputs

Input (one request):
```json
{
  "user_id": 42,
  "candidates": [712, 45, 98],
  "context": {"surface": "home", "locale": "en_US"}
}
```

Output:
```json
{
  "ranked": [
    {"item_id": 712, "score": 1.82},
    {"item_id": 45,  "score": 1.75},
    {"item_id": 98,  "score": 1.61}
  ],
  "model_version": "rank_v17"
}
```

## How ranking works

### 1) What features are required
**Method:** Feature availability checks. 
Use to detect missing features early, for example during feature store outages.

```python
from dataclasses import dataclass
from typing import Any

REQUIRED_FEATURES = [
    "user_tenure_days",
    "item_price",
    "item_category_id",
    "user_recent_views_7d",
]

def feature_availability(feat: dict[str, Any]) -> float:
    present = sum(1 for k in REQUIRED_FEATURES if feat.get(k) is not None)
    return present / len(REQUIRED_FEATURES)
```

### 2) Joining user, item, and context features
**Method:** Feature join. 
Combines user, item, and context features for each candidate.

```python
def build_feature_row(user_id: int, item_id: int, ctx: dict) -> dict:
    uf = user_store.get(user_id)                 # stable user features
    itf = item_store.get(item_id)                # stable item features
    intf = interaction_store.get(user_id, item_id, window_days=7)

    return {
        "user_tenure_days": uf.get("tenure_days"),
        "user_recent_views_7d": intf.get("views_7d"),
        "item_price": itf.get("price"),
        "item_category_id": itf.get("category_id"),
        "surface": ctx.get("surface"),
        "locale": ctx.get("locale"),
    }
```

### 3) Building the batch for scoring
**Method:** Batch feature assembly with explicit defaults. 
Use deterministic encoding and explicit defaults.

```python
import numpy as np

def featurize(user_id: int, candidates: list[int], ctx: dict) -> tuple[np.ndarray, list[float]]:
    rows = []
    avails = []
    for item_id in candidates:
        row = build_feature_row(user_id, item_id, ctx)
        avails.append(feature_availability(row))

        row.setdefault("user_recent_views_7d", 0)
        rows.append(row)

    X = feature_encoder.transform(rows)  # deterministic encoder used in training
    return X, avails
```

### 4) Scoring candidates
**Method:** Scoring model. 
Scores are for ordering. Absolute values can shift across versions.

```python
@dataclass(frozen=True)
class RankConfig:
    min_feature_availability: float = 0.75
    fallback_mode: str = "score_then_popularity"  # bounded degrade
    model_version: str = "rank_v17"

CFG = RankConfig()

def score_batch(X: np.ndarray) -> np.ndarray:
    return model.predict(X)  # shape: [n]
```

### 5) Sorting and tie-breaking
**Method:** Deterministic sorting. 
Use stable secondary keys to avoid churn.

```python
def rank_candidates(candidates: list[int], scores: np.ndarray, ctx: dict) -> list[int]:
    pop = popularity_store.get_many(candidates)

    keyed = [
        (float(scores[i]), float(pop[i]), candidates[i])
        for i in range(len(candidates))
    ]
    keyed.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [t[2] for t in keyed]
```

### 6) What to do when features are missing
**Method:** Fail-soft ranking. 
Degrade in a controlled way instead of guessing silently.

```python
def rank_request(user_id: int, candidates: list[int], ctx: dict, cfg: RankConfig = CFG):
    X, avails = featurize(user_id, candidates, ctx)
    avail_p95 = float(np.percentile(avails, 5))

    if avail_p95 < cfg.min_feature_availability:
        log.warning("rank_feature_gap", extra={"avail_p95": avail_p95, "model_version": cfg.model_version})

        if cfg.fallback_mode == "score_then_popularity":
            pass
        else:
            return fallback_rank(candidates), {"mode": "fallback"}

    scores = score_batch(X)
    ranked = rank_candidates(candidates, scores, ctx)
    return ranked, {"mode": "primary", "avail_p95": avail_p95}
```

### 7) Training objective contract (pairwise)
**Method:** Pairwise ordering loss contract. 
Use to reinforce that training optimizes ordering, not score calibration.

```python
def pairwise_logistic(s_pos: np.ndarray, s_neg: np.ndarray) -> float:
    return float(np.mean(np.log1p(np.exp(-(s_pos - s_neg)))))
```

## What can go wrong and how to notice it
- **Training-serving skew:** encoder mismatch or feature definition drift breaks relevance silently.
- **Label leakage:** future behavior sneaks into features or labels. offline looks great, online dies.
- **Score saturation:** model outputs collapse into narrow range. ranking becomes mostly tie-breaker.
- **Feature gaps:** missingness defaults hide real issues. treat availability as a signal.
- **Churn:** unstable tie-breaking causes visible reshuffles and noisy experiments.


## Things to note
- This component owns ranking correctness and stability.
- Dependent on retrieval quality and candidate diversity.
- Requires bias correction in training data to generalize.
