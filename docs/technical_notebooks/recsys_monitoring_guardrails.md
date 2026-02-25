# Recommender Monitoring: Metrics That Matter with Guardrails
**Scope:** Concrete production metrics, how to compute them, and how to interpret them jointly across retrieval → ranking → serving.

## Metrics (technical + business + what good/neutral/bad means + interactions)

### Latency (p50 / p95 / p99) by stage
- **Technical:** Response-time distribution for retrieval, ranking, post-processing, and end-to-end.
- **Business:** Tail latency drives abandonment and perception of “brokenness,” even when averages look fine.
- **Good:** p95/p99 stable within SLO; stage attribution consistent.
- **Neutral:** Small p50 shifts with unchanged p95/p99.
- **Bad:** p99 spikes, or stage shift (e.g., retrieval p95 jumps) without code changes elsewhere.
- **Interacts with:** ANN params (`k`, `ef`), candidate pool size, feature-store latency, LLM calls, cache hit rate.

```python
# Prometheus-style histograms assumed; compute quantiles from histogram buckets.
from prometheus_client import Histogram

latency = Histogram(
    "recsys_stage_latency_ms",
    "Stage latency",
    ["stage", "version"],
    buckets=(5,10,20,40,60,80,120,200,400,800,1600)
)

# On request completion:
latency.labels(stage="retrieval", version=ver).observe(retrieval_ms)
latency.labels(stage="ranking", version=ver).observe(ranking_ms)
latency.labels(stage="e2e", version=ver).observe(total_ms)
```

### Empty result rate
- **Technical:** Fraction of requests where retrieval returns zero (or below a minimum) candidates after merge/dedupe.
- **Business:** Users see blanks, repetitive content, or irrelevant fillers.
- **Good:** Near-zero and stable by slice (region, device, cohort).
- **Neutral:** Brief blips during deploy or index rebuild windows.
- **Bad:** Sustained increase, or spikes concentrated in a segment (often a data/feature join issue).
- **Interacts with:** fallback rate, index version mismatch, upstream query parsing, filtering/policy removal.

```python
def emit_empty_result_rate(num_requests, num_empty, tags):
    rate = num_empty / max(num_requests, 1)
    emit_gauge("recsys.empty_result_rate", rate, tags)

# guardrail
if empty_rate_5m > 0.01:
    alert("empty results spike", tags={"stage": "retrieval"})
```

### Fallback rate
- **Technical:** Fraction of requests routed to degraded mode (cached vectors, popularity, lexical retrieval, etc.).
- **Business:** Quality dilution that can look “fine” short-term while long-term metrics decay.
- **Good:** Low and expected (only for known cold-start or timeouts).
- **Neutral:** Controlled increases during A/B tests that explicitly exercise fallbacks.
- **Bad:** Spikes after deploys; sustained elevation indicates hidden instability.
- **Interacts with:** latency (timeouts), error rate, feature availability, cache health.

```python
def choose_path(ctx):
    if ctx.retrieval_timeout or ctx.index_unavailable:
        emit_counter("recsys.fallback", 1, {"reason": "retrieval_unavailable"})
        return "fallback"
    return "primary"

fallback_rate = fallbacks_5m / max(requests_5m, 1)
if fallback_rate > 0.15:
    alert("fallback rate high", tags={"service": "recsys"})
```

### Candidate pool size (post-merge, pre-ranking)
- **Technical:** Number of unique candidates reaching ranking after union/merge/dedupe.
- **Business:** Too small reduces choice; too large inflates cost and can hurt latency and model stability.
- **Good:** Stable band (e.g., 200–800) with predictable slice behavior.
- **Neutral:** Minor shifts explained by traffic mix or strategy weights.
- **Bad:** Spikes (cost/latency) or collapse (quality/coverage).
- **Interacts with:** latency, recall@K, ANN tuning, query expansion, filtering.

```python
emit_histogram("recsys.candidate_pool_size", pool_size, {"version": ver})

# guardrail bands (tune per surface)
if pool_size_p95 < 100:
    alert("candidate pool collapse", {"surface": surface, "version": ver})
if pool_size_p95 > 1500:
    alert("candidate pool spike", {"surface": surface, "version": ver})
```

### Retrieval recall@K (offline) + slice stability
- **Technical:** Probability that future positives appear in the retrieved candidate set.
- **Business:** Upper bound on achievable online lift; if recall drops, ranking cannot compensate.
- **Good:** Stable overall and across key slices (new users, long-tail items).
- **Neutral:** Small variance within historical noise.
- **Bad:** Drop concentrated in cold-start or tail; often precedes online regression.
- **Interacts with:** embedding drift, index build bugs, ANN params, candidate pool size.

```python
import numpy as np

def recall_at_k(retrieved_ids, positive_ids, k):
    topk = set(retrieved_ids[:k])
    pos = set(positive_ids)
    return 1.0 if len(topk & pos) > 0 else 0.0

# aggregate by slice
recalls = [recall_at_k(r, p, 200) for r, p in eval_rows]
emit_gauge("offline.retrieval_recall_at_200", float(np.mean(recalls)), {"version": ver})
```

### Ranking quality metric (NDCG@K) + calibration proxy
- **Technical:** Ordering quality over the candidate set; plus a sanity check on score distribution.
- **Business:** Should directionally track CTR/CVR in controlled experiments; calibration drift can break downstream blending.
- **Good:** NDCG stable; score distribution stable (no saturation/collapse).
- **Neutral:** Flat NDCG with stable online metrics under traffic mix shifts.
- **Bad:** Offline NDCG improves while online drops; score distribution compresses or saturates.
- **Interacts with:** label leakage, feature availability, retrieval recall, exploration policies.

```python
def ndcg_at_k(gains, k):
    # gains aligned to ranked list; binary or graded relevance
    gains = np.asarray(gains)[:k]
    denom = np.log2(np.arange(2, gains.size + 2))
    dcg = (gains / denom).sum()
    ideal = np.sort(gains)[::-1]
    idcg = (ideal / denom).sum() if ideal.sum() > 0 else 1.0
    return float(dcg / idcg)

# score distribution monitor (serving)
emit_histogram("recsys.rank_score", score, {"version": ver})
```

### Online business metric (CTR / CVR) with guardrails
- **Technical:** Clicks or conversions per impression, by surface and position band.
- **Business:** Primary outcome; also the easiest to misread without controls (position bias, UI changes).
- **Good:** Lift in properly randomized experiments; stable by position band.
- **Neutral:** Flat during infra-only changes or under seasonal effects that move both control/treatment.
- **Bad:** Drops in treatment with stable infra metrics; divergence from offline changes.
- **Interacts with:** ranking metric, latency, empty rate, UI experiments, traffic allocation.

```python
# Minimal, practical: log impressions with version tags; aggregate later.
log_impression(
    user_id=u,
    item_ids=shown_ids,
    positions=list(range(len(shown_ids))),
    model_version=ver,
    index_version=index_ver,
)

# Guardrail on live CTR delta vs control (pseudo):
if ctr_treatment - ctr_control < -0.005 and latency_p95_ok and empty_rate_ok:
    alert("likely relevance regression", {"surface": surface, "version": ver})
```

### Feature availability rate (request-time)
- **Technical:** Fraction of requests where required features are present and non-default.
- **Business:** Missing features silently degrade quality and make experiments non-actionable.
- **Good:** Near 100% for required features; missingness explained for optional features.
- **Neutral:** Minor drops in optional features with no score distribution shift.
- **Bad:** Drops correlated with score collapse or ranking metric decay.
- **Interacts with:** fallback rate, score distribution, ranking metric, latency (feature store timeouts).


```python
REQUIRED = ["user_tenure_days", "item_category", "price", "recent_views_7d"]

def feature_availability(features):
    present = sum(1 for k in REQUIRED if k in features and features[k] is not None)
    return present / len(REQUIRED)

avail = feature_availability(feature_dict)
emit_histogram("recsys.feature_availability", avail, {"version": ver})

if avail_p95 < 0.9:
    alert("feature missingness high", {"version": ver})
