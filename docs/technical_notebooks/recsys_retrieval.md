# ANN Retrieval in Recommender Systems

## TL;DR

> Candidate retrieval is the system that decides what the ranker is even allowed to see.  
> It exists because scoring millions of items online is impossible, and it operates under stricter latency and reliability constraints than any downstream model.   
> Approximate Nearest Neighbor (ANN) search trades exactness for bounded latency, but correctness is still real: items not retrieved are silently excluded from the product.


**Notes**

This article covers **system-level retrieval design**.  
For lower-level mechanics and failure intuition, see the two notebooks:

- **Notebook 1 — Why ANN exists**  
  Brute force → approximation → recall@K as the real correctness metric.

- **Notebook 2 — Candidate width & failures**  
  How `k`, multi-strategy retrieval, and fallback behavior cap ranking quality.

---

## System Context: Where Retrieval Fits

```
All items (10^6 – 10^7)
        │
        ▼
[ Candidate Retrieval ]
(high recall, strict latency)
        │
        ▼
Candidates (10^2 – 10^3)
        │
        ▼
[ Ranking ]
(precision, business loss)
        │
        ▼
Final recommendations
```

Retrieval exists because ranking cannot operate at corpus scale under online latency budgets.
This makes retrieval a hard gate: ranking is an optimization problem only within the retrieved set.

Two consequences follow:

1. Ranking quality is upper-bounded by retrieval recall.
2. Retrieval failures are often invisible, because nothing downstream can recover missing items.

A strong ranker can reshuffle noise. It cannot invent candidates.


## Offline vs Online Architecture

```
OFFLINE                                         ONLINE
----------------------------------------------------------------
items
  │
  ▼
embeddings ──▶ ANN index (immutable, versioned) ──▶ serve index
                                                       │
query context ──▶ query embedding ──▶ ANN search ──▶ candidates
                                                       │
                                              merge / dedupe / fallback
                                                       │
                                                       ▼
                                                     ranking
```

Offline and online responsibilities must be cleanly separated.

Offline work defines the *search space*: embeddings, index structure, versioning, and validation.
Online work executes *bounded queries*: embedding inference, ANN lookup, and safe degradation.

The ANN index is not a model in the learning sense. It is a read-only data structure, closer to a materialized view than a predictor. Its correctness is established offline and assumed online.

This distinction matters for ownership, rollback strategy, and monitoring.



## What ANN Is Actually Doing 

```
Brute force:
for each item:
    compute distance(query, item)

ANN:
partition space → navigate graph / clusters → evaluate subset
```

Exact nearest-neighbor search scales linearly with corpus size and embedding dimensionality. This is infeasible online.

ANN methods approximate the neighborhood by:

* structuring the vector space
* limiting the number of distance computations
* bounding worst-case latency

The correctness question is not *“did we find the true nearest neighbor?”*
It is *“did the right items survive into the candidate set?”*

This is why retrieval is evaluated with **recall@K against a brute-force baseline**, not prediction loss.


## Brute-Force comparisons
**Exact search on a large fixed subset**
- Offline job.
- Sample 100k–1M representative items.
- Compute exact similarities within the subset.
- Use when full-corpus brute force is too expensive.

**Exact search with relaxed constraints**
- Offline job on the full corpus.
- Same embeddings and distance metric.
- No latency limits, higher memory usage.
- Typically run periodically (e.g. daily validation).

**Historical positive inclusion (weak proxy)**
- Check whether clicked or purchased items appear in retrieved top-K.
- Useful for online monitoring, not a true brute-force reference.


## Candidate Width and Search Budget

Two knobs dominate ANN behavior.

Candidate width (`k`) controls how many items survive retrieval.
Search budget (`ef`, probes, graph expansion) controls how hard the index works to find them.

If `k` is too small, the ranker is starved and downstream models saturate early.
If `k` is too large, latency rises and noise dilutes ranking signal.

Search budget trades latency predictability for recall. The correct operating point is rarely at maximum recall. It is just before the p99 latency curve bends upward.

These parameters should be tuned *for ranking outcomes*, not in isolation.



## Multi-Strategy Retrieval Is the Default, Not an Optimization

No single embedding space covers all product states.

```
                 Behavioral ANN
                        │
                        ▼
Query ──▶ Content ANN ──┼──▶ Union → Dedupe → Ranking
                        ▲
                        │
                 Popular / Recency
```

Behavioral signals dominate when history is rich.
Content signals rescue cold start and novelty.
Heuristics guarantee coverage under failure.

The goal is not purity. It is coverage under uncertainty.

Noise introduced here is acceptable. Missing candidates are not.



## Fallback Is a Product Requirement

Retrieval must always return something.

Index outages, embedding failures, and sparse contexts are normal operating conditions, not edge cases. Systems that assume otherwise ship fragile products.

Fallbacks mean degrading gracefully to safe candidate sources such as popularity, recency, or cached results, when the main retrieval path fails.

Important: fallback paths must be exercised in testing. Untested fallbacks are equivalent to no fallback.




## Evaluation: Treat Retrieval as Infrastructure

### Offline evaluation 

| Metric                          | What it measures                                    | Good (typical)                        | Bad (warning)                           |
| ------------------------------- | --------------------------------------------------- | ------------------------------------- | --------------------------------------- |
| **Recall@K (vs brute force)**   | Fraction of true top-K items that survive retrieval | ≥ 0.90–0.97 at K=100                  | < 0.80 or drops > 5–10% between builds  |
| **Segment-level coverage**      | Candidate availability across user/item segments    | ≥ 95% of segments with ≥ K candidates | Cold or tail segments < 80% coverage    |
| **Embedding drift sensitivity** | Recall stability under embedding updates            | < 2–3% recall change                  | > 5–10% recall loss from minor retrains |

Offline metrics answer: *Is this retrieval setup capable of supporting ranking at all?*
They define an upper bound, not production safety.


### Online evaluation

| Metric                        | What it measures                          | Good (typical)            | Bad (paging-worthy)                    |
| ----------------------------- | ----------------------------------------- | ------------------------- | -------------------------------------- |
| **p95 / p99 latency**         | Tail latency of retrieval calls           | p95 < 50 ms, p99 < 100 ms | p99 > 150–200 ms or unstable tails     |
| **Candidate count stability** | Number of candidates returned per request | Tight band (e.g. 300–500) | Collapses < 50 or spikes > 2× baseline |
| **Downstream ranking deltas** | Ranking metrics after retrieval changes   | ± 0–1% (explainable)      | −2–5% with no ranker changes           |

Online metrics answer: *Is retrieval safe and predictable under real traffic?*



On a final note, retrieval is not judged by AUC or loss.
It is judged by whether it reliably supplies ranking with a viable search space under load.

Many ranking regressions originate in retrieval changes that looked acceptable offline but violated online safety thresholds.
