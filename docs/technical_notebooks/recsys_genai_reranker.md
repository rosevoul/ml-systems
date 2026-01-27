# GenAI Reranker (Post-Ranking)
**Scope:** Applies an LLM as a constrained semantic judge to adjust the ordering of an already-scored candidate list. Designed as an optional refinement layer with bounded influence and predictable rollback behavior.

## When this component is used
- Small candidate sets (≤50) where ordering errors are user-visible.
- Queries with semantic nuance or constraints underrepresented in learned features.
- Surfaces with explicit latency and cost budgets.
- As a refinement step that must be safe to bypass at any time.

## Integration points

```
Candidates + scores
        ↓
  GenAI reranker
        ↓
 Final ordering
        ↓
 UI / policy
```

Consumes ranked candidates and query context. Emits a reordered list only. Does not add, remove, or filter items.

## Example input

```json
{
  "query": "ergonomic office chair",
  "candidates": [
    {"item_id": 712, "title": "Mesh Office Chair", "score": 1.82},
    {"item_id": 45,  "title": "Executive Leather Chair", "score": 1.75},
    {"item_id": 98,  "title": "Drafting Chair", "score": 1.61}
  ]
}
```

## Core implementation

### Canonical prompt (judge-style)

The LLM is framed explicitly as a ranking function. It is not asked to explain or justify decisions.

```python
SYSTEM_PROMPT = """
You are a ranking function.
You output only item_ids in ranked order.
No explanations.
"""

def build_prompt(query, candidates):
    items = [
        {"id": c["item_id"], "title": c["title"]}
        for c in candidates
    ]
    return {
        "system": SYSTEM_PROMPT,
        "user": {
            "query": query,
            "items": items
        }
    }
```

Design notes:
- The system prompt constrains behavior and reduces variance.
- Only titles are provided to avoid leaking price, popularity, or model scores.
- The primary ranker remains the source of truth for relevance signals.

### Deterministic generation with schema enforcement

```python
def llm_rank(prompt):
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.0,
        max_output_tokens=50,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ranking",
                "schema": {
                    "type": "object",
                    "properties": {
                        "order": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["order"]
                }
            }
        }
    )
    return resp.output_parsed["order"]
```

Design notes:
- Zero temperature ensures repeatability.
- Hard schema validation prevents malformed outputs from propagating.
- Token limits cap latency and cost.

### Validation and fail-open behavior

```python
def validate(order, candidates):
    ids = {c["item_id"] for c in candidates}
    return (
        len(order) == len(candidates) and
        set(order) == ids
    )
```

Invalid or non-compliant outputs are logged and ignored. The system falls back to the original ranking without user-visible impact.

### Bounded influence via blending

The LLM signal is intentionally weak and additive.

```python
def blend(primary_rank, llm_rank, alpha=0.2):
    score = {}
    for i, item in enumerate(primary_rank):
        score[item] = 1.0 - alpha * i
    for i, item in enumerate(llm_rank):
        score[item] = score.get(item, 0) + alpha * (1.0 - i)
    return sorted(score, key=score.get, reverse=True)
```

Design notes:
- The primary ranker dominates ordering.
- Alpha is tuned offline and fixed per surface.
- This limits blast radius from prompt or model drift.

## Output

```json
{
  "final_rank": [45, 712, 98]
}
```

Passed directly to downstream policy checks or UI rendering.

## Metrics

### Swap rate

```python
def swap_rate(before, after):
    swaps = sum(1 for i in range(len(before)) if before[i] != after[i])
    return swaps / len(before)
```

- High swap rates indicate over-intervention.
- Used as a guardrail, not an optimization target.

### Incremental lift (experiment-only)

```python
delta_ctr = ctr_llm - ctr_baseline
if delta_ctr < 0:
    disable("llm_reranker")
```

- No sustained lift justifies removal.
- The default state is off.

### Latency and timeout rate

```python
if llm_latency_p95 > 150:
    bypass("llm_reranker")
```

- The reranker must never block the request path.
- Timeouts always fail open.

## Guardrails
- Strict schema validation or no-op.
- Hard candidate count limits.
- Explicit bypass on timeout or error.
- Per-surface allowlisting.

## Known limitations
- Does not improve recall.
- Limited interpretability of individual swaps.
- Sensitive to prompt edits.
- Cost scales linearly with QPS.
