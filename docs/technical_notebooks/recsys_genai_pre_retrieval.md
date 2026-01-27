# GenAI Query Expansion (Pre-Retrieval)
**Scope:** Expands a raw text query into a small set of deterministic rewrites before embedding + retrieval. Used to widen recall upstream of ANN. Optional and fail-open.

## What this component does (and does not do)
- Produces 1–3 query variants intended for the existing embedding + retrieval stack.
- Keeps the original query as the anchor. Always.
- Adds no new retrieval logic. It only changes the text fed into embedding.
- Does not rank, filter, apply inventory constraints, or enforce policy.

## When this component is used
- Query is short, generic, or ambiguous (e.g., 1–2 tokens).
- No strong user profile / behavior signal exists for the request.
- Retrieval recall is the bottleneck, not ranking.
- The surface can tolerate extra latency, or expansions can be cached.

## Integration points

```
Raw query
   ↓
LLM expansion (optional) ──┐
   ↓                      │
Query normalization        │
   ↓                      │
Embed each query variant   │
   ↓                      │
ANN retrieval per variant  │
   ↓                      │
Merge + dedupe candidates ─┘
   ↓
Ranker
```

The rest of the pipeline stays unchanged. If this component is disabled, the original query flows through as-is.

## Example input / output

Input:
```json
{ "query": "office chair", "locale": "en_US", "surface": "search" }
```

Output (what the next stage consumes):
```json
{
  "queries": [
    "office chair",
    "ergonomic office chair",
    "adjustable desk chair lumbar support"
  ],
  "expansion_version": "qe_v3"
}
```

## Core implementation (handoff-grade)

### 1) Query normalization (don’t feed garbage to the LLM)
This protects cost and makes caching actually work.

```python
import re
from dataclasses import dataclass

_ws = re.compile(r"\s+")

def normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = _ws.sub(" ", q)
    # keep punctuation minimal; avoid removing meaning (e.g., "c++", "4k")
    return q[:256]  # hard cap; prevents prompt bloat
```

### 2) Request contract + versioning
Version everything because prompts drift.

```python
@dataclass(frozen=True)
class QEConfig:
    expansion_version: str = "qe_v3"
    max_variants: int = 3
    max_tokens_out: int = 80
    temperature: float = 0.0
    timeout_ms: int = 120
    model: str = "gpt-4.1-mini"

CFG = QEConfig()
```

### 3) Prompt: judge-style rewriting, no explanations
Keep it boring. Boring is reproducible.

```python
SYSTEM_PROMPT = """
You rewrite search queries.
Return 2 concise alternatives to the input query.
Rules:
- No brands.
- No explanations.
- Output one query per line.
- Keep each line under 10 words.
"""
```

### 4) LLM call with strict output hygiene
Treat the LLM like an unreliable dependency.

```python
def llm_expand(raw_query: str, *, cfg: QEConfig = CFG) -> list[str]:
    resp = client.responses.create(
        model=cfg.model,
        input={"system": SYSTEM_PROMPT, "user": raw_query},
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_tokens_out,
        # In production: enforce request timeout at the client/transport layer.
    )
    text = (resp.output_text or "").strip()
    if not text:
        return []
    lines = [normalize_query(x) for x in text.splitlines() if x.strip()]
    return lines
```

### 5) Post-processing: dedupe, cap, preserve anchor
This is the real logic that keeps behavior stable across time.

```python
def dedupe_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def build_variants(query: str, *, cfg: QEConfig = CFG) -> list[str]:
    q0 = normalize_query(query)
    # Anchor always first
    variants = [q0]

    # Expansion is additive
    expansions = llm_expand(q0, cfg=cfg)
    variants.extend(expansions)

    variants = dedupe_keep_order(variants)

    # Hard cap to bound cost downstream (embed + ANN + ranking)
    variants = variants[:cfg.max_variants]

    # Invariant: anchor must exist
    assert variants and variants[0] == q0, "Anchor query must be preserved"
    return variants
```

### 6) Fail-open behavior with explicit bypass
On failure, you get baseline retrieval, not a 500.

```python
def safe_build_variants(query: str, *, cfg: QEConfig = CFG) -> list[str]:
    q0 = normalize_query(query)
    try:
        return build_variants(q0, cfg=cfg)
    except Exception as e:
        log.warning("qe_bypass", extra={"err": str(e), "expansion_version": cfg.expansion_version})
        return [q0]
```

### 7) Caching: make the expensive part optional
If you don’t cache, this is not a real production component.

```python
def cache_key(query: str, locale: str, surface: str, version: str) -> str:
    return f"qe:{version}:{surface}:{locale}:{query}"

def get_variants(query: str, locale: str, surface: str, *, cfg: QEConfig = CFG) -> list[str]:
    q0 = normalize_query(query)
    key = cache_key(q0, locale, surface, cfg.expansion_version)
    cached = redis.get_json(key)
    if cached:
        return cached["queries"]

    variants = safe_build_variants(q0, cfg=cfg)
    redis.set_json(key, {"queries": variants}, ttl_s=7 * 24 * 3600)
    return variants
```

### 8) Wiring into retrieval (what actually runs)
This shows the real control flow a new owner needs to reproduce.

```python
def pre_retrieval_candidates(query: str, locale: str, surface: str) -> list[int]:
    queries = get_variants(query, locale, surface, cfg=CFG)

    all_ids: list[int] = []
    for q in queries:
        vec = embed_text(q)                 # existing embedding service
        ids, _scores = ann_retrieve(vec)    # existing ANN service
        all_ids.extend(ids)

    return dedupe_candidates(all_ids)
```

## Outputs (contract level)
What downstream sees is unchanged: candidates for ranking. The only difference is candidate coverage.

```json
{
  "candidates": [712, 45, 98, 311, 901],
  "sources": {
    "query": "office chair",
    "variants": ["office chair", "ergonomic office chair", "adjustable desk chair lumbar support"],
    "expansion_version": "qe_v3"
  }
}
```

## Guardrails and failure modes (the ones that matter)
- **Semantic drift:** expansions wander from inventory reality; you see more candidates but worse downstream quality.
- **Over-expansion:** too many variants → higher embed/ANN/rank cost; cap variants hard.
- **Cache miss storms:** if normalization changes, hit rate collapses; treat normalization as part of the API.
- **Prompt drift:** small edits change output distribution; version prompts and cache keys together.
- **Latency creep:** LLM p95 rises; enforce timeout and fail open.
- **Locale leakage:** expansions in the wrong language; include locale in cache key and prompt if needed.

## Known limitations
- No inventory grounding; expansions can be “reasonable” but irrelevant.
- Works best for short queries; long queries often need constraint preservation instead of expansion.
- Adds complexity and operational surface area (cache, versioning, drift management).
