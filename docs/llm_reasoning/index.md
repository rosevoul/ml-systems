# LLM-based Reasoning in Recommendation

## Context
Large Language Models (LLMs) are increasingly used in recommendation systems
*not* for embedding retrieval, but for **reasoning over candidates**.

Typical usage is downstream of classical retrieval and ranking models.

---

## Core idea
LLMs excel at:
- Interpreting user intent
- Applying explicit constraints
- Reasoning over tradeoffs
- Producing human-readable explanations

They operate on **small candidate sets**, not millions of items.

---

## Where it fits in a personalization system

```
User → Retrieval → Ranking → LLM Reasoning → Final Selection
```

LLMs are usually placed:
- After candidate generation
- After initial ranking
- Before UI rendering or agent actions

---

## Reasoning patterns

### Intent decomposition
Break high-level goals into concrete criteria.

### Constraint handling
Apply hard filters (budget, availability, policy).

### Candidate re-ranking
Re-order items based on reasoning, not similarity.

---

## Working illustrative example

- [LLM re-ranking notebook](../../notebooks/llm_rerank_toy.ipynb)

The notebook demonstrates:
- Constraint-aware filtering
- Reasoning-style scoring
- Explainable re-ranking logic

---

## Comparison with classical rankers

| Classical Rankers | LLM Reasoning |
|------------------|---------------|
| Numeric features | Symbolic rules |
| Learned weights | Prompt-driven logic |
| Hard to explain  | Naturally interpretable |

---

## Limitations
- Not suitable for large candidate sets
- Latency-sensitive
- Requires guardrails to prevent hallucinations

---

## Evaluation considerations
- Re-ranking quality
- Constraint satisfaction rate
- Latency impact
- User trust and interpretability
