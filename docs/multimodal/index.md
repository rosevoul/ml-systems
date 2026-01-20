# Multimodal modeling for recommendation (text + image + metadata)

We want a scorer **s(u, i, c)** that ranks items *i* for a user *u* in context *c* (query, session, feed position, device, locale). Items may include:

- **Text**: title, description, tags, reviews, OCR text, creator bios
- **Images**: thumbnails, product photos, posters, frames
- **Metadata**: category taxonomy, price, brand, freshness, language, region, safety labels, duration, etc.

Multimodal helps when any single modality is missing/noisy, or when you must generalize to new items (cold start) or sparse interactions.

## Embeddings: the common currency

### CLIP-style joint embeddings

A CLIP-like approach trains encoders for image and text into a **shared vector space** where matching image–text pairs are close and non-matching are far (contrastive learning). Practical implications:

- You can compute **item embeddings** from either image or text and still compare them.
- You can build retrieval/reranking that uses **query text** to retrieve **image-heavy** items, and vice versa.
- You can reuse pretrained models and fine-tune on your domain, which is often the fastest path to lift.

### Separate encoders + alignment

If CLIP-like training data is limited, a common strategy is:

1. Pretrain / use off-the-shelf **text encoder** and **image encoder** separately.
2. Learn a **projection head** per modality into a shared space using weak labels (co-clicks, co-view, purchase, playlist co-occurrence).

This often works well for recommendation where behavioral supervision is abundant.

## Fusion strategies

### Early fusion (feature-level fusion)

**Definition:** Combine modalities before the scoring model, e.g. `x = [e_text; e_image; e_meta]`, then feed to an MLP, tree model, or linear ranker.

**Pros**
- Model can learn cross-modal interactions (e.g., image×category, text×brand).
- Simple inference path: one model call.

**Cons**
- Requires all features at scoring time (or careful missing-modality handling).
- Larger feature vectors → higher latency and memory.
- If trained end-to-end with large encoders, it can be expensive.

**When to use**
- You have reliable feature availability at serving.
- You expect meaningful cross-feature interactions.
- You can control latency (e.g., reranker stage, not retrieval).

### Late fusion (score-level fusion)

**Definition:** Train separate models per modality (or modality groups) and combine their scores:
`score = w1*score_text + w2*score_image + w3*score_meta` (possibly context-dependent gating).

**Pros**
- Robust to missing modalities: drop a score term.
- Easy to debug and A/B: adjust weights, swap a model.
- Good for staged systems (retrieval from one modality, rerank with others).

**Cons**
- Limited interaction modeling (unless you add a gating network).
- Calibration becomes critical (scores must be comparable).

**When to use**
- Heterogeneous pipelines, multiple teams, or incremental rollout.
- You want graceful degradation under feature outages.
- You need strong interpretability and control.

### Hybrid patterns used in production

- **Retrieval:** CLIP-style or two-tower retrieval (query/user tower + item tower), approximate nearest neighbors.
- **Reranking:** early fusion MLP / GBDT using concatenated multimodal embeddings + metadata + behavioral features.
- **Re-ranking + constraints:** safety filters, diversity constraints, business rules.

## Cold start benefits

Multimodal features shine when interaction data is sparse:

- New items: text/image embeddings provide **semantic neighbors** → reasonable initial ranking.
- Long tail: better similarity estimation than IDs alone.
- Cross-lingual / multi-region: vision can generalize when text is short or translated poorly.

A common pattern is to **blend** content-based score with behavior-based score and let the behavior dominate as data accumulates.

## Evaluation: what breaks easily

### Offline pitfalls

1. **Label leakage via metadata**
   - Example: “is_promoted”, “inventory_status”, or “editor_pick” correlates with exposure/click.
   - Fix: separate *policy* features from *content* features; use counterfactual methods.

2. **Exposure bias / position bias**
   - Click labels are biased by what you showed and where.
   - Fix: IPS / propensity weighting, randomized interleaving, unbiased learning-to-rank where feasible.

3. **Hard negatives**
   - Random negatives are too easy; you learn a weak decision boundary.
   - Fix: sample “near misses” (in-bucket items, semantic neighbors, popular items).

4. **Modality imbalance**
   - If text is strong, adding image may appear useless offline even if it helps in edge cases.
   - Fix: slice analysis (short text, no text, low-light images, new items).

5. **Temporal leakage**
   - Train/val split must respect time.
   - Fix: time-based splits; evaluate on future windows.

### Online pitfalls

- **Latency regression**: bigger encoders and larger feature payloads; be explicit about budgets per stage.
- **Feature availability**: images missing, OCR delayed, embeddings not computed; plan defaults.
- **Calibration drift**: late fusion weights require monitoring and periodic re-tuning.
- **Distribution shifts**: creative styles change (e.g., thumbnails), language shifts, new categories.

## Senior MLE plan: build it right

### 1) Define the system stage and latency budget

- Retrieval (10–50ms), rerank (50–150ms), post-rank (rules).
- Decide where multimodal enters (often: retrieval via CLIP/two-tower, rerank via early fusion).

### 2) Decide supervision signal and objective

- Pointwise: click / conversion probability (simple, biased).
- Pairwise: (clicked > unclicked in same impression) for ranking.
- Listwise: approximate NDCG loss (more complex).

### 3) Feature + embedding pipeline

- Offline batch embeddings (daily/hourly): store in feature store / vector DB.
- Online: ensure embedding freshness SLA; backfill; cache.
- Missing modality strategy: learned “missing” token, zero vector + mask, or late fusion drop.

### 4) Evaluation protocol

- Offline: NDCG@K, Recall@K, MAP; plus slices for cold-start and missing modalities.
- Online: interleaving or A/B, guardrails (latency, diversity, safety, revenue).

### 5) Monitoring

- Feature coverage per modality.
- Embedding drift (mean/variance, nearest neighbor stability).
- Score distribution per modality and fused score calibration.

---

## Applied Example
The notebook `multimodal_fusion_notes.ipynb` demonstrates:

- **Text embeddings** via TF‑IDF.
- **Image embeddings** via PCA on small grayscale images.
- **Metadata** via one-hot encoding.
- **Early fusion** by concatenating embeddings and training a logistic ranker.
- **Late fusion** by training separate text and image models and combining their scores.
- Evaluation with **NDCG@10** on synthetic impression logs.

