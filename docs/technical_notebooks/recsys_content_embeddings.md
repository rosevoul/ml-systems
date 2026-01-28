# Content Embeddings (for Retrieval)

## TLDR
This component builds item embeddings from content for retrieval. 
For example, it turns item titles and images into vectors before any user behavior exists. 
It runs offline, produces versioned embedding tables, and is used by ANN retrieval. 

**Methods**
- **TF-IDF + SVD:** deterministic text baseline from item titles and descriptions.
- **Sentence-BERT:** learned text embeddings from a served model.
- **Minimal vision baseline:** cheap image baseline to validate plumbing.
- **ViT:** learned image embeddings from a served model.
- **CLIP:** shared text–image embedding space.
- **Content fusion:** combines text and image embeddings into one vector.

---

## What this component builds
- Generates item embeddings from item content fields (title, description, images).
- Supports multiple embedding methods with the same output format.
- Produces versioned embedding artifacts consumable by retrieval.
- Does not train rankers or apply business rules.

## When this component is needed
- Cold-start items need vectors before interaction data accrues.
- Semantic coverage is needed beyond behavior-only embeddings.
- You want text and image retrieval to work together.
- Embedding computation runs offline with periodic refresh.

## How this component fits in the retrieval flow

```
Item catalog + content
   ↓
Content embedding job (text, image, CLIP)
   ↓
Embedding store (versioned)
   ↓
ANN index build + publish
   ↓
Serving retrieval / ranking
```

Embedding versioning is the safety mechanism. Rollback is a version switch.

## What goes in and what comes out

Input (catalog row):
```json
{
  "item_id": 712,
  "title": "Ergonomic Mesh Office Chair",
  "description": "Adjustable lumbar support and breathable mesh.",
  "image_urls": ["https://.../712_main.jpg"]
}
```

Output:
```json
{
  "item_id": 712,
  "embedding": [0.031, -0.044, ...],
  "version": "content_v12"
}
```

For this component, the output is processed into a fixed-size vector, L2-normalized, and versioned.

## How content embeddings are built

### 1) Simple text baseline from item text
**Method:** TF-IDF + SVD. 
Deterministic offline method, for example from item titles and short descriptions.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def fit_tfidf_svd(texts: list[str], *, dim: int = 128):
    tfidf = TfidfVectorizer(min_df=2, max_features=200_000, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=dim, random_state=0)
    Z = svd.fit_transform(X)
    return tfidf, svd, l2_normalize(Z)

def encode_tfidf_svd(tfidf, svd, texts: list[str]) -> np.ndarray:
    X = tfidf.transform(texts)
    Z = svd.transform(X)
    return l2_normalize(Z)
```

### 2) Learned text embeddings from a served model
**Method:** Sentence-BERT. 
Use when semantic meaning matters, for example “office chair” vs “desk chair”.

```python
def encode_text_sbert(texts: list[str], *, model_version: str) -> np.ndarray:
    resp = text_embed_svc.batch_encode(
        texts=texts,
        model_version=model_version,
        max_batch=256,
        timeout_ms=5000
    )
    E = np.asarray(resp["embeddings"], dtype=np.float32)
    return l2_normalize(E)
```

### 3) Simple image baseline for plumbing checks
**Method:** Minimal vision baseline. 
Use to validate plumbing and indexing before deploying learned models.

```python
def encode_image_minimal(imgs: list[np.ndarray], *, out_dim: int = 128) -> np.ndarray:
    feats = []
    for im in imgs:
        small = im[::4, ::4]                     # downsample
        gx = np.diff(small, axis=1, prepend=0)
        gy = np.diff(small, axis=0, prepend=0)
        f = np.concatenate([small.ravel(), gx.ravel(), gy.ravel()])[:out_dim]
        feats.append(f.astype(np.float32))
    return l2_normalize(np.vstack(feats))
```

### 4) Learned image embeddings from a served model
**Method:** Vision Transformer (ViT). 
Use when visual similarity matters, for example furniture style or color.

```python
def encode_image_vit(img_bytes: list[bytes], *, model_version: str) -> np.ndarray:
    resp = vision_embed_svc.batch_encode(
        images=img_bytes,
        model_version=model_version,
        max_batch=64,
        timeout_ms=8000
    )
    E = np.asarray(resp["embeddings"], dtype=np.float32)
    return l2_normalize(E)
```

### 5) Joint text–image space
**Method:** CLIP. 
Use when text queries should retrieve visually similar items.

```python
def encode_clip_text(texts: list[str], *, model_version: str) -> np.ndarray:
    E = clip_svc.encode_text(texts=texts, model_version=model_version, max_batch=256)
    return l2_normalize(np.asarray(E, dtype=np.float32))

def encode_clip_image(img_bytes: list[bytes], *, model_version: str) -> np.ndarray:
    E = clip_svc.encode_image(images=img_bytes, model_version=model_version, max_batch=64)
    return l2_normalize(np.asarray(E, dtype=np.float32))
```

### 6) Combining text and image embeddings
**Method:** Content fusion. 
Use when both text and images are available for the same item.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ContentConfig:
    version: str = "content_v12"
    backend: str = "clip"         # "tfidf_svd" | "sbert" | "vit" | "clip"
    dim: int = 512

CFG = ContentConfig()

def build_item_embedding(row: dict) -> np.ndarray:
    if CFG.backend == "clip":
        txt = row["title"] + " " + (row.get("description") or "")
        e_txt = encode_clip_text([txt], model_version="clip_v3")[0]
        e_img = encode_clip_image([download_image(row["image_urls"][0])], model_version="clip_v3")[0]
        e = l2_normalize(0.3 * e_txt + 0.7 * e_img)  # simple, stable fusion
        return e

    raise NotImplementedError(CFG.backend)

def write_embeddings(rows: list[dict]):
    for r in rows:
        e = build_item_embedding(r)
        embedding_store.put(item_id=r["item_id"], vec=e, version=CFG.version)
```

## What can go wrong and how to notice it
- **Version skew:** retrieval index built from content_v11 while serving uses content_v12. enforce pointer consistency.
- **Batching bugs:** embedding services return out-of-order results. include item_id alignment checks.
- **Normalization drift:** some backends return unnormalized vectors. enforce L2 at boundary.
- **Missing images or text:** do not invent content. fall back to the available modality or mark as missing.
- **Distribution shift:** catalog text changes (templates, spam). baselines (TF-IDF) can degrade sharply.



## Things to note
- This component owns content embedding quality and stability.
- Content embeddings capture similarity, not preference.
- CLIP fusion weights are heuristic unless trained. keep them stable and versioned.
- Encoding jobs can be expensive. plan refresh cadence and incremental updates.
