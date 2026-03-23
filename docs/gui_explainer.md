# Live Embedding Visualizer — Technical Explainer

A walkthrough of every panel in the GUI and the model machinery behind it.

---

## 1. The frame + attention heatmap

**What you see:** the current video frame with a semi-transparent colour overlay and a red rectangle.

**How it works:**

CLIP's vision encoder is a Vision Transformer (ViT-B/32). When it processes an image it divides the 224×224 input into a 7×7 grid of 32×32-pixel *patches* (49 patches total) plus one special `[CLS]` token. These 50 tokens flow through 12 transformer blocks. Inside every block, each token attends to every other token via multi-head self-attention.

We install a **forward hook** on the very last attention layer (`resblocks[-1].attn`). The hook captures the raw attention weight matrix just before it is applied to the value vectors. We extract the row that belongs to `[CLS]` — the 49 values in that row tell us how much the global representation token is "looking at" each spatial patch. That gives us a (7, 7) map of attention weights.

To turn this into the heatmap:
1. Bilinearly resize the 7×7 map to match the original frame resolution.
2. Normalise to [0, 1] and apply the Jet colour map.
3. Alpha-blend with the raw frame at the user-selected opacity.

For the **red rectangle**: we find the patch with the highest attention weight (argmax of the 7×7 map), then map its row/column back to pixel coordinates. CLIP center-crops the input to `min(H, W) × min(H, W)` before resizing to 224×224, so the mapping accounts for the crop offset and scaling factor.

> **Why ViT-B/32 specifically?** The stored embeddings in `encoded/` were computed with ViT-B/32. Using the same model for attention maps ensures the heatmap reflects exactly what the embedding encodes.

---

## 2. Scene graph (left column, compact)

**What you see:** a small graph where nodes are detected objects, edges label their spatial relationships (above / below / left of / right of / near), and a yellow border signals that an object overlaps with one of the top-3 most-attended patches. A ★ marks the single attention peak.

**How the node positions are computed:**

Object positions come from OWL-ViT (see section 3). Each detected bounding box has a centre `(cx_norm, cy_norm)` in normalised image coordinates (0–1). The graph plots nodes at those positions so the scene graph is literally a spatial layout of the objects as they appear in the frame.

**Spatial relationships:**

For every pair of detected objects we compute `Δx = cx_norm_b − cx_norm_a` and `Δy = cy_norm_b − cy_norm_a`. If the Euclidean distance is below 0.1 we call it *near*; otherwise the dominant axis determines above/below/left of/right of.

**Attention–object overlap:**

We convert each object's `(cx_norm, cy_norm)` to a 7×7 patch index by `r = floor(cy * 7)`, `c = floor(cx * 7)`. If that patch is among the top-3 in the CLIP attention map, the node gets a yellow border. This is the direct connection between *where CLIP is looking* and *what object is there*.

**Undetected objects** (confidence below 0.10) are rendered grey at the bottom of the plot and excluded from edges.

---

## 3. Object localization — OWL-ViT v2 (bottom panel)

**What you see:** the first frame of the episode with coloured bounding boxes and confidence scores. Yellow border = within a top-3 attention patch.

**How OWL-ViT works:**

OWL-ViT (Open-Vocabulary Localization with Vision Transformers) is a CLIP backbone fine-tuned for zero-shot object detection. The key difference from plain CLIP is that OWL-ViT preserves the full sequence of patch tokens rather than collapsing them to a single CLS embedding. Each patch token is matched against a text query embedding, producing one confidence score per (patch, query) pair. Non-maximum suppression groups nearby high-scoring patches into bounding boxes.

Because OWL-ViT produces **absolute confidence scores** (not cosine similarities relative to other patches), it can genuinely say "this object is not in the scene" — a score below 0.10 means the model found no patch that resembles the queried object.

**What we query:**

```python
texts = [["a carrot", "a strawberry", "a green pear", "a red apple", "a bowl"]]
```

The object list comes from `tasks_real_world.json`. We run detection once on the **first frame** of each episode (cached per episode) and reuse the result for all frames. This works because scene layout is effectively static for the real-world episodes — objects don't move significantly between frames.

**Why first-frame only?**

Running OWL-ViT on all 243 frames would take ~8 minutes. The scene graph is meant to show the *structural context* of the episode, not per-frame changes. The attention heatmap already captures per-frame temporal dynamics.

---

## 4. Embedding signals (top-right)

**What you see:** two time-series overlaid on the same axis, plus a yellow cursor at the current frame.

| Signal | Colour | Meaning |
|---|---|---|
| Visual (CLIP) | blue | Cosine similarity of each frame's 512-dim CLIP embedding to the episode mean embedding |
| Audio (WAV2CLIP) | green | Same metric for WAV2CLIP audio embeddings |

Dips indicate frames whose embedding is *far from typical* — which often (not always) coincides with failures or anomalous robot behaviour.

The second panel shows **frame-to-frame cosine distance** (1 − cosine similarity between consecutive frames). Spikes indicate sudden visual change.

---

## 5. PCA trajectory (bottom-right)

**What you see:** a 2D scatter of the visual embedding trajectory through time, colour-coded from dark (early) to bright (late). The current frame is the yellow star. Past trajectory is bright; future is faded.

**How it's computed:**

During encoding (`code/encode.py`) we run PCA on all 243 CLIP embeddings of the episode and project them down to 2 dimensions. The `pca_coords` array stored in `encoded/<episode>.npz` is what gets plotted.

This shows whether failure frames cluster separately in embedding space, and whether the robot's visual state changes smoothly or jumps.

---

## Data flow summary

```
video frame
    │
    ├─► CLIP ViT-B/32 (encode_image)
    │       ├─ CLS token → 512-dim embedding        → PCA / signals plots
    │       └─ last-layer attention weights (hook)  → 7×7 heatmap + red box
    │
    └─► OWL-ViT v2 (first frame only, cached)
            └─ patch × text scores → bounding boxes → scene graph nodes + bottom panel
```
