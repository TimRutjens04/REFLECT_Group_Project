import time

import clip
import streamlit as st

from .config import DEVICE
from .logger import log


@st.cache_resource(show_spinner="Loading CLIP ViT-B/32...")
def load_clip_model():
    """ViT-B/32 — used for attention maps (matches stored embeddings)."""
    log.info("Loading CLIP ViT-B/32 on device=%s", DEVICE)
    t0 = time.perf_counter()
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    log.info("CLIP ViT-B/32 loaded in %.2fs", time.perf_counter() - t0)
    return model, preprocess


@st.cache_resource(show_spinner="Loading OWL-ViT v2 for object detection...")
def load_owlvit_model():
    """OWLv2 — zero-shot object detection with absolute confidence scores."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    log.info("Loading OWL-ViT v2 on device=%s", DEVICE)
    t0 = time.perf_counter()
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")
    model = model.to(DEVICE)
    model.eval()
    log.info("OWL-ViT v2 loaded in %.2fs", time.perf_counter() - t0)
    return model, processor
