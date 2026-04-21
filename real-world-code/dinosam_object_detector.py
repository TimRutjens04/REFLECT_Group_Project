"""Object detection + segmentation using Grounding DINO + SAM.

Drop-in replacement for the old MDETR-based pipeline.
Requires the Grounded-Segment-Anything repo to be cloned and on sys.path:
  - GroundingDINO  (pip install -e GroundingDINO/)
  - segment_anything (pip install -e segment_anything/)

Checkpoints (download once):
  wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.patches import Polygon
from PIL import Image
from skimage.measure import find_contours

# ── Make sure GroundingDINO + SAM are importable ────────────────────────
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

import GroundingDINO.groundingdino.datasets.transforms as GD_T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from segment_anything import SamPredictor, sam_model_registry

# ── Config / paths ──────────────────────────────────────────────────────
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "./groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Thresholds (override via env vars if you want) ──────────────────────
BOX_THRESHOLD = float(os.getenv("GDINO_BOX_THRESH", "0.35"))
TEXT_THRESHOLD = float(os.getenv("GDINO_TEXT_THRESH", "0.25"))
NMS_THRESHOLD = float(os.getenv("GDINO_NMS_THRESH", "0.5"))

# ── Visualisation colours ───────────────────────────────────────────────
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════


def _load_grounding_dino(
    config_path: str = GROUNDING_DINO_CONFIG,
    checkpoint_path: str = GROUNDING_DINO_CHECKPOINT,
    device: str = DEVICE,
) -> torch.nn.Module:
    """Build and load Grounding DINO from config + checkpoint."""
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval()
    return model.to(device)


def _load_sam_predictor(
    checkpoint_path: str = SAM_CHECKPOINT,
    encoder_version: str = SAM_ENCODER_VERSION,
    device: str = DEVICE,
) -> SamPredictor:
    """Build SAM and wrap it in a SamPredictor."""
    sam = sam_model_registry[encoder_version](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)


# Lazy-init globals so the script can be imported without immediately
# loading multi-GB weights.
_gdino_model: torch.nn.Module | None = None
_sam_predictor: SamPredictor | None = None


def _get_models():
    global _gdino_model, _sam_predictor
    if _gdino_model is None:
        _gdino_model = _load_grounding_dino()
    if _sam_predictor is None:
        _sam_predictor = _load_sam_predictor()
    return _gdino_model, _sam_predictor


# ═══════════════════════════════════════════════════════════════════════
# Image pre-processing (Grounding DINO expects its own transform)
# ═══════════════════════════════════════════════════════════════════════

_gdino_transform = GD_T.Compose(
    [
        GD_T.RandomResize([800], max_size=1333),
        GD_T.ToTensor(),
        GD_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _load_image_for_gdino(pil_img: Image.Image):
    """Return (pil_img, transformed_tensor)."""
    image_tensor, _ = _gdino_transform(pil_img, None)  # (3, H, W)
    return image_tensor


# ═══════════════════════════════════════════════════════════════════════
# Grounding DINO detection
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def _get_grounding_output(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    caption: str,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
    device: str = DEVICE,
):
    """Run Grounding DINO and return (boxes_filt, scores_filt, pred_phrases).

    boxes_filt : (N, 4)  in cxcywh normalised format
    scores_filt: (N,)
    pred_phrases: list[str]
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor.unsqueeze(0), captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, 256)
    boxes = outputs["pred_boxes"][0]  # (num_queries, 4)

    # Filter by box threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]  # (N, 256)
    boxes_filt = boxes[filt_mask]  # (N, 4)

    # Build text labels from the token logits
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    scores_filt = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenizer
        )
        scores_filt.append(logit.max().item())
        pred_phrases.append(pred_phrase)

    scores_filt = torch.tensor(scores_filt)
    return boxes_filt, scores_filt, pred_phrases


def _cxcywh_to_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Convert normalised cxcywh boxes to pixel xyxy."""
    cx, cy, bw, bh = boxes.unbind(-1)
    x0 = (cx - 0.5 * bw) * w
    y0 = (cy - 0.5 * bh) * h
    x1 = (cx + 0.5 * bw) * w
    y1 = (cy + 0.5 * bh) * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# SAM segmentation from boxes
# ═══════════════════════════════════════════════════════════════════════


def _segment_with_sam(
    sam_predictor: SamPredictor,
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
) -> np.ndarray:
    """Run SAM on each box and return (N, H, W) boolean masks."""
    sam_predictor.set_image(image_rgb)
    result_masks = []
    for box in boxes_xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        result_masks.append(masks[np.argmax(scores)])
    if len(result_masks) == 0:
        return np.empty((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
    return np.array(result_masks)


# ═══════════════════════════════════════════════════════════════════════
# Visualisation helpers (kept from original script)
# ═══════════════════════════════════════════════════════════════════════


def apply_mask(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.5):
    """Overlay a boolean mask on an image."""
    for c in range(3):
        image[:, :, c] = np.where(
            mask,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    return image


def plot_results(pil_img, scores, boxes, labels, masks):
    """Draw boxes + masks on the image and return a PIL Image."""
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    if masks is None:
        masks = [None] * len(scores)

    assert len(scores) == len(boxes) == len(labels) == len(masks)

    for s, (xmin, ymin, xmax, ymax), label, mask, c in zip(
        scores, boxes.tolist(), labels, masks, colors
    ):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=c,
                linewidth=3,
            )
        )
        ax.text(xmin, ymin, f"{label}: {s:0.2f}", fontsize=8)

        if mask is None:
            continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        for verts in find_contours(padded_mask, 0.5):
            verts = np.fliplr(verts) - 1
            ax.add_patch(Polygon(verts, facecolor="none", edgecolor=c))

    return Image.fromarray(np_image)


# ═══════════════════════════════════════════════════════════════════════
# High-level API (matches old interface)
# ═══════════════════════════════════════════════════════════════════════


def plot_inference(im: Image.Image, caption: str, idx=None, model=None):
    """Grounding DINO detection only (no masks).

    Returns (outputs_dict, labels) to match the old MDETR signature.
    """
    gdino, _ = _get_models()
    img_tensor = _load_image_for_gdino(im)
    boxes_filt, scores_filt, pred_phrases = _get_grounding_output(
        gdino, img_tensor, caption
    )

    w, h = im.size
    boxes_xyxy = _cxcywh_to_xyxy(boxes_filt.cpu(), w, h)

    outputs = {
        "pred_boxes": boxes_filt.unsqueeze(0),
        "scores": scores_filt,
    }
    return outputs, pred_phrases


def plot_inference_segmentation(
    im: Image.Image,
    caption: str,
    seg_model=None,
):
    """Full Grounding DINO → SAM pipeline.

    Returns a dict with the same keys as the old MDETR version:
      probs, labels, bbox_2d, masks, im
    """
    gdino, sam_pred = _get_models()

    # ── 1. Grounding DINO detection ─────────────────────────────────
    img_tensor = _load_image_for_gdino(im)
    boxes_filt, scores_filt, pred_phrases = _get_grounding_output(
        gdino, img_tensor, caption
    )

    w, h = im.size
    boxes_xyxy = _cxcywh_to_xyxy(boxes_filt.cpu(), w, h)

    # ── 2. NMS ──────────────────────────────────────────────────────
    if len(boxes_xyxy) > 0:
        nms_idx = (
            torchvision.ops.nms(boxes_xyxy, scores_filt, NMS_THRESHOLD).numpy().tolist()
        )
        boxes_xyxy = boxes_xyxy[nms_idx]
        scores_filt = scores_filt[nms_idx]
        pred_phrases = [pred_phrases[i] for i in nms_idx]

    print(f"[GroundedSAM] caption='{caption}'  kept={len(boxes_xyxy)} boxes")

    # ── 3. SAM segmentation ─────────────────────────────────────────
    image_rgb = np.array(im)
    masks = _segment_with_sam(sam_pred, image_rgb, boxes_xyxy.numpy())

    # ── 4. Optional mask erosion (same as original) ─────────────────
    shrunk_masks = []
    if len(masks) > 0:
        kernel = np.ones((3, 3), np.uint8)
        for mask in masks:
            eroded = cv2.erode(mask.astype(np.float32), kernel, iterations=2)
            shrunk_masks.append(eroded)
        shrunk_masks = np.array(shrunk_masks)
    else:
        shrunk_masks = masks

    # ── 5. Labels: fall back to caption when phrase is empty ────────
    labels = [p.strip() or caption for p in pred_phrases]

    # ── 6. Plot ─────────────────────────────────────────────────────
    plt.figure(figsize=(16, 10))
    plt.imshow(image_rgb)
    vis_masks = masks if len(masks) > 0 else None
    annotated = plot_results(im, scores_filt, boxes_xyxy, labels, vis_masks)

    return {
        "probs": scores_filt,
        "labels": [caption] * len(masks),
        "bbox_2d": boxes_xyxy,
        "masks": shrunk_masks,
        "im": annotated,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("Robot Failure Summarization")
    parser.add_argument(
        "--folder_name",
        type=str,
        default="",
        help="If pipeline should be run on only one specific folder",
    )
    parser.add_argument(
        "--gdino_config",
        type=str,
        default=GROUNDING_DINO_CONFIG,
    )
    parser.add_argument(
        "--gdino_checkpoint",
        type=str,
        default=GROUNDING_DINO_CHECKPOINT,
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=SAM_CHECKPOINT,
    )
    return parser


if __name__ == "__main__":
    args = config_parser().parse_args()
    os.makedirs(f"object_detection/mdetr/{args.folder_name}", exist_ok=True)

    task_objects = ["faucet", "mug", "sink"]

    total_frames = 1
    for idx in range(1, total_frames + 1):
        plt.figure(figsize=(16, 10))
        im = Image.open(f"real_world/data/test/rgb/3.png").convert("RGB")

        for single_obj_prompt in task_objects:
            retval = plot_inference_segmentation(im, single_obj_prompt)

        print("type(retval['im']): ", type(retval["im"]))
        plt.imshow(retval["im"])
        plt.savefig(f"real_world/state_summary/temp/mdetr/{idx}.png")
        plt.close()
