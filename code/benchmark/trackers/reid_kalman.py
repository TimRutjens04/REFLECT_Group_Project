from __future__ import annotations

import warnings

import cv2
import numpy as np
import torch
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import torchreid
import torchvision.transforms as T

from .base import BenchmarkTracker


# ── Appearance encoder — OSNet (torchreid) ────────────────────────────────────

class OsNetEncoder:
    """
    OSNet-x0.25 appearance encoder via torchreid.
    Produces L2-normalised 512-dim feature vectors.
    Runs on MPS / CUDA / CPU automatically.

    To swap in CLIP instead: replace this class with ClipEncoder from the
    commented block below and update ReidKalmanTracker.__init__ accordingly.
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    _SIZE = (256, 128)   # standard person-ReID input (h, w)

    def __init__(self) -> None:
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._model = torchreid.models.build_model(
            name="osnet_x0_25",
            num_classes=1,      # feature extraction only, class count irrelevant
            pretrained=True,
            use_gpu=self._device != "cpu",
        )
        self._model.eval()
        self._model.to(self._device)

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self._SIZE),
            T.ToTensor(),
            T.Normalize(mean=self._MEAN, std=self._STD),
        ])

    @torch.no_grad()
    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return L2-normalised 512-dim embedding for a BGR image crop."""
        if crop_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)
        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        feat   = self._model(tensor)        # (1, 512)
        feat   = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat.cpu().numpy().squeeze(0).astype(np.float32)


# ── (Alternative) CLIP encoder — uncomment and swap in if OSNet unavailable ───
#
# import clip
# from PIL import Image
#
# class ClipEncoder:
#     def __init__(self, model_name: str = "ViT-B/32") -> None:
#         device = "mps" if torch.backends.mps.is_available() else \
#                  "cuda" if torch.cuda.is_available() else "cpu"
#         self._device = device
#         self._model, self._preprocess = clip.load(model_name, device=device)
#         self._model.eval()
#
#     @torch.no_grad()
#     def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
#         rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
#         pil = Image.fromarray(rgb)
#         t   = self._preprocess(pil).unsqueeze(0).to(self._device)
#         f   = self._model.encode_image(t).float()
#         f   = f / f.norm(dim=-1, keepdim=True)
#         return f.cpu().numpy().squeeze(0)


# ── Simple constant-velocity Kalman filter (numpy only) ───────────────────────

class KalmanBox:
    """
    6-state constant-velocity Kalman filter.
    State: [cx, cy, w, h, vcx, vcy]  Observation: [cx, cy, w, h]
    """

    def __init__(self, bbox: np.ndarray) -> None:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  = float(bbox[2] - bbox[0])
        h  = float(bbox[3] - bbox[1])
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 4] = self.F[1, 5] = 1.0
        self.H = np.zeros((4, 6), dtype=np.float64)
        np.fill_diagonal(self.H, 1.0)
        self.Q = np.eye(6, dtype=np.float64) * 1e-2
        self.R = np.eye(4, dtype=np.float64) * 1e-1
        self.P = np.eye(6, dtype=np.float64) * 10.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_xyxy()

    def update(self, bbox: np.ndarray) -> None:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        z  = np.array([cx, cy, float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1])], dtype=np.float64)
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def _to_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)


# ── Tracker ───────────────────────────────────────────────────────────────────

class ReidKalmanTracker(BenchmarkTracker):
    """
    ReID + Kalman + IoU tracker.

    Motion:      Constant-velocity Kalman filter advanced every frame.
    Appearance:  CLIP embedding extracted at every GDINO detection.
                 The last confident embedding is kept as the reference.
    Re-ID:       On GDINO detection after track loss, accepts the box only if
                 weighted_dist = iou_w*(1-IoU) + app_w*cosine_dist is below
                 (1 - similarity_threshold). This prevents wrong re-ID when
                 a different object appears in the search region.
    """

    def __init__(self, cfg: dict, encoder: OsNetEncoder | None = None) -> None:
        r = cfg["reid"]
        self._iou_w      = float(r["iou_weight"])
        self._app_w      = float(r["appearance_weight"])
        self._sim_thresh = float(r["similarity_threshold"])
        self._encoder    = encoder if encoder is not None else OsNetEncoder()
        self._kalman:   KalmanBox | None  = None
        self._ref_emb:  np.ndarray | None = None
        self._track_id  = 0
        self._id_switches = 0
        self._lost      = False
        self._last_bbox: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "ReID+Kalman"

    @property
    def id_switches(self) -> int:
        return self._id_switches

    def reset(self) -> None:
        self._kalman    = None
        self._ref_emb   = None
        self._track_id  = 0
        self._id_switches = 0
        self._lost      = False
        self._last_bbox = None

    def _crop_encode(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1 = int(max(bbox[0], 0)); y1 = int(max(bbox[1], 0))
        x2 = int(min(bbox[2], w)); y2 = int(min(bbox[3], h))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        return self._encoder.encode(crop)

    def _reid_accepted(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> bool:
        """True if the new detection is a plausible match to the stored track."""
        if self._ref_emb is None or self._last_bbox is None:
            return True
        iou      = _iou(self._last_bbox, bbox)
        new_emb  = self._crop_encode(frame_bgr, bbox)
        cos_dist = 1.0 - float(np.dot(self._ref_emb, new_emb))
        d = self._iou_w * (1.0 - iou) + self._app_w * cos_dist
        return d < (1.0 - self._sim_thresh)

    def step(
        self,
        frame_bgr: np.ndarray,
        gdino_bbox: np.ndarray | None,
        gdino_score: float | None,
    ) -> tuple[np.ndarray | None, int]:
        if gdino_bbox is not None:
            if self._lost and not self._reid_accepted(frame_bgr, gdino_bbox):
                # Appearance check failed — keep predicting with Kalman
                if self._kalman is not None:
                    self._last_bbox = self._kalman.predict()
                return self._last_bbox, self._track_id

            if self._kalman is None:
                self._kalman = KalmanBox(gdino_bbox)
                self._track_id += 1
            else:
                if self._lost:
                    self._id_switches += 1
                    self._track_id += 1
                self._kalman.update(gdino_bbox)
                self._kalman.predict()

            self._ref_emb   = self._crop_encode(frame_bgr, gdino_bbox)
            self._last_bbox = gdino_bbox.copy()
            self._lost      = False
            return self._last_bbox, self._track_id

        if self._kalman is None:
            return None, self._track_id

        pred = self._kalman.predict()
        self._last_bbox = pred
        return pred, self._track_id


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (ua + 1e-6)
