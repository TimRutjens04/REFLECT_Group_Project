"""Application configuration for the live scene-graph demo."""

from __future__ import annotations

from dataclasses import dataclass, field

from .scene_graph.builder import SceneGraphConfig


def _webcam_scene_graph_config() -> SceneGraphConfig:
    """SceneGraphConfig retuned for a desk-scale webcam scene.

    The REFLECT defaults assume a robot cell viewed by a RealSense. At arm's
    length on a desk, objects sit within ~1 m of each other, monocular metric
    depth is noisier than a RealSense, and bbox sizes are larger relative to
    the frame — so distance/jump thresholds are loosened.
    """
    return SceneGraphConfig(
        near_threshold_m=0.45,
        support_depth_threshold_m=0.25,
        jump_threshold_m=0.50,
        iqr_threshold_m=0.40,
        direction_pixel_threshold=40.0,
    )


@dataclass
class AppConfig:
    # models
    yoloe_model: str = "yoloe-11s-seg.pt"
    depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    conf: float = 0.25
    # Tighter, class-agnostic NMS: open-vocab prompts often yield two boxes on
    # one object (same label twice, or synonym prompts), which produces
    # degenerate "mug inside mug" edges downstream.
    iou: float = 0.45
    agnostic_nms: bool = True
    # camera / intrinsics
    hfov_deg: float = 60.0
    width: int = 1280
    height: int = 720
    # performance
    depth_every_k: int = 2
    depth_input_long_side: int | None = None
    # device: None = auto-detect (cuda > mps > cpu)
    device: str | None = None
    # scene graph
    scene_graph: SceneGraphConfig = field(default_factory=_webcam_scene_graph_config)
    # DA-V2 metric checkpoints output metres already; never let the builder's
    # mm/m heuristic rescale them.
    depth_scale_to_m: float = 1.0
    # initial detection prompts (editable live via stdin)
    prompts: list[str] = field(default_factory=lambda: ["person", "cup", "bottle", "book"])
