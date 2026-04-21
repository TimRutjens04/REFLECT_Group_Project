"""
RoboFail Scene Graph Pipeline
===============================
Adapted version of the real-world scene graph pipeline that works on
RoboFail dataset video recordings instead of live robot sensor data.

Replaces:
    - real_world_get_local_sg.py  (get_scene_graph)
    - real_world_hierarchical_prompt.py  (run_real_world_pipeline)

Keeps intact:
    - real_world_scene_graph.py  (Node, Edge, SceneGraph classes)
    - mdetr_object_detector.py  (MDETR detection + segmentation)
    - constants.py  (state dicts, bulky objects, name maps)

Usage:
    python robofail_scene_graph_pipeline.py \
        --video_path path/to/video.mp4 \
        --task_json path/to/task.json \
        --output_dir robofail_output/task_name \
        --object_list "mug,faucet,sink,pot" \
        --depth_model depth_anything_v2
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser
from collections import defaultdict
from typing import Optional

import torch
import cv2

# --- Imports from the original pipeline (keep these files as-is) ---
from real_world_scene_graph import Node, Edge, SceneGraph, get_object_state, state_dict
from constants import BULKY_OBJECTS, real_world_name_map, real_world_obj_state_map
from robofail_video_loader import RoboFailVideoLoader, FrameData

# Try to import 3D dependencies — fall back to 2D-only mode if unavailable
try:
    import open3d as o3d
    from transforms import depth_to_point_cloud
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[Warning] open3d not available. Running in 2D-only scene graph mode.")

# Try to import MDETR
try:
    from mdetr_object_detector import (
        plot_inference_segmentation, seg_model, transform, device
    )
    HAS_MDETR = True
except ImportError:
    HAS_MDETR = False
    print("[Warning] MDETR not available. Object detection will be skipped.")

# Try to import CLIP utils (for object state classification and detection confirmation)
try:
    from main.clip_utils import get_img_feats, get_text_feats, get_nn_text
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("[Warning] CLIP utils not available. Object state detection will be skipped.")

# Try to import AudioCLIP
try:
    from AudioCLIP.real_world_audio import get_sound_events
    HAS_AUDIOCLIP = True
except ImportError:
    HAS_AUDIOCLIP = False
    print("[Warning] AudioCLIP not available. Sound detection will be skipped.")


# ============================================================================
# 2D Scene Graph Builder (fallback when no depth is available)
# ============================================================================

class SceneGraph2D(SceneGraph):
    """
    A 2D variant of the scene graph that infers spatial relationships
    from 2D bounding boxes instead of 3D point clouds.
    
    Used when:
    - No depth sensor data is available
    - Monocular depth estimation is not loaded
    - open3d is not installed
    
    Edge types supported:
    - "on top of": bbox B is above bbox A and overlaps horizontally
    - "inside": bbox B is mostly contained within bbox A
    - "on the left of" / "on the right of": horizontal spatial relation
    - "above" / "below": vertical spatial relation
    - "near": bboxes are close but no specific containment
    - "occluding": significant overlap (front object occluding back)
    """
    
    def add_edge(self, node, new_node):
        """Infer spatial relationship from 2D bounding boxes."""
        if "bowl" in new_node.name and "apple" in node.name:
            return
        
        if node.bbox2d is None or new_node.bbox2d is None:
            return
        
        box_a = node.bbox2d  # [xmin, ymin, xmax, ymax]
        box_b = new_node.bbox2d
        
        # Calculate overlap / containment
        iou, inters = self._get_iou(box_a, box_b)
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        containment_ratio = inters / area_b  # how much of B is inside A
        
        # Centers
        cx_a = (box_a[0] + box_a[2]) / 2
        cy_a = (box_a[1] + box_a[3]) / 2
        cx_b = (box_b[0] + box_b[2]) / 2
        cy_b = (box_b[1] + box_b[3]) / 2
        
        # Normalized distance (relative to image diagonal)
        img_diag = max(1, np.sqrt(box_a[2]**2 + box_a[3]**2))  # rough estimate
        dist = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2) / img_diag
        
        # Horizontal and vertical overlap
        h_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
        v_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
        
        width_b = max(1, box_b[2] - box_b[0])
        height_b = max(1, box_b[3] - box_b[1])
        
        # --- Determine edge type ---
        
        # INSIDE: new_node is mostly contained in node
        if containment_ratio > 0.6 and new_node.name.split('-')[0] not in BULKY_OBJECTS:
            if node.name.split('-')[0] in ["countertop", "stove burner", "table"]:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
            else:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")
            return
        
        # ON TOP OF: B is above A with horizontal overlap
        if (cy_b < cy_a  # B center is higher (lower y = higher in image)
            and h_overlap > 0.5 * width_b  # significant horizontal overlap
            and abs(box_b[3] - box_a[1]) < 0.15 * max(area_a, area_b)**0.5  # bottoms close
            and new_node.name.split('-')[0] not in BULKY_OBJECTS):
            self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
            return
        
        # Skip bulky <-> bulky relations for spatial edges
        if node.name.split('-')[0] in BULKY_OBJECTS and new_node.name.split('-')[0] in BULKY_OBJECTS:
            return
        
        # CLOSE OBJECTS: infer direction
        if dist < 0.3:
            dx = cx_b - cx_a
            dy = cy_b - cy_a
            
            if abs(dy) > abs(dx) * 1.5:
                # Primarily vertical
                if dy < 0:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, "above")
                else:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, "below")
            elif abs(dx) > abs(dy) * 1.5:
                # Primarily horizontal
                if dx < 0:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on the left of")
                else:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on the right of")
            elif iou > 0.1:
                # Significant overlap -> occluding
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "occluding")
            else:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "near")
    
    @staticmethod
    def _get_iou(box_a, box_b):
        """Calculate IoU and intersection area between two [xmin, ymin, xmax, ymax] boxes."""
        ixmin = max(box_a[0], box_b[0])
        ixmax = min(box_a[2], box_b[2])
        iymin = max(box_a[1], box_b[1])
        iymax = min(box_a[3], box_b[3])
        
        iw = max(0, ixmax - ixmin)
        ih = max(0, iymax - iymin)
        inters = iw * ih
        
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inters
        
        iou = inters / max(union, 1e-6)
        return iou, inters


# ============================================================================
# Object Detection (wraps MDETR for video frames)
# ============================================================================

def detect_objects_in_frame(rgb: np.ndarray, object_list: list, step_idx: int) -> dict:
    """
    Run MDETR object detection + segmentation on a single RGB frame.
    
    Args:
        rgb: (H, W, 3) uint8 RGB image
        object_list: list of object name strings to detect
        step_idx: frame index (for logging/saving)
    
    Returns:
        dict with keys: total_detections, labels, scores, pred_masks, bbox_2d
    """
    if not HAS_MDETR:
        print(f"  [Frame {step_idx}] MDETR not available, returning empty detections")
        return _empty_detections()
    
    outputs = {
        'total_detections': 0,
        'labels': np.array([]),
        'scores': np.array([]),
        'pred_masks': np.array([]),
        'bbox_2d': np.array([]),
    }
    
    im = Image.fromarray(rgb)
    
    for single_obj_prompt in object_list:
        retval = plot_inference_segmentation(im, single_obj_prompt, seg_model)
        
        if len(retval['masks']) == 0:
            continue
        
        if len(outputs['pred_masks']) == 0:
            outputs['pred_masks'] = retval['masks']
            outputs['bbox_2d'] = retval['bbox_2d']
        else:
            outputs['pred_masks'] = np.concatenate((outputs['pred_masks'], retval['masks']))
            outputs['bbox_2d'] = np.concatenate((outputs['bbox_2d'], retval['bbox_2d']))
        
        outputs['scores'] = np.concatenate((outputs['scores'], retval['probs']))
        outputs['labels'] = np.concatenate((outputs['labels'], retval['labels']))
    
    outputs['total_detections'] = len(outputs['scores'])
    return outputs


def confirm_detections_clip(rgb, outputs, object_list):
    """
    Use CLIP to confirm/filter MDETR detections (same logic as original confirm_obj_det).
    Falls back to unfiltered detections if CLIP is unavailable.
    """
    if not HAS_CLIP or outputs['total_detections'] == 0:
        return outputs
    
    h, w, _ = rgb.shape
    
    # Sort by score
    if len(outputs['scores']) > 1:
        sorted_indices = np.argsort(outputs['scores'])
        outputs['labels'] = outputs['labels'][sorted_indices]
        outputs['scores'] = outputs['scores'][sorted_indices]
        outputs['pred_masks'] = outputs['pred_masks'][sorted_indices]
        outputs['bbox_2d'] = outputs['bbox_2d'][sorted_indices]
    
    confirmed = {
        'total_detections': 0,
        'labels': np.array([]),
        'scores': np.array([]),
        'pred_masks': np.array([]),
        'bbox_2d': np.array([]),
    }
    
    for idx in range(len(outputs['bbox_2d'])):
        box = outputs['bbox_2d'][idx]
        h1 = max(0, int(box[1] - 10))
        h2 = min(h, int(box[3] + 10))
        w1 = max(0, int(box[0] - 10))
        w2 = min(w, int(box[2] + 10))
        cropped_img = rgb[h1:h2, w1:w2]
        
        if cropped_img.size == 0:
            continue
        
        img_feats = get_img_feats(cropped_img)
        obj_name_feats = get_text_feats(object_list)
        sorted_obj_names, sorted_scores = get_nn_text(object_list, obj_name_feats, img_feats)
        
        label = outputs['labels'][idx]
        clip_conf = False
        for i in range(len(sorted_scores)):
            if i == 0 and sorted_obj_names[i] == label:
                clip_conf = True
                break
            elif sorted_scores[i] > 0.23 and label == sorted_obj_names[i]:
                clip_conf = True
                break
        
        if clip_conf:
            if len(confirmed['pred_masks']) == 0:
                confirmed['pred_masks'] = np.expand_dims(outputs['pred_masks'][idx], axis=0)
                confirmed['bbox_2d'] = np.expand_dims(outputs['bbox_2d'][idx], axis=0)
            else:
                confirmed['pred_masks'] = np.concatenate((
                    confirmed['pred_masks'],
                    np.expand_dims(outputs['pred_masks'][idx], axis=0)
                ))
                confirmed['bbox_2d'] = np.concatenate((
                    confirmed['bbox_2d'],
                    np.expand_dims(outputs['bbox_2d'][idx], axis=0)
                ))
            confirmed['scores'] = np.append(confirmed['scores'], outputs['scores'][idx])
            confirmed['labels'] = np.append(confirmed['labels'], outputs['labels'][idx])
    
    confirmed['total_detections'] = len(confirmed['scores'])
    return confirmed


def edit_label(label_arr):
    """Rename labels and add instance counters for duplicates (from original pipeline)."""
    real_world_label_arr = []
    for old_label in label_arr:
        if old_label in real_world_name_map:
            real_world_label_arr.append(real_world_name_map[old_label])
        else:
            real_world_label_arr.append(old_label)
    
    ctr_dict = {}
    for i in range(len(real_world_label_arr)):
        label = real_world_label_arr[i]
        if label not in ctr_dict:
            ctr_dict[label] = [i]
        else:
            ctr_dict[label].append(i)
    
    for k, v in ctr_dict.items():
        if len(v) > 1:
            counter = 1
            for idx in v:
                real_world_label_arr[idx] = f'{k}-{counter}'
                counter += 1
    
    return real_world_label_arr


def _empty_detections():
    return {
        'total_detections': 0,
        'labels': np.array([]),
        'scores': np.array([]),
        'pred_masks': np.array([]),
        'bbox_2d': np.array([]),
    }


# ============================================================================
# Scene Graph Construction (3D or 2D)
# ============================================================================

def build_scene_graph_3d(
    rgb: np.ndarray,
    depth: np.ndarray,
    outputs: dict,
    intrinsics: np.ndarray,
    total_points_dict: dict,
    bbox3d_dict: dict,
    distractor_list: list,
) -> tuple:
    """
    Build a 3D scene graph from RGB-D + detections.
    This is the direct equivalent of get_scene_graph() in real_world_get_local_sg.py.
    """
    pcd_dict, bbox2d_dict = {}, {}
    local_sg = SceneGraph()
    
    for idx in range(outputs['total_detections']):
        label = outputs['labels'][idx]
        if label.split("-")[0] in distractor_list:
            continue
        if outputs['scores'][idx] < 0:
            continue
        
        # Depth -> point cloud using mask
        masked_depth = depth * outputs['pred_masks'][idx]
        point_3d = depth_to_point_cloud(intrinsics, masked_depth)
        
        if len(point_3d) == 0:
            continue
        
        # Downsample + denoise
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(point_3d)
        voxel_down_pcd = obj_pcd.voxel_down_sample(voxel_size=0.01)
        
        if len(voxel_down_pcd.points) > 10:
            _, ind = voxel_down_pcd.remove_statistical_outlier(
                nb_neighbors=min(1500, len(voxel_down_pcd.points)),
                std_ratio=0.1
            )
            inlier = voxel_down_pcd.select_by_index(ind)
            pcd_dict[label] = np.array(inlier.points)
        else:
            pcd_dict[label] = np.array(voxel_down_pcd.points)
        
        # Accumulate points for global objects
        if label in ["fridge", "coffee machine", "table"] and label in total_points_dict:
            total_points_dict[label] = np.concatenate((total_points_dict[label], pcd_dict[label]))
        else:
            total_points_dict[label] = pcd_dict[label]
        
        # 3D bounding box
        boxes3d_pts = o3d.utility.Vector3dVector(total_points_dict[label])
        box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)
        bbox3d_dict[label] = box
        bbox2d_dict[label] = outputs['bbox_2d'][idx]
    
    # Build scene graph from 3D data
    local_sg = SceneGraph()
    for label in pcd_dict.keys():
        node = Node(
            name=label,
            object_id=label,
            pos3d=bbox3d_dict[label].get_center(),
            corner_pts=np.array(bbox3d_dict[label].get_box_points()),
            bbox2d=bbox2d_dict[label],
            pcd=total_points_dict[label],
            depth=None,
        )
        local_sg.add_node_wo_edge(node)
        local_sg.add_node(node, rgb)
    
    return local_sg, bbox3d_dict, total_points_dict, bbox2d_dict


def build_scene_graph_2d(
    rgb: np.ndarray,
    outputs: dict,
    distractor_list: list,
) -> tuple:
    """
    Build a 2D scene graph from RGB + detections only (no depth required).
    Uses 2D bounding box heuristics for spatial relationships.
    """
    bbox2d_dict = {}
    local_sg = SceneGraph2D()
    
    for idx in range(outputs['total_detections']):
        label = outputs['labels'][idx]
        if label.split("-")[0] in distractor_list:
            continue
        if outputs['scores'][idx] < 0:
            continue
        
        bbox2d = outputs['bbox_2d'][idx]
        bbox2d_dict[label] = bbox2d
        
        # Create a node with 2D info only
        # pos3d is approximated from bbox center (x, y, 0) — used for relative positioning
        cx = (bbox2d[0] + bbox2d[2]) / 2
        cy = (bbox2d[1] + bbox2d[3]) / 2
        
        node = Node(
            name=label,
            object_id=label,
            pos3d=np.array([cx, cy, 0]),  # pseudo-3D
            corner_pts=np.array([  # pseudo corners from 2D bbox
                [bbox2d[0], bbox2d[1], 0],
                [bbox2d[2], bbox2d[1], 0],
                [bbox2d[0], bbox2d[3], 0],
                [bbox2d[2], bbox2d[3], 0],
                [bbox2d[0], bbox2d[1], 1],
                [bbox2d[2], bbox2d[1], 1],
                [bbox2d[0], bbox2d[3], 1],
                [bbox2d[2], bbox2d[3], 1],
            ]),
            bbox2d=bbox2d,
            pcd=np.array([[cx, cy, 0]]),  # minimal pcd
            depth=None,
        )
        local_sg.add_node_wo_edge(node)
        local_sg.add_node(node, rgb)
    
    return local_sg, bbox2d_dict


# ============================================================================
# Text Serialization (same as original get_scene_text)
# ============================================================================

def get_scene_text(scene_graph: SceneGraph) -> str:
    """Convert a scene graph to natural language text for LLM consumption."""
    output = ""
    for node in scene_graph.nodes:
        node_name = node.name
        if node.state is not None:
            node_name = f"{node_name} ({node.state})"
        output += (node_name + ", ")
    
    if len(scene_graph.nodes) != 0:
        output = output[:-2] + ". "
    
    for edge in scene_graph.edges.values():
        start_node_name = str(edge.start)
        end_node_name = str(edge.end)
        
        # Filter redundant relations (same logic as original)
        if edge.edge_type == 'on the left of':
            if (end_node_name, start_node_name) in scene_graph.edges:
                other = scene_graph.edges[(end_node_name, start_node_name)]
                if other.edge_type == 'on the right of':
                    continue
        if edge.edge_type == 'below':
            if (end_node_name, start_node_name) in scene_graph.edges:
                other = scene_graph.edges[(end_node_name, start_node_name)]
                if other.edge_type == 'above':
                    continue
        if edge.edge_type == 'near':
            if (end_node_name, start_node_name) in scene_graph.edges:
                other = scene_graph.edges[(end_node_name, start_node_name)]
                if other.edge_type in ('on top of', 'inside', 'near'):
                    continue
        
        output += f"{start_node_name} is {edge.edge_type} {end_node_name}. "
    
    return output.strip()


def convert_frame_to_timestamp(frame_idx: int, fps: float = 30.0) -> str:
    """Convert frame index to MM:SS timestamp."""
    seconds = frame_idx / fps
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# Main Pipeline
# ============================================================================

def run_robofail_pipeline(
    video_path: str,
    task_json_path: str,
    output_dir: str,
    object_list: list,
    distractor_list: list = None,
    depth_model: str = "depth_anything_v2",
    sample_rate: int = 30,
    device: str = "cuda:0",
):
    """
    Full scene graph generation pipeline for RoboFail video recordings.
    
    This is the main entry point — equivalent to run_real_world_pipeline()
    but operating on video files instead of zarr sensor recordings.
    
    Args:
        video_path: Path to the RoboFail video file (.mp4)
        task_json_path: Path to the task annotation JSON
        output_dir: Directory for all outputs (scene graphs, summaries, etc.)
        object_list: List of object names to detect (e.g. ["mug", "faucet", "sink"])
        distractor_list: List of object names to ignore in scene graph
        depth_model: "depth_anything_v2", "zoedepth", or None for 2D-only
        sample_rate: Process every N-th frame (default 30 = ~1fps at 30fps video)
        device: CUDA device string
    
    Returns:
        dict with:
            - global_sg: SceneGraph object
            - key_frames: list of key frame indices
            - L1_summary: str (detailed per-keyframe summary)
            - L2_summary: str (action-level summary)
            - sound_events: dict of detected sound events
    """
    if distractor_list is None:
        distractor_list = []
    
    # 1. Load video
    loader = RoboFailVideoLoader(
        video_path=video_path,
        task_json_path=task_json_path,
        output_dir=output_dir,
        depth_model=depth_model,
        device=device,
    )
    loader.setup()
    
    use_3d = (loader._depth_model is not None) and HAS_OPEN3D
    print(f"\n[Pipeline] Mode: {'3D (with depth)' if use_3d else '2D (bbox only)'}")
    
    # 2. Extract audio for sound detection
    sound_events = {}
    sound_det_idx_dict = {}
    if HAS_AUDIOCLIP:
        audio_path = loader.extract_audio()
        if audio_path is not None:
            try:
                detected_sounds = get_sound_events(audio_path=audio_path, volume_thresh=0.04)
                for sound_range, sound_label in detected_sounds.items():
                    frame_idx = int(sound_range[1] * loader.metadata.fps)
                    sound_det_idx_dict[frame_idx] = sound_label
                sound_events = detected_sounds
                print(f"  Detected {len(detected_sounds)} sound events")
            except Exception as e:
                print(f"  Warning: Sound detection failed: {e}")
    
    # 3. Get action boundaries (if available from task JSON)
    interact_actions = loader.get_action_boundaries()
    interact_actions_end_idx = [idx[1] for idx in interact_actions.keys()]
    print(f"  Action segments: {len(interact_actions)}")
    
    # 4. Dense scene graph generation (Level-0)
    print("\n[Pipeline] Generating dense scene graphs...")
    key_frames = []
    total_points_dict, bbox3d_dict = {}, {}
    prev_graph = SceneGraph()
    local_graphs = {}
    all_bbox2d_dicts = {}
    
    # Determine which frames to process
    frame_indices = loader.get_sample_indices(sample_rate)
    # Always include action boundary frames
    for end_idx in interact_actions_end_idx:
        if end_idx not in frame_indices and end_idx < loader.metadata.total_frames:
            frame_indices.append(end_idx)
    # Always include sound event frames
    for sound_idx in sound_det_idx_dict.keys():
        if sound_idx not in frame_indices and sound_idx < loader.metadata.total_frames:
            frame_indices.append(sound_idx)
    frame_indices = sorted(set(frame_indices))
    
    for i, step_idx in enumerate(frame_indices):
        if i % 50 == 0:
            print(f"  Processing frame {step_idx}/{loader.metadata.total_frames} "
                  f"({i+1}/{len(frame_indices)} sampled frames)")
        
        # Read frame
        try:
            frame = loader.get_frame(step_idx)
        except IndexError:
            continue
        
        rgb = frame.rgb
        depth = frame.depth
        
        # Object detection
        outputs = detect_objects_in_frame(rgb, object_list, step_idx)
        
        if outputs['total_detections'] == 0:
            continue
        
        # CLIP confirmation + label editing
        outputs = confirm_detections_clip(rgb, outputs, object_list)
        if outputs['total_detections'] == 0:
            continue
        outputs['labels'] = edit_label(list(outputs['labels']))
        
        # Save detections
        det_path = os.path.join(output_dir, "mdetr_obj_det", "clip_processed_det", f"{step_idx}.pickle")
        with open(det_path, 'wb') as f:
            pickle.dump(outputs, f)
        
        # Build scene graph
        if use_3d and depth is not None:
            local_sg, bbox3d_dict, total_points_dict, bbox2d_dict = build_scene_graph_3d(
                rgb, depth, outputs, loader.intrinsics_matrix,
                total_points_dict, bbox3d_dict, distractor_list,
            )
        else:
            local_sg, bbox2d_dict = build_scene_graph_2d(rgb, outputs, distractor_list)
        
        # NOTE: Gripper state is not available from video.
        # If you have annotations for held objects, add them here:
        # local_sg.edges[("object_name", "robot gripper")] = Edge(...)
        
        # Save local scene graph
        local_graphs[step_idx] = local_sg
        all_bbox2d_dicts[step_idx] = bbox2d_dict
        sg_path = os.path.join(output_dir, "local_graphs", f"local_sg_{step_idx}.pkl")
        with open(sg_path, 'wb') as f:
            pickle.dump(local_sg, f)
        
        # Key frame selection
        # 1. Scene graph changed
        if local_sg != prev_graph:
            if step_idx not in key_frames:
                key_frames.append(step_idx)
                prev_graph = local_sg
        
        # 2. Action boundary
        if step_idx in interact_actions_end_idx:
            if step_idx not in key_frames:
                key_frames.append(step_idx)
        
        # 3. Sound event
        if step_idx in sound_det_idx_dict:
            if step_idx not in key_frames:
                key_frames.append(step_idx)
    
    key_frames = sorted(key_frames)
    print(f"\n  Selected {len(key_frames)} key frames out of {len(frame_indices)} processed")
    
    # 5. Build global scene graph
    print("[Pipeline] Building global scene graph...")
    global_sg = SceneGraph() if use_3d else SceneGraph2D()
    
    # Collect all detected objects across frames for the global graph
    all_labels = set()
    for sg in local_graphs.values():
        for node in sg.nodes:
            all_labels.add(node.name)
    
    if use_3d:
        for label in total_points_dict.keys():
            if label in bbox3d_dict and label in all_bbox2d_dicts.get(key_frames[-1] if key_frames else 0, {}):
                bbox2d = all_bbox2d_dicts[key_frames[-1]][label]
                new_node = Node(
                    name=label, object_id=label,
                    pos3d=bbox3d_dict[label].get_center(),
                    corner_pts=np.array(bbox3d_dict[label].get_box_points()),
                    bbox2d=bbox2d,
                    pcd=total_points_dict[label],
                    global_node=True,
                )
                global_sg.add_node_wo_edge(new_node)
                global_sg.add_node(new_node, rgb)
    else:
        # For 2D mode, use the last key frame's local graph as the global graph
        if key_frames and key_frames[-1] in local_graphs:
            global_sg = local_graphs[key_frames[-1]]
    
    # Save global scene graph
    with open(os.path.join(output_dir, "global_sg.pkl"), 'wb') as f:
        pickle.dump(global_sg, f)
    
    # 6. Generate Level-1 summary (per key frame)
    print("[Pipeline] Generating L1 summary (per key frame)...")
    L1_captions = []
    fps = loader.metadata.fps
    
    for step_idx in key_frames:
        if step_idx == 0:
            continue
        
        caption = ""
        timestamp = convert_frame_to_timestamp(step_idx, fps)
        
        # Add action label if this frame is at an action boundary
        for (start, end), action in interact_actions.items():
            if start <= step_idx <= end:
                caption += f"{timestamp}. Action: {action}."
                break
        
        if not caption:
            caption += f"{timestamp}."
        
        # Add visual observation (scene graph text)
        if step_idx in local_graphs:
            scene_text = get_scene_text(local_graphs[step_idx])
            if scene_text:
                caption += f" Visual observation: {scene_text}"
        
        # Add auditory observation
        if step_idx in sound_det_idx_dict:
            caption += f" Auditory observation: {sound_det_idx_dict[step_idx]}."
        
        caption += "\n"
        
        # Skip duplicate timestamps
        if L1_captions and caption.split(".")[0] == L1_captions[-1].split(".")[0]:
            continue
        
        L1_captions.append(caption)
    
    L1_summary = "".join(L1_captions)
    with open(os.path.join(output_dir, "state_summary_L1.txt"), 'w') as f:
        f.write(L1_summary)
    
    # 7. Generate Level-2 summary (action-level, only at action boundaries)
    print("[Pipeline] Generating L2 summary (action-level)...")
    L2_captions = []
    for caption in L1_captions:
        if "Action:" in caption:
            L2_captions.append(caption.replace("Action", "Goal"))
    
    L2_summary = "".join(L2_captions)
    with open(os.path.join(output_dir, "state_summary_L2.txt"), 'w') as f:
        f.write(L2_summary)
    
    # Save key frames list
    with open(os.path.join(output_dir, "L1_key_frames.txt"), 'w') as f:
        for frame in key_frames:
            f.write(f"{frame}\n")
    
    # Cleanup
    loader.close()
    
    print(f"\n[Pipeline] Done!")
    print(f"  Key frames: {len(key_frames)}")
    print(f"  L1 captions: {len(L1_captions)}")
    print(f"  L2 captions: {len(L2_captions)}")
    print(f"  Output dir: {output_dir}")
    
    return {
        "global_sg": global_sg,
        "local_graphs": local_graphs,
        "key_frames": key_frames,
        "L1_summary": L1_summary,
        "L2_summary": L2_summary,
        "sound_events": sound_events,
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def config_parser():
    parser = ArgumentParser("RoboFail Scene Graph Pipeline")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to RoboFail video file (.mp4)")
    parser.add_argument("--task_json", type=str, default=None,
                        help="Path to task annotation JSON")
    parser.add_argument("--output_dir", type=str, default="robofail_output",
                        help="Output directory")
    parser.add_argument("--object_list", type=str, required=True,
                        help="Comma-separated list of objects to detect")
    parser.add_argument("--distractor_list", type=str, default="",
                        help="Comma-separated list of distractor objects to ignore")
    parser.add_argument("--depth_model", type=str, default="depth_anything_v2",
                        choices=["depth_anything_v2", "zoedepth", "none"],
                        help="Monocular depth model (or 'none' for 2D-only)")
    parser.add_argument("--sample_rate", type=int, default=30,
                        help="Process every N-th frame")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


if __name__ == "__main__":
    args = config_parser().parse_args()
    
    object_list = [o.strip() for o in args.object_list.split(",")]
    distractor_list = [d.strip() for d in args.distractor_list.split(",") if d.strip()]
    depth_model = None if args.depth_model == "none" else args.depth_model
    
    results = run_robofail_pipeline(
        video_path=args.video_path,
        task_json_path=args.task_json,
        output_dir=args.output_dir,
        object_list=object_list,
        distractor_list=distractor_list,
        depth_model=depth_model,
        sample_rate=args.sample_rate,
        device=args.device,
    )
    
    print("\n=== Global Scene Graph ===")
    print(results["global_sg"])
    print("\n=== L2 Summary (Action-Level) ===")
    print(results["L2_summary"])
