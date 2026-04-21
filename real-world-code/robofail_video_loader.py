"""
RoboFail Video Data Loader
===========================
Extracts RGB frames, audio, and estimates depth from RoboFail dataset video files.
Replaces the zarr-based real-world data loading with video-based loading.

Dependencies:
    pip install opencv-python-headless numpy torch torchvision torchaudio
    pip install transformers  # for Depth Anything V2

Optional:
    pip install ffmpeg-python  # for robust audio extraction
"""

import os
import cv2
import json
import numpy as np
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoMetadata:
    """Stores extracted metadata about a RoboFail video recording."""
    video_path: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration_sec: float
    has_audio: bool = False
    audio_path: Optional[str] = None


@dataclass
class FrameData:
    """Single frame of data, analogous to what the real-world pipeline reads from zarr."""
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32, in mm (to match real-world pipeline)
    frame_idx: int = 0
    timestamp_sec: float = 0.0


class RoboFailVideoLoader:
    """
    Loads video data from RoboFail dataset recordings and provides
    frame-by-frame access compatible with the real-world scene graph pipeline.
    
    Usage:
        loader = RoboFailVideoLoader(
            video_path="path/to/video.mp4",
            task_json_path="path/to/task.json",  # optional
            output_dir="robofail_output/task_name",
            depth_model="depth_anything_v2",  # or "zoedepth" or None
        )
        loader.setup()
        
        # Get a specific frame
        frame = loader.get_frame(100)
        rgb, depth = frame.rgb, frame.depth
        
        # Extract audio for AudioCLIP
        audio_path = loader.extract_audio()
        
        # Get frame indices at regular intervals
        indices = loader.get_sample_indices(sample_rate=30)
    """
    
    def __init__(
        self,
        video_path: str,
        task_json_path: Optional[str] = None,
        output_dir: str = "robofail_output",
        depth_model: str = "depth_anything_v2",
        device: str = "cuda:0",
    ):
        self.video_path = video_path
        self.task_json_path = task_json_path
        self.output_dir = output_dir
        self.depth_model_name = depth_model
        self.device = device
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._depth_model = None
        self._depth_transform = None
        self.metadata: Optional[VideoMetadata] = None
        self.task_info: Optional[dict] = None
        
        # Pseudo-intrinsics for estimated depth (will be set based on video resolution)
        # These are approximate and calibrated for monocular depth estimation
        self.intrinsics_matrix: Optional[np.ndarray] = None
    
    def setup(self):
        """Initialize video capture, load task info, and optionally load depth model."""
        # Open video
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        
        # Extract metadata
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.metadata = VideoMetadata(
            video_path=self.video_path,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            duration_sec=total_frames / fps if fps > 0 else 0,
        )
        
        # Generate pseudo camera intrinsics based on video resolution
        # Using reasonable defaults (focal length ~ width, principal point ~ center)
        # NOTE: These are approximations. If your sim has known camera params, override this.
        fx = fy = width  # common assumption for ~90° FoV
        cx, cy = width / 2.0, height / 2.0
        self.intrinsics_matrix = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ])
        
        # Load task info if provided
        if self.task_json_path and os.path.exists(self.task_json_path):
            with open(self.task_json_path, 'r') as f:
                self.task_info = json.load(f)
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "local_graphs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "mdetr_obj_det", "det"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "mdetr_obj_det", "clip_processed_det"), exist_ok=True)
        
        # Load depth estimation model
        if self.depth_model_name is not None:
            self._load_depth_model()
        
        print(f"[RoboFailVideoLoader] Loaded video: {self.video_path}")
        print(f"  Frames: {total_frames}, FPS: {fps:.1f}, Resolution: {width}x{height}")
        print(f"  Duration: {self.metadata.duration_sec:.1f}s")
        print(f"  Depth model: {self.depth_model_name or 'None (2D-only mode)'}")
    
    def _load_depth_model(self):
        """Load monocular depth estimation model."""
        import torch
        
        if self.depth_model_name == "depth_anything_v2":
            try:
                from transformers import pipeline
                self._depth_model = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=self.device if torch.cuda.is_available() else "cpu",
                )
                print("  Loaded Depth Anything V2 (Small)")
            except Exception as e:
                print(f"  Warning: Could not load Depth Anything V2: {e}")
                print("  Falling back to 2D-only mode (no depth)")
                self._depth_model = None
                self.depth_model_name = None
                
        elif self.depth_model_name == "zoedepth":
            try:
                import torch
                self._depth_model = torch.hub.load(
                    "isl-org/ZoeDepth", "ZoeD_N", pretrained=True
                ).to(self.device).eval()
                print("  Loaded ZoeDepth")
            except Exception as e:
                print(f"  Warning: Could not load ZoeDepth: {e}")
                self._depth_model = None
                self.depth_model_name = None
    
    def get_frame(self, frame_idx: int) -> FrameData:
        """
        Read a specific frame from the video and optionally estimate depth.
        
        Args:
            frame_idx: 0-based frame index
            
        Returns:
            FrameData with rgb (H,W,3 uint8) and depth (H,W float32 in mm, or None)
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr_frame = self._cap.read()
        if not ret:
            raise IndexError(f"Could not read frame {frame_idx}")
        
        # Convert BGR -> RGB (the real-world pipeline expects RGB)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # Estimate depth
        depth = None
        if self._depth_model is not None:
            depth = self._estimate_depth(rgb)
        
        return FrameData(
            rgb=rgb,
            depth=depth,
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / self.metadata.fps if self.metadata.fps > 0 else 0,
        )
    
    def _estimate_depth(self, rgb: np.ndarray) -> np.ndarray:
        """
        Estimate depth from an RGB image using the loaded monocular depth model.
        
        Returns depth in millimeters (float32) to match the real-world pipeline's
        convention (depth_to_point_cloud expects mm-scale depth).
        
        NOTE: Monocular depth is up-to-scale. The absolute values are approximate.
        For scene graph purposes (relative distances), this is usually sufficient.
        """
        import torch
        from PIL import Image
        
        if self.depth_model_name == "depth_anything_v2":
            pil_img = Image.fromarray(rgb)
            result = self._depth_model(pil_img)
            depth_map = np.array(result["depth"], dtype=np.float32)
            # Resize to match input resolution
            if depth_map.shape[:2] != rgb.shape[:2]:
                depth_map = cv2.resize(depth_map, (rgb.shape[1], rgb.shape[0]))
            # Depth Anything outputs relative depth (0-1 range typically).
            # Scale to approximate metric depth in mm.
            # This scaling factor is approximate — tune based on your sim's camera setup.
            depth_mm = depth_map * 1000.0  # rough scaling to mm
            
        elif self.depth_model_name == "zoedepth":
            pil_img = Image.fromarray(rgb)
            with torch.no_grad():
                depth_map = self._depth_model.infer_pil(pil_img)
            depth_map = np.array(depth_map, dtype=np.float32)
            depth_mm = depth_map * 1000.0  # ZoeDepth outputs meters, convert to mm
        else:
            return None
        
        return depth_mm
    
    def extract_audio(self, output_filename: str = "audio.wav") -> Optional[str]:
        """
        Extract audio track from video using ffmpeg.
        Returns path to extracted .wav file, or None if no audio / extraction fails.
        """
        audio_path = os.path.join(self.output_dir, output_filename)
        
        if os.path.exists(audio_path):
            self.metadata.has_audio = True
            self.metadata.audio_path = audio_path
            return audio_path
        
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", self.video_path,
                    "-vn",  # no video
                    "-acodec", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "1",  # mono
                    audio_path,
                    "-y",  # overwrite
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and os.path.exists(audio_path):
                self.metadata.has_audio = True
                self.metadata.audio_path = audio_path
                print(f"  Extracted audio to: {audio_path}")
                return audio_path
            else:
                print(f"  Warning: ffmpeg audio extraction failed: {result.stderr[:200]}")
                return None
        except FileNotFoundError:
            print("  Warning: ffmpeg not found. Install ffmpeg for audio extraction.")
            return None
        except subprocess.TimeoutExpired:
            print("  Warning: ffmpeg timed out during audio extraction.")
            return None
    
    def get_sample_indices(self, sample_rate: int = 30) -> list:
        """
        Get frame indices sampled at a regular interval.
        Analogous to iterating over zarr frames in the real-world pipeline.
        
        Args:
            sample_rate: Sample every N frames (default 30 = ~1 per second at 30fps)
        """
        return list(range(0, self.metadata.total_frames, sample_rate))
    
    def get_action_boundaries(self) -> dict:
        """
        Extract action stage boundaries from task JSON annotations.
        
        Returns a dict mapping (start_frame, end_frame) -> action_name,
        analogous to get_interact_actions() in the real-world pipeline.
        
        If no task JSON or no action annotations, returns empty dict.
        """
        if self.task_info is None:
            return {}
        
        # RoboFail dataset format — adapt these keys to match your actual JSON structure
        actions = {}
        
        # Try common annotation formats
        if "actions" in self.task_info and "stages" in self.task_info:
            # Format: stages dict maps stage_id -> [start, end], actions list
            stages = self.task_info["stages"]
            action_list = self.task_info["actions"]
            for stage_id, (start, end) in stages.items():
                stage_idx = int(stage_id)
                if stage_idx < len(action_list):
                    action_name = action_list[stage_idx]
                    if action_name != "Terminate":
                        actions[(start, end)] = action_name
                        
        elif "action_segments" in self.task_info:
            # Format: list of {action, start_frame, end_frame}
            for seg in self.task_info["action_segments"]:
                actions[(seg["start_frame"], seg["end_frame"])] = seg["action"]
        
        elif "actions" in self.task_info:
            # Fallback: evenly divide video across actions
            action_list = self.task_info["actions"]
            n_actions = len([a for a in action_list if a != "Terminate"])
            if n_actions > 0:
                frames_per_action = self.metadata.total_frames // n_actions
                for i, action in enumerate(action_list):
                    if action == "Terminate":
                        continue
                    start = i * frames_per_action
                    end = min((i + 1) * frames_per_action - 1, self.metadata.total_frames - 1)
                    actions[(start, end)] = action
        
        return actions
    
    def frame_to_timestamp(self, frame_idx: int) -> str:
        """Convert frame index to MM:SS timestamp string."""
        seconds = frame_idx / self.metadata.fps if self.metadata.fps > 0 else 0
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def close(self):
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __del__(self):
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
