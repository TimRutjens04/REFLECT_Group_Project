"""
Manual EEF tip annotation tool for camera-robot extrinsics calibration.

For each displayed frame: click the gripper contact point (bottom-center of
the metallic gripper box, where fingers touch objects). Press SPACE/ENTER to
confirm and advance. Press 's' to skip a frame. Press 'q'/ESC to quit early.

Saves annotations to pipeline/eef_annotations.json, then solves for
T_cam_robot via SVD and writes it to pipeline/T_cam_robot.npy.

Usage:
    poetry run python3 pipeline/annotate_eef.py
    poetry run python3 pipeline/annotate_eef.py --frames 16
    poetry run python3 pipeline/annotate_eef.py --solve-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import cv2
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader.rgbd_loader import VideoRgbdFrameProvider
from data_loader.task_loader import TaskLoader

# RealSense D435i intrinsics (from REFLECT constants)
K = np.array([
    [914.27246, 0.0,       647.0733 ],
    [0.0,       913.2658,  356.32526],
    [0.0,       0.0,       1.0      ],
], dtype=np.float64)

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "example_data" / "real_data" / "putAppleBowl1"
VIDEO_PATH = DATA_DIR / "videos" / "color.mp4"
ZARR_PATH  = DATA_DIR / "replay_buffer.zarr"
ANN_PATH   = Path(__file__).parent / "eef_annotations.json"
OUT_PATH   = Path(__file__).parent / "T_cam_robot.npy"


def _sample_frames(eef: np.ndarray, n: int) -> list[int]:
    """Farthest-point sampling in robot XYZ → maximally diverse EEF poses."""
    selected = [0]
    for _ in range(n - 1):
        dists = np.min(
            np.linalg.norm(eef[:, None, :] - eef[selected][None, :, :], axis=2),
            axis=1,
        )
        selected.append(int(np.argmax(dists)))
    return sorted(selected)


def annotate(n_frames: int) -> list[dict]:
    zr  = zarr.open_group(str(ZARR_PATH), mode="r")
    eef = np.array(zr["data/robot_eef_pose"][:, :3])
    ts  = np.array(zr["data/timestamp"][:]) - zr["data/timestamp"][0]

    frame_ids = _sample_frames(eef, n_frames)

    task = TaskLoader(ROOT / "example_data").get(1)
    provider = VideoRgbdFrameProvider(task)
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    annotations: list[dict] = []
    click: list[tuple[int, int]] = []

    buf: list[np.ndarray] = [None]

    def on_mouse(event, x, y, flags, _param):
        if buf[0] is None:
            return
        view = buf[0].copy()
        cv2.rectangle(view, (0, view.shape[0] - 30), (220, view.shape[0]), (0, 0, 0), -1)
        cv2.putText(view, f"x={x}  y={y}", (6, view.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if event == cv2.EVENT_LBUTTONDOWN:
            click.clear()
            click.append((x, y))
            cv2.drawMarker(view, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            buf[0] = view
        cv2.imshow("Annotate EEF tip", view)

    cv2.namedWindow("Annotate EEF tip", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate EEF tip", 1280, 720)

    for i, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, bgr = cap.read()
        if not ret:
            continue

        try:
            depth_mm = provider._read_depth(fid)  # float32, values in mm
        except RuntimeError:
            depth_mm = None

        eef_xyz = eef[fid]
        display = bgr.copy()

        lines = [
            f"Frame {i+1}/{len(frame_ids)}  idx={fid}  t={ts[fid]:.1f}s",
            f"EEF robot XYZ: ({eef_xyz[0]:.3f}, {eef_xyz[1]:.3f}, {eef_xyz[2]:.3f}) m",
            "CLICK: bottom-center of the metal gripper box (contact point)",
            "SPACE/ENTER=confirm  s=skip  q/ESC=quit",
        ]
        for li, line in enumerate(lines):
            cv2.putText(display, line, (10, 28 + li * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        click.clear()
        buf[0] = display
        cv2.setMouseCallback("Annotate EEF tip", on_mouse)
        cv2.imshow("Annotate EEF tip", display)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord(' '), 13):
                if click:
                    break
                warn = display.copy()
                cv2.putText(warn, "Click the gripper tip first!", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("Annotate EEF tip", warn)
            elif key == ord('s'):
                click.clear()
                break
            elif key in (ord('q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                return annotations

        if not click:
            continue

        u, v    = click[0]
        depth_m = None
        if depth_mm is not None:
            # Gripper metal surfaces cause IR depth holes — use median of valid patch.
            R   = 8
            y0, y1 = max(0, v - R), min(depth_mm.shape[0], v + R + 1)
            x0, x1 = max(0, u - R), min(depth_mm.shape[1], u + R + 1)
            patch  = depth_mm[y0:y1, x0:x1] * 0.001  # mm → m
            valid  = patch[(patch > 0.1) & (patch < 5.0)]
            if valid.size > 0:
                depth_m = float(np.median(valid))

        ann = {
            "frame_id":  fid,
            "timestamp": float(ts[fid]),
            "pixel_uv":  [u, v],
            "eef_robot": eef_xyz.tolist(),
            "depth_m":   depth_m,
        }
        annotations.append(ann)
        print(f"  frame {fid}: pixel=({u},{v})  depth={depth_m}  EEF={eef_xyz.round(3)}")

    cap.release()
    cv2.destroyAllWindows()
    return annotations


def solve(annotations: list[dict]) -> np.ndarray:
    """
    Solve R, t such that P_cam ≈ R @ P_robot + t via SVD (Procrustes).

    P_cam[i]   = back-projected pixel (u,v) at depth d using intrinsics K
    P_robot[i] = eef_robot XYZ from zarr (robot base frame, metres)
    """
    P_cam, P_robot = [], []
    for ann in annotations:
        if ann["depth_m"] is None:
            print(f"  skip frame {ann['frame_id']}: no valid depth")
            continue
        u, v  = ann["pixel_uv"]
        d     = ann["depth_m"]
        x_cam = (u - K[0, 2]) * d / K[0, 0]
        y_cam = (v - K[1, 2]) * d / K[1, 1]
        P_cam.append([x_cam, y_cam, d])
        P_robot.append(ann["eef_robot"])

    if len(P_cam) < 3:
        raise ValueError(f"Need ≥ 3 valid correspondences, got {len(P_cam)}")

    Pc = np.array(P_cam,   dtype=np.float64)
    Pr = np.array(P_robot, dtype=np.float64)

    c_cam   = Pc.mean(axis=0)
    c_robot = Pr.mean(axis=0)
    H       = (Pr - c_robot).T @ (Pc - c_cam)
    U, _, Vt = np.linalg.svd(H)
    R        = Vt.T @ U.T
    if np.linalg.det(R) < 0:       # correct reflection
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_cam - R @ c_robot

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t

    residuals = np.linalg.norm((R @ Pr.T).T + t - Pc, axis=1)
    print(f"\nResiduals (m): mean={residuals.mean():.4f}  max={residuals.max():.4f}")
    print(f"N correspondences used: {len(P_cam)}")
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames",     type=int,  default=12)
    parser.add_argument("--solve-only", action="store_true")
    args = parser.parse_args()

    if args.solve_only:
        if not ANN_PATH.exists():
            raise FileNotFoundError(f"No annotations at {ANN_PATH}. Run without --solve-only first.")
        annotations = json.loads(ANN_PATH.read_text())
    else:
        print(f"Opening annotation window for {args.frames} frames...")
        annotations = annotate(args.frames)
        ANN_PATH.write_text(json.dumps(annotations, indent=2))
        print(f"Saved {len(annotations)} annotations → {ANN_PATH}")

    if len(annotations) < 3:
        print("Need ≥ 3 annotations to solve. Exiting.")
        return

    T = solve(annotations)
    np.save(str(OUT_PATH), T)
    print(f"\nT_cam_robot:\n{T.round(4)}")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
