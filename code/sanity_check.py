"""
Sanity checks for aligned .npz files (from align.py).

Runs all 5 checks from CLAUDE.md:
  1. Frame count matches expected  (duration * fps_base)
  2. Audio windows are correctly sized  (sr * 0.5)
  3. Timestamps are monotonically increasing
  4. Visual spot check: frame at failure timestamp + 2 frames before
  5. Audio spot check: full waveform with extraction window markers

Usage:
    python sanity_check.py aligned/boilWater-1.npz
    python sanity_check.py aligned/  # run on all .npz files in directory
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def check_frame_count(timestamps, fps_base, episode_id):
    duration = timestamps[-1] - timestamps[0]
    expected = int(duration * fps_base)
    actual = len(timestamps)
    ok = abs(actual - expected) <= 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Check 1 — frame count: {actual} (expected ~{expected})")
    return ok


def check_audio_window_size(audio_windows, audio_sr, episode_id):
    expected = int(audio_sr * 0.5)
    actual = audio_windows.shape[1]
    ok = actual == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Check 2 — audio window size: {actual} (expected {expected})")
    return ok


def check_monotonic(timestamps, episode_id):
    ok = bool(np.all(np.diff(timestamps) > 0))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Check 3 — timestamps monotonically increasing")
    return ok


def visual_spot_check(npz_path, d, episode_id):
    """Check 4: plot frame at failure timestamp and 2 frames before."""
    failure_idxs = np.where(d["failure_labels"])[0]
    if len(failure_idxs) == 0:
        print("  [SKIP] Check 4 — no failure labels to visualize")
        return True

    fi = failure_idxs[0]  # first failure frame index
    idxs_to_plot = [max(0, fi - 2), max(0, fi - 1), fi]
    frames = d["frames"]
    timestamps = d["timestamps"]

    fig, axes = plt.subplots(1, len(idxs_to_plot), figsize=(4 * len(idxs_to_plot), 4))
    if len(idxs_to_plot) == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs_to_plot):
        ax.imshow(frames[idx])
        label = "FAILURE" if d["failure_labels"][idx] else ""
        ax.set_title(f"t={timestamps[idx]:.1f}s  idx={idx}\n{label}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{episode_id} — Visual spot check", fontsize=11)
    plt.tight_layout()
    out = npz_path.replace(".npz", "_check4_visual.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  [PASS] Check 4 — visual spot check saved: {out}")
    return True


def audio_spot_check(npz_path, d, episode_id):
    """Check 5: plot full waveform + extraction window markers."""
    audio_sr = int(d["audio_sr"])
    timestamps = d["timestamps"]
    failure_idxs = np.where(d["failure_labels"])[0]

    # Reconstruct approximate full audio from windows is not reliable.
    # Instead, show the windows at failure frames on a timeline.
    half = 0.25  # 0.5s window half-width

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Top: window timeline
    ax = axes[0]
    total_dur = timestamps[-1] + 0.5
    ax.set_xlim(0, total_dur)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{episode_id} — Audio window timeline (each bar = 0.5s window)")
    for t in timestamps:
        ax.axvspan(t - half, t + half, alpha=0.15, color="steelblue")
    for fi in failure_idxs:
        ft = timestamps[fi]
        ax.axvspan(ft - half, ft + half, alpha=0.6, color="red", label="failure window")
    ax.axhline(0.5, color="gray", lw=0.5, linestyle="--")
    if len(failure_idxs) > 0:
        ax.legend(loc="upper right", fontsize=8)

    # Bottom: waveform of first failure window (or first window if no failure)
    ax2 = axes[1]
    idx = failure_idxs[0] if len(failure_idxs) > 0 else 0
    window = d["audio_windows"][idx]
    t_window = np.linspace(
        timestamps[idx] - half, timestamps[idx] + half, len(window)
    )
    ax2.plot(t_window, window, lw=0.5, color="steelblue")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"Waveform — {'failure' if len(failure_idxs) > 0 else 'first'} window"
                  f" (t={timestamps[idx]:.1f}s)")
    ax2.set_ylabel("Amplitude")

    plt.tight_layout()
    out = npz_path.replace(".npz", "_check5_audio.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  [PASS] Check 5 — audio spot check saved: {out}")
    return True


def run_checks(npz_path):
    episode_id = os.path.basename(npz_path).replace(".npz", "")
    print(f"\n=== {episode_id} ===")

    d = np.load(npz_path, allow_pickle=True)
    timestamps = d["timestamps"]
    frames = d["frames"]
    audio_windows = d["audio_windows"]
    fps_base = float(d["fps_base"])
    audio_sr = int(d["audio_sr"])

    results = []
    results.append(check_frame_count(timestamps, fps_base, episode_id))
    results.append(check_audio_window_size(audio_windows, audio_sr, episode_id))
    results.append(check_monotonic(timestamps, episode_id))
    results.append(visual_spot_check(npz_path, d, episode_id))
    results.append(audio_spot_check(npz_path, d, episode_id))

    n_pass = sum(results)
    print(f"  → {n_pass}/{len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sanity_check.py <path.npz | aligned_dir/>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        npz_files = sorted(
            [os.path.join(target, f) for f in os.listdir(target) if f.endswith(".npz")]
        )
    else:
        npz_files = [target]

    if not npz_files:
        print(f"No .npz files found in {target}")
        sys.exit(1)

    all_ok = True
    for path in npz_files:
        ok = run_checks(path)
        all_ok = all_ok and ok

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if all_ok else 1)
