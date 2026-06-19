# Video + time-series sync dashboard

A synchronized dashboard for the REFLECT perception pipeline. Plays the robot video
with live time-series charts, metric cards, and event badges that update in real time
as the video plays or as you scrub the seek bar.

Everything fits in one viewport — no scrolling. This is a separate dashboard from the
Streamlit app in `../dashboard/`; the two are independent.

## Run

    python "dashboard 2/run.py"

This auto-discovers every sequence under `pipeline/real_world/jsonl/`, bakes them all
into `index.html`, stages each sequence's video under `video/{seq}/color.mp4`, starts a
local HTTP server (with video seek support), and opens the dashboard in your browser.

## What you see

A dropdown in the header switches between sequences in-browser, with no server reload.
Play the video: the header frame counter, the live metric cards, the playhead lines on
the three charts, the scene-graph relation strip, and the event badges all update in
sync. Drag the seek bar and everything jumps to that frame. When a sequence has a
ground-truth failure window it is shaded light-red on the charts (e.g. putAppleBowl1,
01:00–01:09, frames 1800–2070). Sequences missing a module degrade gracefully — e.g.
putPearDrawer1 has no depth log, so the depth panel shows "Depth not available."

## Change the default / add sequences

New sequence folders under `pipeline/real_world/jsonl/` are picked up automatically on
the next build — no code change needed. To change which sequence is selected on first
load, edit `DEFAULT_SEQUENCE` (and `PORT` if needed) at the top of `build.py`, then
rerun. A sequence is included as long as it has `{seq}__detection.jsonl` and
`{seq}__tracking.jsonl`; depth / scene_graph / validation are optional.

## Stop the server

Ctrl-C in the terminal that started it.

## Files

- `run.py`     — build + serve + open browser (range-capable server)
- `build.py`   — discovers all sequences, reads their JSONLs, writes `index.html` with every sequence's data embedded
- `index.html` — generated artifact (committed for convenience)
- `video/{seq}/color.mp4` — each sequence's video symlinked/copied here so the server can serve it (plus a `source.json` per sequence for debugging)
