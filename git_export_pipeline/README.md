This folder contains the minimal reproducible export for the simulated-data pipeline.

Included:
- `BoilWater2 Performance.ipynb`
- `build_sim_artifacts.py`
- `extract_video.py`
- `Requirements 1.txt`
- `sim_results_boilwater.json`
- `APPLE IN FRIDGE 1.xlsx`
- `Experiment.docx`
- `Data/Data/Sim Data/boilWater-1/task.json`
- `Data/Data/Sim Data/boilWater-1/frames/`

Not included:
- `yoloe-11l-seg.pt`
- `mobileclip_blt.ts`
- Hugging Face model cache files
- Full raw sim datasets outside `boilWater-1/frames`

To fully rerun the notebook, the environment must install the Python dependencies and download/load the model weights used by Grounding DINO, OWL-ViT, and YOLOE.
