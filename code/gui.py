"""Live Embedding Visualizer — entry point."""
import matplotlib

matplotlib.use("Agg")
from gui.ui import main

if __name__ == "__main__":
    main()
