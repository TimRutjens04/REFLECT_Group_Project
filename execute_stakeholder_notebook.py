from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.validator import normalize

import jupyter_core.paths


WORKDIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = WORKDIR / "Stakeholder_ClosedVocab_Justification.ipynb"
HTML_PATH = WORKDIR / "Stakeholder_ClosedVocab_Justification.html"


def main() -> None:
    # Work around Windows ACL issues when Jupyter writes its temporary connection file.
    jupyter_core.paths.win32_restrict_file_to_user = lambda fname: None

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    _, notebook = normalize(notebook)

    executor = ExecutePreprocessor(timeout=5400, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(WORKDIR)}})

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    exporter = HTMLExporter()
    body, _ = exporter.from_notebook_node(notebook)
    HTML_PATH.write_text(body, encoding="utf-8")

    print(NOTEBOOK_PATH)
    print(HTML_PATH)


if __name__ == "__main__":
    main()
