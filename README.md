# Analogy Evaluation Hichipteam

Repository for the analogy evaluation experiments used in the research report.
Shared setup and dataset preparation are documented here. Track-specific
commands are documented inside each track folder.

## Repository Structure

```text
.
  README.md              # Shared setup and dataset instructions
  challenge-dataset/     # Local downloaded dataset, ignored by Git
  text/                  # Text-track scoring pipeline and retained results
  video/                 # Video-track workspace, ignored when local-only
```

## Common Setup

Create one Python environment from the repository root. The current runnable
pipeline is the text track, so install its dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r text\requirements.txt
```

## Data Download

Download the official dataset once at the repository root. The same
`challenge-dataset/` directory can be used by both text and video experiments.

```python
from huggingface_hub import snapshot_download
from datasets import load_dataset

LOCAL_DIR = "challenge-dataset"

snapshot_download(
    repo_id="analogy-evaluation/challenge-dataset",
    repo_type="dataset",
    local_dir=LOCAL_DIR,
)

ds = load_dataset(LOCAL_DIR)
ds.save_to_disk(LOCAL_DIR)
```

## Track Instructions

Text-track commands and API configuration are in [`text/README.md`](text/README.md).
