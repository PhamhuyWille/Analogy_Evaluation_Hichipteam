# Analogy Evaluation Hichipteam

Repository for the analogy evaluation experiments used in the research report.
Shared setup and dataset preparation are documented here. Track-specific
commands are documented inside each track folder.

## Repository Structure

```text
.
├── README.md                # Shared setup and dataset instructions
├── .gitignore
├── challenge-dataset/       # Local downloaded dataset, ignored by Git
├── .env.example             # API keys template (copy to .env)
├── text/                    # Text-track scoring pipeline and results
│   ├── src/                 # TCC, MS, M scorers + merge script
│   ├── results/             # Retained result files
│   ├── README.md
│   └── requirements.txt
├── video/                   # Video-track scoring pipeline and results
│   ├── src/                 # VA, VC, VE scorers + pipeline script
│   ├── results/             # Retained result files
│   ├── README.md
│   └── requirements.txt
├── notebook/                # Exploratory notebooks
│   ├── data.ipynb
│   ├── distribution.ipynb
│   └── submission.ipynb
└── submission/              # Final submission CSVs
```

## Quick Start

```bash
# 1. Create environment
python -m venv .venv
.venv\Scripts\activate
pip install -r text\requirements.txt
pip install -r video\requirements.txt

# 2. Configure API keys
copy .env.example .env
# Edit .env and fill in your real API keys
```

## Data Download

Download the official dataset once at the repository root. The same
`challenge-dataset/` directory is used by both text and video experiments.

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

- Text-track: [`text/README.md`](text/README.md)
- Video-track: [`video/README.md`](video/README.md)
