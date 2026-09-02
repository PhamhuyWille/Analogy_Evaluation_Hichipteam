# Text Analogy Evaluation

This folder contains the text-track judging pipeline used for the research
report. It evaluates long-form analogies with three independent metrics:
Target Concept Coverage (TCC), Mapping Strength (MS), and Metaphoricity (M).

## Folder Structure

```text
text/
  src/
    TCC.py                # TCC scorer, OpenRouter, default model: openai/gpt-4o-mini
    MS.py                 # MS scorer, DashScope, default model: qwen3-next-80b-a3b-thinking
    M.py                  # M scorer, DashScope, default model: qwen3-max
    final.py              # Merge TCC/MS/M scores into results/submission.csv
  requirements.txt        # Python dependencies
  results/                # Kept result files for the selected report runs
```

## Installation

```bash
cd ..
python -m venv .venv
.venv\Scripts\activate
pip install -r text\requirements.txt
```

Shared setup and dataset download instructions are in the root
[`README.md`](../README.md).

## Configuration

Create `.env` from `.env.example` **at the project root** and fill in your API
credentials. TCC uses OpenRouter. MS and M use DashScope workspace routing.
The text scripts auto-discover the root `.env` — no need to copy it into
subdirectories. Do not commit `.env`.

```bash
copy .env.example .env
# Edit .env → set OPENROUTER_API_KEY, DASHSCOPE_API_KEY, DASHSCOPE_WORKSPACE_ID
```

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_WORKSPACE_ID=your_dashscope_workspace_id_here
```

All experiment defaults are hard-coded in the three scorer files so the report
configuration is reproducible:

```text
src/TCC.py -> openai/gpt-4o-mini through OpenRouter
src/MS.py  -> qwen3-next-80b-a3b-thinking through DashScope
src/M.py   -> qwen3-max through DashScope
```

## Running the Text Scorers

Run from the `text/` folder.

Target Concept Coverage:

```bash
python src/TCC.py --input ..\challenge-dataset --split test
```

Mapping Strength:

```bash
python src/MS.py --input ..\challenge-dataset --split test
```

Metaphoricity:

```bash
python src/M.py --input ..\challenge-dataset --split test
```

Merge the three score files into the final submission:

```bash
python src/final.py
```

Outputs are written under `results/`. The checked-in result files are the
selected report outputs only; local datasets, caches, and private environment
files are ignored by Git.

## Submission Columns

All generated submission files use this column order:

```text
id,TCC,MS,M,VC,VA,VE
```
