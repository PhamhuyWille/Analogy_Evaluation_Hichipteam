# Video Analogy Evaluation

This folder contains the video-track scoring pipeline. It evaluates
educational animations with three independent metrics: Visual Alignment (VA),
Visual Clarity (VC), and Visual Engagement (VE).

## Folder Structure

```text
video/
  src/
    VA.py             # VA scorer, Gemini 3.1 Pro Preview
    VC.py             # VC scorer, Gemini 3.5 Flash Lite
    VE.py             # VE scorer, Gemini 3.5 Flash Lite
    pipeline.py       # Run all three scorers sequentially
  requirements.txt    # Python dependencies
  results/            # Retained result files for the selected report runs
```

## Installation

```bash
cd ..
python -m venv .venv
.venv\Scripts\activate
pip install -r video\requirements.txt
```

Shared setup and dataset download instructions are in the root
[`README.md`](../README.md).

## Configuration

Create `.env` from `.env.example` **at the project root** and fill in your
Gemini API key. The video scripts auto-discover the root `.env` — no need to
copy it into subdirectories.

```bash
copy .env.example .env
# Edit .env → set GEMINI_API_KEY=your_real_key
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## Running the Video Scorers

Run from the `video/` folder.

Visual Alignment:

```bash
python src/VA.py
```

Visual Clarity:

```bash
python src/VC.py
```

Visual Engagement:

```bash
python src/VE.py
```

Run all three sequentially:

```bash
python src/pipeline.py
```

## Submission Columns

All generated submission files use this column order:

```text
id,TCC,MS,M,VC,VA,VE
```
