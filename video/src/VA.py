# VA higest score

import json
import os
import re
import time
from collections import Counter

import pandas as pd
from datasets import load_from_disk
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils import score_submission, setup_run_logger

# Load .env from cwd, video/, or project root (whichever is found first).
_script_dir = Path(__file__).resolve().parent
for _candidate in [Path.cwd() / ".env", _script_dir / ".env", _script_dir.parent / ".env", _script_dir.parents[1] / ".env"]:
    if _candidate.exists():
        load_dotenv(dotenv_path=_candidate, override=True)
        break
else:
    load_dotenv()

SPLIT = "test"
SPLITS_PATH = "challenge-dataset"
OUT_PATH = f"output/submission_VA_video_{SPLIT}.csv"

MODEL = "gemini-3.1-pro-preview"

# Number of independent scoring passes; scores aggregated via majority vote,
# ties broken toward the lower score (matches the v3 VC approach).
N_SAMPLES = 1

TEXT_METRICS = ["TCC", "MS", "M"]
VIDEO_METRICS = ["VC", "VA", "VE"]

MODEL_SCORED_METRICS = ["VA"]
PLACEHOLDER_VIDEO_METRICS = [m for m in VIDEO_METRICS if m not in MODEL_SCORED_METRICS]

DEFAULT_TEXT_SCORE = 0
DEFAULT_VIDEO_SCORE = 0

# Which ground-truth column in the validation split holds the VA label.
VA_LABEL_COL = "VA"
ANCHOR_SCORES = (0, 1, 2, 3)

RUBRIC = """You are a calibrated human annotator scoring educational animations for a CS analogy evaluation benchmark. Your task is to assign a VISUAL ALIGNMENT (VA) score.

═══════════════════════════════════════════════════════
CRITICAL CALIBRATION — READ BEFORE SCORING
═══════════════════════════════════════════════════════
Real annotators find the following distribution across student animations:
  ~15% score 0 (poor)  |  ~35% score 1 (fair)  |  ~35% score 2 (good)  |  ~15% score 3 (excellent)

DO NOT default to high scores. Many student animations only partially depict the analogy.
A score of 3 (EXCELLENT) requires that EVERY beat is shown correctly — this is rare.
A score of 2 (GOOD) means ≥75% beats shown; any omission of a key beat prevents it.
When uncertain between two scores, choose the LOWER one.

═══════════════════════════════════════════════════════
WHAT VA MEASURES
═══════════════════════════════════════════════════════
VA = how faithfully the animation visualises the CONTENT of the analogy text.
- Does the video show ALL the key concepts/objects/characters named in the text?
- Does the video show them IN THE SAME ORDER as the text introduces them?
- Are there contradictions (video shows something the text explicitly says should NOT happen)?

VA does NOT care about drawing quality, polish, or pacing — those are VC and VE.

═══════════════════════════════════════════════════════
SCORE DECISION TREE  (use strictly)
═══════════════════════════════════════════════════════
After listing the key analogy beats, count how many appear in the video:

  0 = POOR      Fewer than half the beats shown, OR a major contradiction.
                The animation seems unrelated or actively wrong.
  1 = FAIR      Roughly half the beats present. Core idea hinted at, but multiple
                essential steps missing or significantly out of order.
  2 = GOOD      Most beats shown (≥75%) with correct order; 1-2 omitted or
                slightly mis-sequenced. No contradictions.
  3 = EXCELLENT Every beat shown, correctly sequenced, nothing contradicting the text.
                The video is a faithful visual retelling. Reserve for truly complete coverage.

Boundary test: "Can you identify the analogy from the video alone without guessing?"
  — if NO → not a 2 or 3.
  — if the CORE MECHANISM of the analogy is missing → cap at 1.

═══════════════════════════════════════════════════════
REFERENCE EXAMPLES (REAL, HUMAN-LABELED)
═══════════════════════════════════════════════════════
Before the video to score, you will be shown up to four REAL reference videos pulled
from the labeled validation set, each tagged with its true human-assigned VA score
(0, 1, 2, and 3). Use them to calibrate: they are not hypothetical descriptions —
they are actual annotated examples at each severity level. Compare the target video's
beat coverage against these anchors before deciding.
"""


def build_prompt(target, description, analogy):
    return f"""{RUBRIC}
═══════════════════════════════════════════════════════
NOW SCORE THIS VIDEO
═══════════════════════════════════════════════════════
TARGET CONCEPT: {target}

DESCRIPTION: {description}

ANALOGY TEXT (this is the reference — every named concept is a "beat"):
{analogy}

WATCH THE VIDEO (the one after the reference examples above), then reason step-by-step:
  Step 1 — List every distinct beat/concept in the analogy above (numbered).
  Step 2 — For each beat, state: ✓ shown / ✗ missing / ~ partial.
  Step 3 — Note any contradictions (video shows opposite of analogy).
  Step 4 — Count coverage fraction. Is the CORE MECHANISM of the analogy shown?
  Step 5 — Apply the decision tree, calibrating against the reference examples you were shown.
  Step 6 — Assign VA ∈ {{0,1,2,3}} and write a one-sentence rationale.

WARNING: Do not award 3 simply because the video "looks related" to the analogy.
Only award 3 if you can confirm every single beat is visually present.

Respond with JSON only: {{"VA": <integer 0-3>, "rationale": "<one concise sentence>"}}"""


def extract_first_json_object(text):
    text = text.strip()
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in: {text!r}")

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text, start)
    return obj


def parse_scores(text):
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    obj = extract_first_json_object(text)

    if "VA" not in obj:
        raise ValueError(f"Missing metric VA in model output: {obj}")

    score = int(round(float(obj["VA"])))
    score = max(0, min(3, score))

    return {"VA": score}


def resolve_video_path(row):
    video = row["video"]
    if isinstance(video, dict):
        video_path = video.get("path")
    else:
        video_path = video

    if video_path is None:
        raise ValueError(f"Could not resolve video path from row['video']: {video!r}")

    if os.path.isabs(video_path):
        return video_path

    return os.path.join(SPLITS_PATH, video_path)


def upload_and_wait(client, video_path):
    uploaded = client.files.upload(file=video_path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(1.5)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Video upload failed: {uploaded.state}")
    return uploaded


def select_anchor_rows(dataset, score_col=VA_LABEL_COL, scores=ANCHOR_SCORES):
    """Pick the first validation row found for each target VA score."""
    anchors = {}
    for i in range(len(dataset)):
        row = dataset[i]
        s = row.get(score_col)
        if s in scores and s not in anchors:
            anchors[s] = i
        if len(anchors) == len(scores):
            break
    return anchors


def upload_anchor_videos(client, dataset, anchor_rows, logger=None):
    """Upload each anchor video once; reused across every scoring call in this run."""
    anchors = []
    for score in sorted(anchor_rows):
        idx = anchor_rows[score]
        row = dataset[idx]
        video_path = resolve_video_path(row)
        uploaded = upload_and_wait(client, video_path)
        anchors.append(
            {
                "idx": idx,
                "score": score,
                "file": uploaded,
                "target": row["target"],
                "analogy": row["analogy"],
            }
        )
        if logger:
            logger.info("Uploaded anchor for VA=%d (row %d): %s", score, idx, video_path)
    return anchors


def cleanup_anchor_videos(client, anchors, logger=None):
    for a in anchors:
        try:
            client.files.delete(name=a["file"].name)
        except Exception as e:
            if logger:
                logger.warning("Failed to delete anchor file %s: %s", a["file"].name, e)


def build_anchor_contents(anchors, current_idx=None):
    """Interleave anchor videos with a short label caption. Skips an anchor
    that IS the row currently being scored, to avoid leaking the answer."""
    contents = []
    for a in anchors:
        if current_idx is not None and a["idx"] == current_idx:
            continue
        contents.append(a["file"])
        contents.append(
            f"REFERENCE EXAMPLE — ground-truth VA = {a['score']} (this is a real "
            f"human-annotated label, not a guess). Target concept: {a['target']}. "
            f"Analogy text for this reference: {a['analogy']}\n"
            f"Study which beats this video does and does not cover at this score level."
        )
    return contents


def score_one(client, video_path, target, description, analogy, anchors, current_idx=None, max_retries=3, logger=None):
    prompt = build_prompt(target, description, analogy)

    uploaded = upload_and_wait(client, video_path)

    json_schema = {
        "type": "OBJECT",
        "properties": {
            "VA": {"type": "INTEGER", "description": "Visual Alignment score: 0=Poor, 1=Fair, 2=Good, 3=Excellent"},
            "rationale": {"type": "STRING", "description": "One-sentence justification referencing specific beats covered or missing"},
        },
        "required": ["VA", "rationale"],
    }

    anchor_contents = build_anchor_contents(anchors, current_idx=current_idx)

    try:
        samples = []
        for sample_idx in range(N_SAMPLES):
            for attempt in range(max_retries):
                try:
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=[*anchor_contents, uploaded, prompt],
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json",
                            response_schema=json_schema,
                            thinking_config=types.ThinkingConfig(
                                thinking_level="HIGH",
                            ),
                        ),
                    )
                    s = parse_scores(resp.text)
                    samples.append(s["VA"])
                    if logger:
                        logger.info("    sample %d/%d → VA=%d", sample_idx + 1, N_SAMPLES, s["VA"])
                    break
                except Exception as e:
                    if logger is not None:
                        logger.warning("sample %d attempt %s failed: %s", sample_idx + 1, attempt + 1, e)
                    else:
                        print(f"    sample {sample_idx+1} attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)

        if not samples:
            return {"VA": 1}

        counts = Counter(samples)
        max_count = max(counts.values())
        final_va = min(k for k, v in counts.items() if v == max_count)  # tie → conservative
        return {"VA": final_va}

    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


def make_submission_row(example_id, va_result):
    row = {
        "id": example_id,
        "TCC": DEFAULT_TEXT_SCORE,
        "MS": DEFAULT_TEXT_SCORE,
        "M": DEFAULT_TEXT_SCORE,
        "VA": va_result["VA"],
    }

    for metric in PLACEHOLDER_VIDEO_METRICS:
        row[metric] = DEFAULT_VIDEO_SCORE

    return row


def main():
    os.makedirs("output", exist_ok=True)
    logger, log_path = setup_run_logger(MODEL, SPLIT)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    dataset = load_from_disk(SPLITS_PATH)
    test = dataset[SPLIT]

    # Anchors are always drawn from the labeled validation split, even when
    # SPLIT == "test" — the hidden test split has no true labels to anchor from.
    anchor_source = dataset["validation"]
    anchor_rows = select_anchor_rows(anchor_source)
    anchors = upload_anchor_videos(client, anchor_source, anchor_rows, logger=logger)
    logger.info("Anchor rows selected (score -> validation index): %s", anchor_rows)

    rows = []

    logger.info("Loaded %s examples from %s", len(test), SPLITS_PATH)

    try:
        for i in range(len(test)):
            row = test[i]
            video_path = resolve_video_path(row)

            logger.info("[%s/%s] %s (%s)", i + 1, len(test), row["target"], video_path)

            # Only meaningful when scoring the validation split itself: prevents
            # a row from being shown as its own reference example.
            current_idx = i if SPLIT == "validation" else None

            scores = score_one(
                client,
                video_path,
                row["target"],
                row["description"],
                row["analogy"],
                anchors,
                current_idx=current_idx,
                logger=logger,
            )

            submission_row = make_submission_row(i, scores)

            logger.info("    -> %s", submission_row)

            rows.append(submission_row)
    finally:
        cleanup_anchor_videos(client, anchors, logger=logger)

    submission_df = pd.DataFrame(rows)

    submission_df = submission_df[["id", "TCC", "MS", "M", "VC", "VA", "VE"]]

    submission_df.to_csv(OUT_PATH, index=False)

    logger.info("Wrote CSV submission: %s", OUT_PATH)
    logger.info("Preview:\n%s", submission_df.head().to_string(index=False))

    try:
        results = score_submission(OUT_PATH, SPLIT)

        logger.info("%s", "=" * 40)
        logger.info("%10s %8s %8s", "metric", "tau-b", "rho")
        for metric in MODEL_SCORED_METRICS:
            logger.info(
                "%10s %8.3f %8.3f",
                metric,
                results["kendall"][metric],
                results["spearman"][metric],
            )

    except Exception as e:
        logger.info("Skipping local scoring: %s", e)

    logger.info("Completed run. Log path: %s", log_path)


if __name__ == "__main__":
    main()