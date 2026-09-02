# Best score

import json
import os
import re
import statistics
import time

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
OUT_PATH = f"output/submission_VC_video_{SPLIT}.csv"

MODEL = "gemini-3.5-flash-lite"

# Labeled validation set (has true VC) — used as the anchor pool for
# few-shot grounding, regardless of which split we're scoring. When
# scoring a validation video that happens to BE one of the anchors,
# that specific anchor is skipped for that call to avoid leakage.
ANCHOR_PARQUET_PATH = "challenge-dataset/data/validation-00000-of-00001.parquet"
MAX_ANCHORS_PER_SCORE = 2  # was 1 — more topic diversity per severity level

N_SAMPLES = 5
TEMPERATURE = 0.7

TEXT_METRICS = ["TCC", "MS", "M"]
VIDEO_METRICS = ["VC", "VA", "VE"]

MODEL_SCORED_METRICS = ["VC"]
PLACEHOLDER_VIDEO_METRICS = [m for m in VIDEO_METRICS if m not in MODEL_SCORED_METRICS]

DEFAULT_TEXT_SCORE = 0
DEFAULT_VIDEO_SCORE = 0

SCORE_LABELS = {
    0: "Poor",
    1: "Fair",
    2: "Good",
    3: "Excellent",
}

# Official competition rubric, used verbatim (holistic — no invented
# sub-dimension decision tree layered on top of it).
RUBRIC = """Video — Visual Clarity (VC) — scale 0-3
How clearly the visual elements represent the text: whether each object is properly shown and positioned, the text is legible and well placed, and the motions are meaningful.

Good: all objects are clearly represented on screen; text in the animation is readable and correctly placed; motions are purposeful and help illustrate the idea.
Bad: incorrectly drawn objects; text overlaps with visuals or goes out of bounds; distracting or confusing animations; inconsistent or misleading visual design.

Score | Meaning
0 | Poor: visuals are unclear or misleading; major issues with object accuracy, text placement, or motion.
1 | Fair: some parts are understandable, but several visual flaws (e.g. misplaced text, confusing motion) make comprehension difficult.
2 | Good: mostly clear visuals with minor issues in layout, text, or animation that don't seriously hinder understanding.
3 | Excellent: all visuals are clear, accurate, and well-composed; objects, text, and motion work together to enhance understanding.
"""

JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "VC": {"type": "INTEGER", "description": "0=Poor, 1=Fair, 2=Good, 3=Excellent"},
        "rationale": {"type": "STRING", "description": "One or two sentences citing specific object/text/motion issues (or their absence)"},
    },
    "required": ["VC", "rationale"],
}


def load_anchor_pool(logger=None):
    """Load the labeled validation set and pick up to MAX_ANCHORS_PER_SCORE
    videos per true score value, to use as few-shot grounding. Multiple
    anchors per level give the model flaw *variety* instead of one fixed
    exemplar, which is what let v3 overfit to its 4-video anchor set."""
    labels_df = pd.read_parquet(ANCHOR_PARQUET_PATH)
    if "VC" not in labels_df.columns:
        raise ValueError(f"Expected a VC column in {ANCHOR_PARQUET_PATH}, got: {list(labels_df.columns)}")

    anchor_by_score = {}
    for idx, row in labels_df.iterrows():
        score = int(row["VC"])
        anchor_by_score.setdefault(score, [])
        if len(anchor_by_score[score]) < MAX_ANCHORS_PER_SCORE:
            anchor_by_score[score].append(idx)

    if logger:
        dist = labels_df["VC"].value_counts().sort_index()
        logger.info("Empirical VC distribution on labeled validation set: %s", dict(dist))
        logger.info("Anchor pool (score -> validation row idxs): %s", anchor_by_score)
        for score, idxs in anchor_by_score.items():
            if len(idxs) < MAX_ANCHORS_PER_SCORE:
                logger.warning(
                    "Only %d anchor(s) available for VC=%d (wanted %d) — "
                    "labeled validation set doesn't have enough examples at this level.",
                    len(idxs), score, MAX_ANCHORS_PER_SCORE,
                )

    return labels_df, anchor_by_score


def distribution_hint(labels_df):
    dist = labels_df["VC"].value_counts(normalize=True).sort_index()
    parts = [f"{int(score)} ({SCORE_LABELS[int(score)]}): {pct:.0%}" for score, pct in dist.items()]
    return (
        "For calibration: on the labeled validation set for this dataset, the true VC "
        "distribution is roughly " + ", ".join(parts) + ". This is empirical, not a quota to "
        "enforce — judge each video on its own merits — but it tells you these animations "
        "skew toward having real, noticeable flaws. Don't default to a high score out of politeness."
    )


def upload_and_wait(client, video_path):
    uploaded = client.files.upload(file=video_path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(1.5)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Video upload failed for {video_path}: {uploaded.state}")
    return uploaded


def upload_anchor_files(client, labels_df, anchor_by_score, logger=None):
    """Upload each anchor video once; reused for every scoring call.
    Returns {score: [uploaded_file, ...]}."""
    anchor_files = {}
    for score, idxs in anchor_by_score.items():
        anchor_files[score] = []
        for idx in idxs:
            row = labels_df.iloc[idx]
            video_path = resolve_video_path(row)
            if logger:
                logger.info("Uploading anchor video for VC=%d: %s", score, video_path)
            anchor_files[score].append(upload_and_wait(client, video_path))
    return anchor_files


def extract_first_json_object(text):
    text = text.strip()
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in: {text!r}")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text, start)
    return obj


def parse_score(text):
    obj = extract_first_json_object(text)
    if "VC" not in obj:
        raise ValueError(f"Missing metric VC in model output: {obj}")
    score = int(round(float(obj["VC"])))
    return max(0, min(3, score))


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


def build_contents(anchor_by_score, anchor_files, target_row_id, target_uploaded, target, description, analogy, dist_hint):
    contents = [RUBRIC, dist_hint]

    if anchor_files:
        contents.append(
            "Before scoring, study these reference videos that human annotators already scored, "
            "so your scale matches theirs. Different videos at the same score can look different — "
            "focus on the underlying standard, not surface similarity to any one example:"
        )
        for score in sorted(anchor_files.keys()):
            anchor_idxs = anchor_by_score[score]
            files_for_score = anchor_files[score]
            for anchor_idx, anchor_file in zip(anchor_idxs, files_for_score):
                if target_row_id is not None and anchor_idx == target_row_id:
                    continue  # this video IS the current target — skip to avoid leaking the answer
                contents.append(f"Reference — human score VC = {score} ({SCORE_LABELS[score]}):")
                contents.append(anchor_file)

    contents.append(f"""
Now score this NEW video using the same standard as the references above.

TARGET CONCEPT: {target}
DESCRIPTION: {description}
ANALOGY TEXT (context only — VC scores clarity, not content accuracy):
{analogy}

Watch the video, briefly reason about object accuracy, text placement, and motion purposefulness, then respond with JSON only, no markdown fences:
{{"VC": <integer 0-3>, "rationale": "<one to two sentences>"}}
""")
    contents.append(target_uploaded)
    return contents


def score_one(client, video_path, target, description, analogy, target_row_id,
              anchor_by_score, anchor_files, dist_hint, max_retries=3, logger=None):
    target_uploaded = upload_and_wait(client, video_path)

    try:
        samples = []
        n_samples = getattr(score_one, "_n_samples", N_SAMPLES)

        for sample_idx in range(n_samples):
            for attempt in range(max_retries):
                try:
                    contents = build_contents(
                        anchor_by_score, anchor_files, target_row_id, target_uploaded,
                        target, description, analogy, dist_hint,
                    )
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=TEMPERATURE,
                            response_mime_type="application/json",
                            response_schema=JSON_SCHEMA,
                            thinking_config=types.ThinkingConfig(
                                thinking_level="HIGH",
                            ),
                        ),
                    )
                    vc = parse_score(resp.text)
                    samples.append(vc)
                    if logger:
                        logger.info("    sample %d/%d → VC=%d", sample_idx + 1, n_samples, vc)
                    break
                except Exception as e:
                    if logger is not None:
                        logger.warning("sample %d attempt %s failed: %s", sample_idx + 1, attempt + 1, e)
                    else:
                        print(f"    sample {sample_idx+1} attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)

        if not samples:
            return {"VC": 1}

        # Neutral aggregation: median of the 5 samples. With an odd sample
        # count this always lands exactly on one of the observed scores (no
        # rounding ambiguity) and has no directional bias — unlike v3's
        # "tie-break toward lower" rule, which was reverse-engineered from
        # the 12-example validation skew and didn't generalize to test.
        final_vc = int(statistics.median(samples))

        if logger:
            logger.info("    samples=%s -> VC=%d (median)", samples, final_vc)

        return {"VC": final_vc}

    finally:
        try:
            client.files.delete(name=target_uploaded.name)
        except Exception:
            pass


def make_submission_row(example_id, vc_result):
    row = {
        "id": example_id,
        "TCC": DEFAULT_TEXT_SCORE,
        "MS": DEFAULT_TEXT_SCORE,
        "M": DEFAULT_TEXT_SCORE,
        "VC": vc_result["VC"],
    }
    for metric in PLACEHOLDER_VIDEO_METRICS:
        row[metric] = DEFAULT_VIDEO_SCORE
    return row


def main():
    os.makedirs("output", exist_ok=True)
    logger, log_path = setup_run_logger(MODEL, SPLIT)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    labels_df, anchor_by_score = load_anchor_pool(logger=logger)
    dist_hint = distribution_hint(labels_df)

    test = load_from_disk(SPLITS_PATH)[SPLIT]

    if SPLIT == "validation" and len(test) != len(labels_df):
        logger.warning(
            "validation split length (%d) != anchor parquet length (%d) — "
            "row-index alignment between them may be wrong, double check before trusting anchors.",
            len(test), len(labels_df),
        )

    anchor_files = upload_anchor_files(client, labels_df, anchor_by_score, logger=logger)

    rows = []
    logger.info("Loaded %s examples from %s", len(test), SPLITS_PATH)

    try:
        for i in range(len(test)):
            row = test[i]
            video_path = resolve_video_path(row)
            logger.info("[%s/%s] %s (%s)", i + 1, len(test), row["target"], video_path)

            # When scoring the validation split itself, row index i lines up
            # with the anchor parquet's row index (see alignment check above),
            # so we pass it through to skip self-anchoring.
            target_row_id = i if SPLIT == "validation" else None

            scores = score_one(
                client, video_path, row["target"], row["description"], row["analogy"],
                target_row_id, anchor_by_score, anchor_files, dist_hint, logger=logger,
            )
            submission_row = make_submission_row(i, scores)
            logger.info("    -> %s", submission_row)
            rows.append(submission_row)
    finally:
        for files in anchor_files.values():
            for f in files:
                try:
                    client.files.delete(name=f.name)
                except Exception:
                    pass

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
                metric, results["kendall"][metric], results["spearman"][metric],
            )
    except Exception as e:
        logger.info("Skipping local scoring: %s", e)

    logger.info("Completed run. Log path: %s", log_path)


if __name__ == "__main__":
    main()