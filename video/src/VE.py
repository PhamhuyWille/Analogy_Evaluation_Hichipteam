# Final for VE score

import json
import os
import re
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
OUT_PATH = f"output/submission_VE_video_{SPLIT}.csv"

MODEL = "gemini-3.5-flash-lite"

# Number of independent scoring passes; scores aggregated via majority vote.
N_SAMPLES = 3

TEXT_METRICS = ["TCC", "MS", "M"]
VIDEO_METRICS = ["VC", "VA", "VE"]

MODEL_SCORED_METRICS = ["VE"]
PLACEHOLDER_VIDEO_METRICS = [m for m in VIDEO_METRICS if m not in MODEL_SCORED_METRICS]

DEFAULT_TEXT_SCORE = 0
DEFAULT_VIDEO_SCORE = 0

RUBRIC = """You are a calibrated human annotator scoring educational animations for a CS analogy evaluation benchmark. Your task is to assign a VISUAL ENGAGEMENT (VE) score.

═══════════════════════════════════════════════════════
CRITICAL CALIBRATION — READ BEFORE SCORING
═══════════════════════════════════════════════════════
Real annotators distribute scores roughly as follows across student animations:
  ~15% score 0 (poor)  |  ~35% score 1 (fair)  |  ~35% score 2 (good)  |  ~15% score 3 (excellent)
Use the FULL scale. Do not cluster at the low end — videos that are genuinely engaging SHOULD score 2 or 3.
When truly uncertain between two adjacent scores, go with your gut based on overall viewer experience.

═══════════════════════════════════════════════════════
WHAT VE MEASURES
═══════════════════════════════════════════════════════
VE measures how effectively the animation holds the viewer's attention.

GOOD signals (increase score):
  + Appealing color palette: vibrant, harmonious; color guides attention
  + Well-designed shapes/characters that are pleasant to look at
  + Smooth, fluid motion and transitions
  + Varied pacing that keeps the viewer curious
  + Clear visual narrative with a sense of progression

BAD signals (decrease score):
  - Repetitive visuals: same element loops with no change
  - Visual noise: too many competing elements, cluttered screen
  - Dull color: flat grey/white with no contrast signals
  - Abrupt or jittery motion: elements snap in with no transition
  - Dead zones: >3 seconds where nothing on screen changes

═══════════════════════════════════════════════════════
SCORE SCALE
═══════════════════════════════════════════════════════
  0 = LACKS ENGAGEMENT
      Dull or disjointed; the viewer's interest wanes quickly.
      Mostly static, colorless, or so chaotic it is unpleasant to watch.

  1 = SOME MOMENTS OF INTEREST
      There are brief engaging moments, but overall the video suffers from
      low engagement or long stretches of monotony.

  2 = GENERALLY ENGAGING
      The video holds attention for most of its runtime. May have brief dips
      (short static segment, slightly repetitive section) but overall pleasant.

  3 = CONSISTENTLY CAPTIVATES
      The pacing, visuals, and presentation style are highly engaging THROUGHOUT.
      Color, shapes, and motion all work together. No dead zones. Rare.

═══════════════════════════════════════════════════════
CALIBRATED EXAMPLES
═══════════════════════════════════════════════════════

[Example A — VE = 0]
White slide with black text changing every few seconds. One static icon, no animation, no color.
Why 0: Completely static; no engagement signal at all.

[Example B — VE = 1]
A colorful character walks across the screen for the first 10 seconds, then the rest (25 s)
is a static annotated diagram with no movement.
Why 1: One engaging moment early on, but a long monotonous stretch dominates.

[Example C — VE = 2]
Smooth animated transitions throughout most of the video. Pleasant blues and greens.
Near the end, final point shown as static text for ~8 seconds.
Why 2: Generally engaging with good color and motion; brief static dip at end.

[Example D — VE = 3]
Consistent animated storytelling. Elements slide in, bounce, fade with varied motion.
Colors shift meaningfully as concept evolves. No segment feels stuck or repetitive.
Why 3: Captivating from start to finish.
"""

SELF_CORRECTION_BLOCK = """
═══════════════════════════════════════════════════════
SELF-CORRECTION PROTOCOL
═══════════════════════════════════════════════════════
After forming an initial score, actively try to falsify it before committing:
  - If you leaned toward 0 or 1: look again for any 5+ second stretch of genuinely
    smooth motion, appealing color, or narrative progression. If found, does it
    change the balance enough to bump the score up?
  - If you leaned toward 2 or 3: look again for dead zones (>3s no change),
    repetitive loops, or jittery/abrupt transitions you may have glossed over.
    If found, does it drag the score down?
  - Check calibration drift: are you clustering low out of caution rather than
    judgment? Real annotators use 2 and 3 often (70% combined) — don't default
    to 1 just to be "safe."
  - Only change your initial score if the re-check surfaces something concrete
    (a specific moment/timestamp-level observation), not just second-guessing.
"""


def build_prompt(target, description, analogy):
    return f"""{RUBRIC}
{SELF_CORRECTION_BLOCK}
═══════════════════════════════════════════════════════
NOW SCORE THIS VIDEO
═══════════════════════════════════════════════════════
TARGET CONCEPT: {target}

DESCRIPTION: {description}

ANALOGY TEXT (context only — VE is about how engaging the video is to watch, not content):
{analogy}

WATCH THE VIDEO carefully, then reason step-by-step and fill in every field below:

  observations: For each of Color/Shapes, Motion, Dead Zones, Pacing — write one
    short concrete clause describing what you actually saw (e.g. "static diagram
    from 0:12-0:24, no motion").

  initial_score: Your VE score (0-3) based on those observations alone, before
    any self-correction.

  self_correction: Apply the SELF-CORRECTION PROTOCOL above. State explicitly
    whether you found anything that should change the initial score, and why
    or why not.

  VE: Your final score (0-3) after self-correction. May equal initial_score.

  rationale: One concise sentence naming the dominant final engagement strength
    OR weakness.

Respond with JSON only, no other text."""


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

    if "VE" not in obj:
        raise ValueError(f"Missing metric VE in model output: {obj}")

    score = int(round(float(obj["VE"])))
    score = max(0, min(3, score))

    return {
        "VE": score,
        "initial_score": obj.get("initial_score"),
        "self_correction": obj.get("self_correction"),
        "observations": obj.get("observations"),
        "rationale": obj.get("rationale"),
    }


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


def score_one(client, video_path, target, description, analogy, max_retries=3, logger=None):
    prompt = build_prompt(target, description, analogy)

    uploaded = client.files.upload(file=video_path)

    while uploaded.state.name == "PROCESSING":
        time.sleep(1.5)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Video upload failed: {uploaded.state}")

    json_schema = {
        "type": "OBJECT",
        "properties": {
            "observations": {
                "type": "OBJECT",
                "properties": {
                    "color_shapes": {
                        "type": "STRING",
                        "description": "What was observed about color palette and shape/character design",
                    },
                    "motion": {
                        "type": "STRING",
                        "description": "What was observed about motion quality: smooth/varied vs jittery/abrupt/absent",
                    },
                    "dead_zones": {
                        "type": "STRING",
                        "description": "Any stretches >3s where nothing on screen changes, with rough timestamps if possible",
                    },
                    "pacing": {
                        "type": "STRING",
                        "description": "Overall pacing and sense of narrative progression",
                    },
                },
                "required": ["color_shapes", "motion", "dead_zones", "pacing"],
            },
            "initial_score": {
                "type": "INTEGER",
                "description": "VE score (0-3) based on observations alone, before self-correction",
            },
            "self_correction": {
                "type": "STRING",
                "description": "Result of applying the self-correction protocol: what was re-checked, and whether/why the score changed or stayed the same",
            },
            "VE": {
                "type": "INTEGER",
                "description": "Final Visual Engagement score after self-correction: 0=Lacks engagement, 1=Some moments of interest, 2=Generally engaging, 3=Consistently captivates",
            },
            "rationale": {
                "type": "STRING",
                "description": "One sentence naming the dominant final engagement strength or weakness observed",
            },
        },
        "required": ["observations", "initial_score", "self_correction", "VE", "rationale"],
    }

    try:
        samples = []
        for sample_idx in range(N_SAMPLES):
            for attempt in range(max_retries):
                try:
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=[uploaded, prompt],
                        config=types.GenerateContentConfig(
                            temperature=0.8,
                            response_mime_type="application/json",
                            response_schema=json_schema,
                            thinking_config=types.ThinkingConfig(
                                thinking_level="HIGH",
                            ),
                        ),
                    )
                    s = parse_scores(resp.text)
                    samples.append(s["VE"])  # guaranteed int by parse_scores
                    if logger:
                        flipped = (
                            s["initial_score"] is not None
                            and int(s["initial_score"]) != s["VE"]
                        )
                        logger.info(
                            "    sample %d/%d → initial=%s final=VE=%d%s | self_correction: %s",
                            sample_idx + 1,
                            N_SAMPLES,
                            s["initial_score"],
                            s["VE"],
                            " (FLIPPED)" if flipped else "",
                            s["self_correction"],
                        )
                    break
                except Exception as e:
                    if logger is not None:
                        logger.warning("sample %d attempt %s failed: %s", sample_idx+1, attempt + 1, e)
                    else:
                        print(f"    sample {sample_idx+1} attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)

        if not samples:
            return {"VE": 1}

        from collections import Counter
        counts = Counter(samples)
        max_count = max(counts.values())
        final_ve = min(k for k, v in counts.items() if v == max_count)  # tie → conservative
        return {"VE": final_ve}

    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


def make_submission_row(example_id, ve_result):
    row = {
        "id": example_id,
        "TCC": DEFAULT_TEXT_SCORE,
        "MS": DEFAULT_TEXT_SCORE,
        "M": DEFAULT_TEXT_SCORE,
        "VE": ve_result["VE"],
    }

    for metric in PLACEHOLDER_VIDEO_METRICS:
        row[metric] = DEFAULT_VIDEO_SCORE

    return row


def main():
    os.makedirs("output", exist_ok=True)
    logger, log_path = setup_run_logger(MODEL, SPLIT)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    test = load_from_disk(SPLITS_PATH)[SPLIT]

    rows = []

    logger.info("Loaded %s examples from %s", len(test), SPLITS_PATH)

    for i in range(len(test)):
        row = test[i]
        video_path = resolve_video_path(row)

        logger.info("[%s/%s] %s (%s)", i + 1, len(test), row["target"], video_path)

        scores = score_one(
            client,
            video_path,
            row["target"],
            row["description"],
            row["analogy"],
            logger=logger,
        )

        submission_row = make_submission_row(i, scores)

        logger.info("    -> %s", submission_row)

        rows.append(submission_row)

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