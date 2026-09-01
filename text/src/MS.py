"""
MS.py
=====
Mapping Strength (MS) scoring pipeline for the Long-Form Analogy Evaluation
task. It uses the DashScope OpenAI-compatible API and is tuned for reasoning
models with lower token usage.

Three important implementation details:

1. reasoning.effort controls the hidden reasoning-token budget instead of
   relying only on max_tokens:
       {"model": ..., "reasoning": {"effort": "low"}, ...}
2. lean_output=True asks the model to return only "Score: X" while keeping the
   decomposition implicit.
3. Usage tracking records prompt, completion, and reasoning tokens for each
   call.

The default report model is qwen3-next-80b-a3b-thinking. See README.md for
environment setup and run commands.
"""

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from datasets import load_from_disk

import pandas as pd
from scipy.stats import kendalltau, spearmanr
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit(
        "Missing dependency: openai. Install it with: pip install openai"
    ) from e

DEFAULT_MODELS = [
    "qwen3-next-80b-a3b-thinking",
]
DASHSCOPE_REGION = "ap-southeast-1"

SCORE_CANDIDATES = ["0", "1", "2"]  # MS score scale: 0, 1, 2

# Target concepts used as few-shot examples in build_ms_messages().
# Exclude them from held-out validation correlations to avoid leakage.
MS_FEWSHOT_TARGETS = {
}

REASONING_EFFORT_CHOICES = ["none", "minimal", "low", "medium", "high"]

# ============================================================================
# build_ms_messages
# ============================================================================

def build_ms_messages(
    target: str,
    description: str,
    analogy: str,
    split: str = "validation",
    lean_output: bool = True,
) -> list[dict]:

    if split == "test":
        lean_output = True

    system_msg = (
        "You are a careful evaluator of analogy quality, reasoning like a "
        "cognitive scientist assessing structural analogies. "
        "Be conservative. Never invent missing correspondences. "
        "Evaluate only what is explicitly supported by the analogy."
    )

    user_msg = f"""You will be given a Target Concept, its Description, and a generated Analogy that explains the Target Concept by comparing it to a different, more familiar source domain.

Your task is to rate the Analogy on one metric: Mapping Strength (MS).

Background:
A genuine analogy transfers the STRUCTURE of a source domain onto a target concept.

A strong analogy preserves:
- entities,
- functions,
- causal roles,
- relations between components,

rather than merely sharing similar words or attributes.

Important Rules:

1. Use ONLY evidence explicitly present in the Analogy.
2. NEVER invent additional correspondences using outside knowledge.
3. NEVER complete missing mappings on behalf of the author.
4. If a mapping only becomes valid after your own interpretation, classify it as MISSING.
5. Do NOT reward explanations that merely restate the target concept using different words.
6. Assume an average reader only has access to the Description and the Analogy.
   If the correspondence cannot be directly inferred from the Analogy itself,
   it is MISSING.

Evaluation Criteria (Mapping Strength):

0:
- No meaningful structural mapping.
- Mainly superficial similarities.
- Arbitrary comparison.
- Inconsistent mapping.

1:
- Some structural correspondences exist.
- Others are missing, weak, superficial, or inconsistent.

2:
- Structural relations are consistently preserved.
- Important components from the Description are covered.
- The source domain naturally explains the target.

Evaluation Steps:

Step 1.
Read the Description.

Extract the key components
(C1, C2, C3, ...).

Focus primarily on
- functions,
- causal relations,
- interactions,
not isolated properties.

--------------------------------

Step 2.

Identify the source domain.

Identify the major source entities.

--------------------------------

Step 3.

For EACH component:

a.
Quote the EXACT phrase from the Analogy.

Do NOT paraphrase.

If none exists, write

"no matching phrase found"

b.

Ask yourself:

"Does this quoted phrase explicitly establish the required structural correspondence?"

OR

"Am I relying on my own knowledge to make this correspondence?"

Only the FIRST is acceptable.

If the SECOND is true,
classify the component as MISSING.

c.

Classify exactly one:

RELATIONAL
- the structural relation/function is explicitly preserved.

SURFACE-ONLY
- only wording or attributes match.

MISSING
- absent, implicit, or requires additional interpretation.

--------------------------------

Step 4.

Check consistency.

Does every source entity represent the same target component throughout the Analogy?

Does any mapping shift?

Does the Analogy contradict itself?

--------------------------------

Step 5.

Assign the final score.

When uncertain between RELATIONAL and MISSING,
always choose MISSING.

When uncertain between scores,
choose the LOWER score.

Target Concept:
{target}

Description:
{description}

Analogy:
{analogy}
"""

    if lean_output:
        user_msg += """

Perform all reasoning internally.

Output exactly:

Score: <0, 1, or 2>
"""
    else:
        user_msg += """

Output exactly:

Key components:
<list>

Source domain:
<text>

Correspondence table:
- C1: quote "<exact phrase>" -> RELATIONAL/SURFACE-ONLY/MISSING | reason: <brief literal justification>

Consistency:
<brief>

Score: <0, 1, or 2>
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
# --------------------------------------------------------------------------
# DashScope OpenAI-compatible API client
# --------------------------------------------------------------------------

def load_environment(env_file: str = ".env") -> None:
    env_path = Path(env_file)
    candidates = [env_path] if env_path.is_absolute() else [
        Path.cwd() / env_path,
        Path(__file__).resolve().parent / env_path,
        Path(__file__).resolve().parents[1] / env_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=True)
            return


def get_dashscope_base_url() -> str:
    workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID")
    if not workspace_id:
        raise RuntimeError(
            "DASHSCOPE_WORKSPACE_ID was not found. "
            "Create .env from .env.example and fill in your workspace id."
        )
    return f"https://{workspace_id}.{DASHSCOPE_REGION}.maas.aliyuncs.com/compatible-mode/v1"


def get_client(env_file: str = ".env") -> OpenAI:
    load_environment(env_file)
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            f"DASHSCOPE_API_KEY was not found in {env_file}. "
            "Create .env from .env.example and fill in your API key."
        )

    return OpenAI(
        api_key=api_key,
        base_url=get_dashscope_base_url(),
    )


def extract_usage(response) -> dict:
    """Extract token usage when the provider returns it."""
    out = {"prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None}
    try:
        data = response.model_dump()
        usage = data.get("usage") or {}
        out["prompt_tokens"] = usage.get("prompt_tokens")
        out["completion_tokens"] = usage.get("completion_tokens")
        details = usage.get("completion_tokens_details") or {}
        out["reasoning_tokens"] = details.get("reasoning_tokens")
    except Exception:
        pass
    return out


def extract_reasoning_trace(response) -> str | None:
    """Extract the reasoning trace when the provider returns it."""
    try:
        data = response.model_dump()
        msg = (data.get("choices") or [{}])[0].get("message", {})
        return msg.get("reasoning") or msg.get("reasoning_content")
    except Exception:
        return None


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    want_logprobs: bool = True,
    top_logprobs: int = 5,
    max_tokens: int = 5000,
    reasoning_effort: str = "low",
    max_retries: int = 3,
):
    for attempt in range(max_retries):

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_headers": {
                "HTTP-Referer": "https://local-tcc-pipeline",
                "X-Title": "MS scorer",
            },
        }

        # Use extra_body for provider-specific parameters in the OpenAI SDK.
        if reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}

        if want_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        try:
            return client.chat.completions.create(**kwargs)

        except Exception as exc:
            error_msg = str(exc)

            # Model does not support logprobs.
            if (
                want_logprobs
                and (
                    "logprobs are not supported" in error_msg
                    or "unsupported_parameter" in error_msg
                    or "structured-outputs" in error_msg
                )
            ):
                print(f"  [info] {model} does not support logprobs; falling back to self-consistency")

                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)

                try:
                    return client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    print(f"  [warn] Retry without logprobs also failed: {exc2}")

            wait = 2 ** attempt
            print(
                f"  [warn] {model} failed on attempt {attempt+1}/{max_retries}: {exc}\n"
                f"         -> waiting {wait}s before retry..."
            )
            time.sleep(wait)

    return None

# --------------------------------------------------------------------------
# Parse scores and compute logprob-weighted expected scores
# --------------------------------------------------------------------------

def parse_discrete_score(text: str) -> int | None:
    match = re.search(
        r"Score\s*:?\s*\**([0-2])\**",
        text,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def expected_score_from_logprobs(response) -> float | None:
    """Find the final score token and compute the expected value from top logprobs."""
    choice = response.choices[0]
    if choice.logprobs is None or choice.logprobs.content is None:
        return None

    tokens = choice.logprobs.content
    target_idx = None
    for i, tok in enumerate(tokens):
        if tok.token.strip() in SCORE_CANDIDATES:
            target_idx = i  # keep the last score-token occurrence
    if target_idx is None:
        return None

    tok = tokens[target_idx]
    candidates = {tok.token.strip(): tok.logprob}
    for alt in (tok.top_logprobs or []):
        stripped = alt.token.strip()
        if stripped in SCORE_CANDIDATES:
            candidates.setdefault(stripped, alt.logprob)

    if not candidates:
        return None

    probs = {k: math.exp(v) for k, v in candidates.items()}
    total = sum(probs.values())
    if total <= 0:
        return None
    probs = {k: v / total for k, v in probs.items()}
    return sum(int(k) * p for k, p in probs.items())


def self_consistency_score(
    client, model, messages, *,
    max_tokens: int, reasoning_effort: str,
    first_score=None, first_usage=None,
    k=3, temperature=0, max_consecutive_failures=5,
):
    """Self-consistency fallback for models that do not return usable logprobs."""
    scores = []
    usages = [first_usage] if first_usage else []

    if first_score is not None:
        scores.append(first_score)

    consecutive_failures = 0
    while len(scores) < k:

        resp = call_model(
            client, model, messages,
            temperature=temperature,
            want_logprobs=False,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

        s = parse_discrete_score(resp.choices[0].message.content or "") if resp is not None else None
        if resp is not None:
            usages.append(extract_usage(resp))

        if s is not None:
            scores.append(s)
            consecutive_failures = 0
            continue

        consecutive_failures += 1
        if consecutive_failures >= max_consecutive_failures:
            print(
                f"  [warn] Could not parse a valid score for {max_consecutive_failures} "
                f"consecutive calls; stopping self-consistency early ({len(scores)}/{k} samples)"
            )
            break

    if not scores:
        return None, [], usages
    return sum(scores) / len(scores), scores, usages


def sum_usage(usages: list[dict]) -> dict:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "n_calls": 0}
    for u in usages:
        if not u:
            continue
        total["n_calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            if u.get(key) is not None:
                total[key] += u[key]
    return total


# --------------------------------------------------------------------------
# Score one sample
# --------------------------------------------------------------------------

def score_one_sample(client: OpenAI, model: str, target: str, description: str,
                      analogy: str, *, k_self_consistency: int, top_logprobs: int,
                      max_tokens: int, reasoning_effort: str, lean_output: bool,
                    ) -> dict:
    messages = build_ms_messages(target, description, analogy, lean_output=lean_output)
    t0 = time.time()

    resp = call_model(client, model, messages, temperature=0.0,
                       want_logprobs=True, top_logprobs=top_logprobs,
                       max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    method = "logprob_weighted"
    expected = None
    discrete = None
    raw_text = None
    reasoning_trace = None
    all_discrete_samples: list[int] = []
    first_usage = None

    if resp is not None:
        raw_text = resp.choices[0].message.content or ""
        reasoning_trace = extract_reasoning_trace(resp)
        first_usage = extract_usage(resp)
        discrete = parse_discrete_score(raw_text)
        # Trust logprobs only when the response actually reached the "Score:" line.
        if discrete is not None:
            expected = expected_score_from_logprobs(resp)

    usages = [first_usage] if first_usage else []

    if expected is None:
        method = "self_consistency"
        expected, all_discrete_samples, sc_usages = self_consistency_score(
            client, model, messages,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
            first_score=discrete, first_usage=None,  # first_usage is counted separately in usages
            k=k_self_consistency,
        )
        usages.extend(sc_usages)
        if discrete is None and all_discrete_samples:
            discrete = round(sum(all_discrete_samples) / len(all_discrete_samples))

    latency = time.time() - t0
    usage_total = sum_usage(usages)
    return {
        "target": target,
        "method": method,
        "discrete_score": discrete,
        "expected_score": expected,
        "self_consistency_samples": all_discrete_samples or None,
        "raw_response": raw_text,
        "reasoning_trace": reasoning_trace,
        "latency_sec": round(latency, 2),
        "n_api_calls": usage_total["n_calls"],
        "prompt_tokens": usage_total["prompt_tokens"],
        "completion_tokens": usage_total["completion_tokens"],
        "reasoning_tokens": usage_total["reasoning_tokens"],
    }


def run_experiment(
    client: OpenAI,
    model: str,
    df: pd.DataFrame,
    output_dir: Path,
    *,
    k_self_consistency: int,
    top_logprobs: int,
    split_name: str,
    max_tokens: int,
    reasoning_effort: str,
    lean_output: bool,
):
    # ============================================================
    # Validation: exclude few-shot examples.
    # Test: keep the full split.
    # ============================================================

    if split_name == "validation":
        eval_df = (
            df[~df["target"].isin(MS_FEWSHOT_TARGETS)]
            .copy()
            .reset_index(drop=False)
        )

        print(
            f"[info] Held-out validation: {len(eval_df)}/{len(df)} "
            f"(excluded {len(df)-len(eval_df)} few-shot samples)"
        )
    else:
        eval_df = df.reset_index(drop=False)

    print(
        f"\n=== Running model: {model} ({len(eval_df)} total samples) | "
        f"reasoning_effort={reasoning_effort} | "
        f"lean_output={lean_output} | "
        f"max_tokens={max_tokens} ==="
    )

    # Use stable filenames per model and split so runs can resume.
    model_slug = model.replace("/", "__")
    
    # Omit timestamps so each model has one stable detail file.
    detail_path = output_dir / f"{model_slug}__MS__{split_name}.jsonl"
    score_file = output_dir / f"{model_slug}__{split_name}_MS_scores.csv"

    rows = []
    processed_ids = set()

    # ============================================================
    # AUTO-RESUME: load previous rows if the detail file already exists.
    # ============================================================
    if detail_path.exists():
        with open(detail_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rows.append(data)
                    # Track processed IDs.
                    if "id" in data:
                        processed_ids.add(data["id"])
                    elif "sample_index" in data:
                        processed_ids.add(data["sample_index"])
                except json.JSONDecodeError:
                    pass

        print(f" [info] Found an existing log file. Restored {len(rows)} processed samples.")
    else:
        # Create an empty file first so later appends are safe.
        open(detail_path, "w", encoding="utf-8").close()

    # Remove already processed rows from eval_df.
    if processed_ids:
        # Use the original id column when available; otherwise use the row index.
        id_col = "id" if "id" in eval_df.columns else "index"
        
        # Keep only rows not present in processed_ids.
        eval_df = eval_df[~eval_df[id_col].isin(processed_ids)]
        print(f" [info] Skipped {len(processed_ids)} samples; {len(eval_df)} samples remain.")

    if eval_df.empty:
        print(f" [info] Model {model} has already completed this split; skipping API calls.")
    else:
        print(f" [info] Streaming run log to: {detail_path}")

    # ============================================================
    # API loop: run only samples that are not complete yet.
    # ============================================================
    for _, row in eval_df.iterrows():

        result = score_one_sample(
            client,
            model,
            target=str(row["target"]),
            description=str(row["description"]),
            analogy=str(row["analogy"]),
            k_self_consistency=k_self_consistency,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            lean_output=lean_output,
        )

        # Preserve the original id.
        result["sample_index"] = int(row["index"])

        if "id" in row:
            result["id"] = row["id"]

        if "MS" in eval_df.columns:
            result["ground_truth_MS"] = row["MS"]

        rows.append(result)

        print(
            f"  sample {result.get('id', result['sample_index'])}: "
            f"method={result['method']} "
            f"expected={result['expected_score']} "
            f"calls={result['n_api_calls']} "
            f"tokens(prompt/compl/reason)="
            f"{result['prompt_tokens']}/"
            f"{result['completion_tokens']}/"
            f"{result['reasoning_tokens']} "
            f"({result['latency_sec']}s)"
        )

        # ============================================================
        # Write progress immediately after each sample.
        # ============================================================
        
        # 1. Append the result row to the JSONL file.
        with open(detail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
        # 2. Rewrite the CSV with all rows accumulated so far.
        current_score_df = pd.DataFrame({
            "id": [r.get("id", r["sample_index"]) for r in rows],
            "score": [
                r["expected_score"] if r["expected_score"] is not None else r["discrete_score"] 
                for r in rows
            ]
        })
        current_score_df["score"] = current_score_df["score"].round(0)
        current_score_df.to_csv(score_file, index=False)


    result_df = pd.DataFrame(rows)
    print(f"  -> Completed the split. Saved scores: {score_file}")
    
    # Generate one timestamp at the end for the summary log.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ============================================================
    # Usage statistics
    # ============================================================

    total_prompt = result_df["prompt_tokens"].sum() if not result_df.empty else 0
    total_completion = result_df["completion_tokens"].sum() if not result_df.empty else 0
    total_reasoning = result_df["reasoning_tokens"].sum() if not result_df.empty else 0
    total_calls = result_df["n_api_calls"].sum() if not result_df.empty else 0

    print(
        f"  -> Total tokens used across this and resumed runs: "
        f"prompt={total_prompt}, "
        f"completion={total_completion}, "
        f"reasoning={total_reasoning}, "
        f"api_calls={total_calls}"
    )

    method_used = result_df["method"].mode().iat[0] if not result_df.empty else "N/A"

    summary = {
        "timestamp_utc": timestamp,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "lean_output": lean_output,
        "n_samples": len(rows),
        "method_used": method_used,
        "total_api_calls": int(total_calls),
        "total_prompt_tokens": int(total_prompt),
        "total_completion_tokens": int(total_completion),
        "total_reasoning_tokens": int(total_reasoning),
        "detail_file": str(detail_path),
    }

    # ============================================================
    # Correlation
    # ============================================================

    if split_name == "validation" and not result_df.empty:

        merged = result_df.dropna(subset=["expected_score"])

        if len(merged) >= 3:

            rho, rho_p = spearmanr(
                merged["expected_score"],
                merged["ground_truth_MS"],
            )

            tau, tau_p = kendalltau(
                merged["expected_score"],
                merged["ground_truth_MS"],
            )

            summary.update({
                "spearman_rho": round(rho, 4),
                "spearman_p": round(rho_p, 4),
                "kendall_tau": round(tau, 4),
                "kendall_p": round(tau_p, 4),
                "n_scored": len(merged),
            })

            print(
                f"  -> Spearman={summary.get('spearman_rho', 'N/A')} "
                f"(p={summary.get('spearman_p', 'N/A')}), "
                f"Kendall={summary.get('kendall_tau', 'N/A')} "
                f"(p={summary.get('kendall_p', 'N/A')})"
            )

        else:
            print("[warn] Not enough samples to compute Spearman/Kendall")

    return summary

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def append_experiment_log(output_dir: Path, summary: dict) -> None:
    log_path = output_dir / "experiment_log_MS.csv"
    row_df = pd.DataFrame([summary])
    if log_path.exists():
        row_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(log_path, mode="w", header=True, index=False)


def main():
    parser = argparse.ArgumentParser(description="Score MS (Mapping Strength) with DashScope")

    parser.add_argument("--input", default="challenge-dataset",
                         help="Path to the HuggingFace dataset")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                         help="Comma-separated model list")
    parser.add_argument("--output-dir", default="results/ms")
    parser.add_argument("--k-self-consistency", type=int, default=3,
                         help="Number of samples when logprobs are unavailable")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=700,
                         help="Token budget per call, including hidden reasoning tokens when present")
    parser.add_argument("--reasoning-effort", default="low", choices=REASONING_EFFORT_CHOICES,
                         help="Hidden reasoning budget for reasoning models "
                              "('none' disables reasoning; 'low' is the default cost/quality balance)")
    parser.add_argument("--output-mode", default="lean", choices=["lean", "full"],
                         help="'lean' asks only for the Score line; 'full' asks for the mapping table")

    args = parser.parse_args()

    dataset = load_from_disk(args.input)
    if args.split not in dataset:
        raise ValueError(f"Split '{args.split}' does not exist. Available splits: {list(dataset.keys())}")

    df = dataset[args.split].to_pandas()

    required_cols = {"target", "description", "analogy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    lean_output = args.output_mode == "lean"

    summaries = []
    for model in models:
        summary = run_experiment(
            client=client, model=model, df=df, output_dir=output_dir,
            k_self_consistency=args.k_self_consistency,
            top_logprobs=args.top_logprobs,
            split_name=args.split,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            lean_output=lean_output,
        )
        append_experiment_log(output_dir, summary)
        summaries.append(summary)

    print("\n=== Result summary ===")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
