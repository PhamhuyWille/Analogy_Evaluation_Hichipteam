import argparse
import json
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
    "openai/gpt-4o-mini",
]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SCORE_CANDIDATES = ["0", "1", "2"]

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
def build_tcc_messages(target: str, description: str, analogy: str) -> list[dict]:
    system_msg = (
        "You are an expert evaluator of educational analogies. "
        "Your task is to evaluate how well an analogy explains a target concept "
        "based on the provided concept description. Always strictly follow the "
        "evaluation guidelines and the required output format."
    )

    user_msg = f"""Evaluate how well an analogy explains a target concept based on the provided concept description.

For the example below, assign a discrete score of 0, 1, or 2.

SCORING CRITERIA

Score 2 — Strong / Accurate Analogy
Give a 2 when the analogy successfully communicates the main idea, mechanism, or defining properties of the target concept.
The analogy does NOT need to mention every detail in the description. Minor omissions, differences in terminology, examples, or simplifications are acceptable as long as they do not change the core meaning of the concept.
A 2 is appropriate when:
- The main mechanism or relationship is correctly represented.
- The analogy preserves the important functional relationship between the analogy and the target concept.
- Any omitted details are secondary rather than essential.
- The analogy does not introduce a substantially misleading interpretation.

Score 1 — Partial / Incomplete Analogy
Give a 1 when the analogy captures a meaningful part of the concept but has a noticeable conceptual limitation.
A 1 is appropriate when:
- One important aspect of the concept is missing while the central idea is still partially conveyed.
- The analogy is substantially simplified and loses an important part of the concept.
- The analogy contains a limited misleading element, but the overall analogy still has meaningful correspondence to the target concept.
Do NOT assign 1 merely because the analogy omits minor details, examples, terminology, implementation-specific information, or secondary applications.

Score 0 — Incorrect / Misleading Analogy
Give a 0 only when the analogy fundamentally fails to represent the target concept.
A 0 is appropriate when:
- The analogy contradicts or substantially misrepresents the core mechanism of the concept.
- The analogy maps the concept to a fundamentally different mechanism.
- Most of the essential meaning is absent.
- The analogy would likely cause a learner to form a seriously incorrect mental model.

IMPORTANT EVALUATION PRINCIPLES

1. Focus on conceptual equivalence, not wording. Do not require the analogy to explicitly use the same terminology as the description.
2. Do not require complete coverage. Distinguish between CORE CLAIMS (properties/mechanisms that define the concept) and SUPPORTING DETAILS (examples, applications, implementation details). Missing a supporting detail should normally NOT prevent a score of 2.
3. Missing information is less serious than incorrect information. Minor omission -> usually still 2. Important omission -> possibly 1. Direct contradiction/fundamental distortion -> usually 0 or 1.
4. Evaluate the analogy as an analogy. Do not penalize it simply because it uses a different domain, omits technical terminology, or focuses on the most intuitive aspect of the concept.
5. Evaluate scope carefully. Only call something a "scope mismatch" when the analogy actively suggests that the concept is limited to something it is not.
6. Contradictions are more serious than omissions.
7. Consider the likely learner interpretation. "If a learner understood this analogy, would they have a substantially correct understanding of the target concept?"
8. Do not over-penalize domain differences. Evaluate whether the relevant RELATIONSHIP is preserved.

EVALUATION PROCESS
Step 1 — Identify the 2–4 most important/core claims of the target concept.
Step 2 — Check whether the analogy conveys those core claims (COVERED / PARTIALLY COVERED / MISSING / MISLEADING).
Step 3 — Determine whether missing claims are core claims or supporting details.
Step 4 — Check whether the analogy introduces contradictions or misleading mappings.
Step 5 — Evaluate the overall conceptual equivalence and likely learner interpretation.
Step 6 — Assign the final score. 
IMPORTANT: Do NOT mechanically assign the score based on the number of covered claims.

Target Concept: {target}
Description: {description}
Analogy: {analogy}

OUTPUT FORMAT (Follow exactly)

Essential Claims:
- I1: ...
- I2: ...
- I3: ...

Coverage Check:
- I1: COVERED / PARTIALLY COVERED / MISSING / MISLEADING
  Evidence: "..."
  Reason: ...
- I2: COVERED / PARTIALLY COVERED / MISSING / MISLEADING
  Evidence: "..."
  Reason: ...
- I3: COVERED / PARTIALLY COVERED / MISSING / MISLEADING
  Evidence: "..."
  Reason: ...

Conceptual Accuracy:
<Briefly explain whether the analogy preserves the central meaning of the target concept. Distinguish between minor omissions, important omissions, and genuine contradictions or misleading mappings.>

Final Analysis:
<Give a concise explanation of why the analogy deserves 0, 1, or 2.>

Score: <0, 1, or 2>"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

# --------------------------------------------------------------------------
# OpenRouter API client
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


def get_client(env_file: str = ".env") -> OpenAI:
    load_environment(env_file)
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            f"OPENROUTER_API_KEY was not found in {env_file}. "
            "Create .env from .env.example and fill in your OpenRouter API key."
        )

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    want_logprobs: bool = True,
    top_logprobs: int = 5,
    max_retries: int = 3,
    max_tokens: int = 7000,
    reasoning_effort: str = None,
):
    for attempt in range(max_retries):
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_headers": {
                "HTTP-Referer": "https://local-tcc-pipeline",
                "X-Title": "TCC scorer",
            },
        }

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        if want_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        try:
            return client.chat.completions.create(**kwargs)

        except Exception as exc:
            error_msg = str(exc)

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
    choice = response.choices[0]
    if choice.logprobs is None or choice.logprobs.content is None:
        return None

    tokens = choice.logprobs.content
    target_idx = None
    for i, tok in enumerate(tokens):
        if tok.token.strip() in SCORE_CANDIDATES:
            target_idx = i  
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

    import math
    probs = {k: math.exp(v) for k, v in candidates.items()}
    total = sum(probs.values())
    if total <= 0:
        return None
    probs = {k: v / total for k, v in probs.items()}
    return sum(int(k) * p for k, p in probs.items())


def self_consistency_score(
    client,
    model,
    messages,
    first_score=None,
    k=3,
    temperature=0,
    max_consecutive_failures=5,
    max_tokens=9000,
    reasoning_effort=None,
):
    scores = []
    sc_prompt_tokens = 0
    sc_completion_tokens = 0

    if first_score is not None:
        scores.append(first_score)

    consecutive_failures = 0
    while len(scores) < k:
        resp = call_model(
            client,
            model,
            messages,
            temperature=temperature,
            want_logprobs=False,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

        if resp is not None:
            # Accumulate token usage from retry calls.
            if hasattr(resp, 'usage') and resp.usage:
                sc_prompt_tokens += resp.usage.prompt_tokens
                sc_completion_tokens += resp.usage.completion_tokens

            s = parse_discrete_score(resp.choices[0].message.content or "")
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

    final_score = sum(scores) / len(scores) if scores else None
    return final_score, scores, sc_prompt_tokens, sc_completion_tokens


# --------------------------------------------------------------------------
# Score one sample
# --------------------------------------------------------------------------

def score_one_sample(client: OpenAI, model: str, target: str, description: str,
                     analogy: str, k_self_consistency: int, top_logprobs: int,
                     max_tokens: int, reasoning_effort: str) -> dict:
    messages = build_tcc_messages(target, description, analogy)
    t0 = time.time()

    resp = call_model(client, model, messages, temperature=0.0,
                       want_logprobs=True, top_logprobs=top_logprobs,
                       max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    method = "logprob_weighted"
    expected = None
    discrete = None
    raw_text = None
    all_discrete_samples: list[int] = []
    
    # Initialize token counters.
    prompt_tokens = 0
    completion_tokens = 0

    if resp is not None:
        # Count tokens from the first API call.
        if hasattr(resp, 'usage') and resp.usage:
            prompt_tokens += resp.usage.prompt_tokens
            completion_tokens += resp.usage.completion_tokens

        raw_text = resp.choices[0].message.content or ""
        discrete = parse_discrete_score(raw_text)
        if discrete is not None:
            expected = expected_score_from_logprobs(resp)

    # Fall back here when the model does not provide usable logprobs.
    if expected is None:
        method = "self_consistency"
        expected, all_discrete_samples, sc_pt, sc_ct = self_consistency_score(
            client,
            model,
            messages,
            first_score=discrete,
            k=k_self_consistency,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        
        # Add token usage from self-consistency calls.
        prompt_tokens += sc_pt
        completion_tokens += sc_ct

        if discrete is None and all_discrete_samples:
            discrete = round(sum(all_discrete_samples) / len(all_discrete_samples))

    latency = time.time() - t0
    return {
        "target": target,
        "method": method,
        "discrete_score": discrete,
        "expected_score": expected,
        "self_consistency_samples": all_discrete_samples or None,
        "raw_response": raw_text,
        "latency_sec": round(latency, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# --------------------------------------------------------------------------
# Run one experiment
# --------------------------------------------------------------------------

def run_experiment(client: OpenAI, model: str, df: pd.DataFrame,
                   output_dir: Path, k_self_consistency: int,
                   top_logprobs: int, split_name: str, 
                   max_tokens: int, reasoning_effort: str, output_mode: str) -> dict:
    print(f"\n=== Running model: {model} ({len(df)} samples) ===")
    rows = []
    for i, row in df.iterrows():
        result = score_one_sample(
            client, model,
            target=str(row.get("target", "")),
            description=str(row.get("description", "")),
            analogy=str(row.get("analogy", "")),
            k_self_consistency=k_self_consistency,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        result["sample_index"] = int(i)
        if "TCC" in df.columns:
            result["ground_truth_TCC"] = row["TCC"]
        rows.append(result)
        
        # Print per-sample progress with token usage.
        if output_mode != "lean":
             print(f"  sample {i}: method={result['method']} "
                   f"expected={result['expected_score']} "
                   f"({result['latency_sec']}s) "
                   f"[Tokens: {result.get('prompt_tokens', 0)} in / {result.get('completion_tokens', 0)} out]")
        elif i % 10 == 0: 
             print(f"  [Lean Mode] Processed {i} samples...")

    result_df = pd.DataFrame(rows)

    score_df = pd.DataFrame({
        "id": result_df["sample_index"],
        "score": result_df["expected_score"].fillna(result_df["discrete_score"])
    })

    score_file = output_dir / f"{split_name}_scores.csv"
    score_df.to_csv(score_file, index=False)

    print(f"  -> Saved scores: {score_file}")

    model_slug = model.replace("/", "__")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    detail_path = output_dir / f"{model_slug}__{timestamp}.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Saved detailed results: {detail_path}")

    summary = {
        "timestamp_utc": timestamp,
        "model": model,
        "n_samples": len(df),
        "method_used": result_df["method"].mode().iat[0] if not result_df.empty else None,
        "detail_file": str(detail_path),
    }

    if "TCC" in df.columns:
        valid = result_df.dropna(subset=["expected_score"])
        gt = df.loc[valid.index, "TCC"] if len(valid) == len(df) else None
        merged = result_df.dropna(subset=["expected_score"]).merge(
            df.reset_index().rename(columns={"index": "sample_index"})[["sample_index", "TCC"]],
            on="sample_index", how="left",
        )
        if len(merged) >= 3:
            rho, rho_p = spearmanr(merged["expected_score"], merged["TCC"])
            tau, tau_p = kendalltau(merged["expected_score"], merged["TCC"])
            summary.update({
                "spearman_rho": round(rho, 4),
                "spearman_p": round(rho_p, 4),
                "kendall_tau": round(tau, 4),
                "kendall_p": round(tau_p, 4),
                "n_scored": len(merged),
            })
            print(f"  -> Spearman={summary['spearman_rho']} "
                  f"(p={summary['spearman_p']}), "
                  f"Kendall={summary['kendall_tau']} (p={summary['kendall_p']})")
        else:
            print("  [warn] Not enough valid samples to compute Spearman/Kendall")

    return summary

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def append_experiment_log(output_dir: Path, summary: dict) -> None:
    log_path = output_dir / "experiment_log.csv"
    row_df = pd.DataFrame([summary])
    if log_path.exists():
        row_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(log_path, mode="w", header=True, index=False)


def main():
    parser = argparse.ArgumentParser(description="Score TCC with OpenRouter")

    parser.add_argument("--input", default="challenge-dataset", help="Path to the HuggingFace dataset")
    parser.add_argument("--split", default="validation", choices=["validation", "test"], help="Dataset split")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model list")
    parser.add_argument("--output-dir", default="results/tcc")
    parser.add_argument("--k-self-consistency", type=int, default=1, help="Number of samples when logprobs are unavailable")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--reasoning-effort", type=str, default="medium", help="Model reasoning effort")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum output tokens")
    parser.add_argument("--output-mode", type=str, default="normal", help="Output mode, e.g. lean for reduced logging")

    args = parser.parse_args()

    dataset = load_from_disk(args.input)

    if args.split not in dataset:
        raise ValueError(
            f"Split '{args.split}' does not exist. "
            f"Available splits: {list(dataset.keys())}"
        )

    df = dataset[args.split].to_pandas()

    required_cols = {"target", "description", "analogy"}
    missing = required_cols - set(df.columns)

    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summaries = []

    for model in models:
        summary = run_experiment(
            client=client,
            model=model,
            df=df,
            output_dir=output_dir,
            k_self_consistency=args.k_self_consistency,
            top_logprobs=args.top_logprobs,
            split_name=args.split,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            output_mode=args.output_mode,
        )

        append_experiment_log(output_dir, summary)
        summaries.append(summary)

    print("\n=== Result summary ===")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
