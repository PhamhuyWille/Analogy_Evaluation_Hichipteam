"""
M.py
====
Metaphoricity (M) scoring pipeline for the Long-Form Analogy Evaluation task.
It uses the DashScope OpenAI-compatible API and exposes these CLI options:
  --reasoning-effort
  --max-tokens
  --output-mode

The default report model is qwen3-max. See README.md for environment setup and
run commands.
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
    "qwen3-max",
]
DASHSCOPE_REGION = "ap-southeast-1"

SCORE_CANDIDATES = ["0", "1", "2"]  # M score scale: 0, 1, 2
REASONING_EFFORT_CHOICES = ["none", "minimal", "low", "medium", "high"]


# --------------------------------------------------------------------------
# Prompt (G-Eval-style: decomposition + form-filling) rubric for Metaphoricity
# --------------------------------------------------------------------------

def build_m_messages(target: str, description: str, analogy: str,
                     split: str = "validation",
                     lean_output: bool = True) -> list[dict]:
    if split == "test":
        lean_output = True

    system_msg = (
        "You are a careful evaluator of analogy quality for concept explanation. "
        "You always follow the requested output format exactly."
    )
    
    # Keep the original prompt structure, with clearer score boundaries.
    user_msg = f"""You will be given a Target Concept, its Description, and a generated
Analogy written to explain that concept.

Your task is to rate the Analogy on one metric: Metaphoricity (M).

Evaluation Criteria:
Metaphoricity (0-2) - the extent to which the Analogy is genuinely figurative
(it uses a distinct source domain/concept to illuminate the target, rather
than merely restating or paraphrasing the Description) and how well-developed
that figurative comparison is.

0 - Not figurative: the text is essentially a literal definition, restatement,
    paraphrase of the Description, or an example from the EXACT SAME domain 
    (e.g., explaining software/hardware using another computer program/code).
1 - Somewhat figurative: a distinct source domain is used, but the comparison
    is extremely thin, brief, or minimally developed (only a single vague sentence).
2 - Clearly figurative and well-developed: a well-developed, distinct source 
    domain is used to illuminate the target concept. (Note: Familiar or standard 
    analogies e.g., cooking, traffic, security, still receive a 2 if they are 
    elaborated with structural correspondences).

Evaluation Steps (do this internally):
1. Compare the Analogy to the Description: does the Analogy introduce a
   distinct source domain/concept, or is it a literal restatement / same-domain example?
2. If a distinct source domain is used, judge whether it provides an elaborated 
   comparison (Score 2) or just a thin/brief mention (Score 1).
3. Based on steps 1-2, assign a score of 0, 1, or 2.

Target Concept: {target}
Description: {description}
Analogy: {analogy}
"""

    if lean_output:
        user_msg += """
Do the analysis internally.

Output format (follow exactly, nothing else, nothing before or after):
Score: <0, 1, or 2>"""
    else:
        user_msg += """
Output format (follow exactly, nothing after the last line):
Figurative or literal: <short judgement>
Creativity analysis: <one or two sentences>
Score: <0, 1, or 2>"""

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
    max_tokens: int = 700,
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
                "X-Title": "M scorer",
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
            err_lower = error_msg.lower()

            # Detect providers/models that do not support logprobs.
            is_logprob_error = "logprobs" in err_lower and (
                "not supported" in err_lower 
                or "unsupported" in err_lower 
                or "invalid_parameter" in err_lower
            )

            if want_logprobs and is_logprob_error:
                print(f"  [info] {model} does not support logprobs; falling back to self-consistency")

                # Remove logprob parameters before retrying.
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)

                try:
                    return client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    print(f"  [warn] Retry without logprobs also failed: {exc2}")

            # Retry other errors such as network failures, timeouts, or rate limits.
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
    if not text:
        return None
        
    # 1. Match the standard "Score: X" format.
    match = re.search(r"Score\s*:?\s*\**([0-2])\**", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 2. Fallback: find a standalone 0, 1, or 2 when "Score" is absent.
    numbers = re.findall(r"\b([0-2])\b", text)
    if numbers:
        return int(numbers[-1])  # Use the last valid score-like number in the response.

    return None


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
    k=3, temperature=0.8, max_consecutive_failures=5,
):
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
                      split: str, max_tokens: int, reasoning_effort: str, lean_output: bool) -> dict:
    messages = build_m_messages(target, description, analogy, split=split, lean_output=lean_output)
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
        if discrete is not None:
            expected = expected_score_from_logprobs(resp)

    usages = [first_usage] if first_usage else []

    if expected is None:
        method = "self_consistency"
        expected, all_discrete_samples, sc_usages = self_consistency_score(
            client, model, messages,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
            first_score=discrete, first_usage=None,
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


# --------------------------------------------------------------------------
# Run one experiment for one model across the full dataset
# --------------------------------------------------------------------------

def run_experiment(client: OpenAI, model: str, df: pd.DataFrame,
                    output_dir: Path, *, k_self_consistency: int,
                    top_logprobs: int, split_name: str, max_tokens: int,
                    reasoning_effort: str, lean_output: bool) -> dict:
    print(f"\n=== Running model: {model} ({len(df)} samples) | "
          f"reasoning_effort={reasoning_effort} | lean_output={lean_output} | "
          f"max_tokens={max_tokens} ===")
    rows = []
    for i, row in df.iterrows():
        result = score_one_sample(
            client, model,
            target=str(row.get("target", "")),
            description=str(row.get("description", "")),
            analogy=str(row.get("analogy", "")),
            k_self_consistency=k_self_consistency,
            top_logprobs=top_logprobs,
            split=split_name,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            lean_output=lean_output,
        )
        result["sample_index"] = int(i)
        if "M" in df.columns:
            result["ground_truth_M"] = row["M"]
        rows.append(result)
        print(f"  sample {i}: method={result['method']} "
              f"expected={result['expected_score']} "
              f"calls={result['n_api_calls']} "
              f"tokens(prompt/compl/reason)="
              f"{result['prompt_tokens']}/{result['completion_tokens']}/{result['reasoning_tokens']} "
              f"({result['latency_sec']}s)")

    result_df = pd.DataFrame(rows)

    # Save scores for submission generation.
    score_df = pd.DataFrame({
        "id": result_df["sample_index"],
        "score": result_df["expected_score"].fillna(result_df["discrete_score"])
    })
    score_file = output_dir / f"{split_name}_M_scores.csv"
    score_df.to_csv(score_file, index=False)
    print(f"  -> Saved scores: {score_file}")

    # Save detailed per-sample results.
    model_slug = model.replace("/", "__")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    detail_path = output_dir / f"{model_slug}__M__{timestamp}.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Saved detailed results: {detail_path}")

    total_prompt = result_df["prompt_tokens"].sum()
    total_completion = result_df["completion_tokens"].sum()
    total_reasoning = result_df["reasoning_tokens"].sum()
    total_calls = result_df["n_api_calls"].sum()
    print(f"  -> Total tokens used: prompt={total_prompt}, "
          f"completion={total_completion}, reasoning={total_reasoning}, "
          f"api_calls={total_calls}")

    summary = {
        "timestamp_utc": timestamp,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "lean_output": lean_output,
        "n_samples": len(df),
        "method_used": result_df["method"].mode().iat[0] if not result_df.empty else None,
        "total_api_calls": int(total_calls),
        "total_prompt_tokens": int(total_prompt),
        "total_completion_tokens": int(total_completion),
        "total_reasoning_tokens": int(total_reasoning),
        "detail_file": str(detail_path),
    }

    if "M" in df.columns:
        merged = result_df.dropna(subset=["expected_score"]).merge(
            df.reset_index().rename(columns={"index": "sample_index"})[["sample_index", "M"]],
            on="sample_index", how="left",
        )
        if len(merged) >= 3:
            rho, rho_p = spearmanr(merged["expected_score"], merged["M"])
            tau, tau_p = kendalltau(merged["expected_score"], merged["M"])
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
    log_path = output_dir / "experiment_log_M.csv"
    row_df = pd.DataFrame([summary])
    if log_path.exists():
        row_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(log_path, mode="w", header=True, index=False)


def main():
    parser = argparse.ArgumentParser(description="Score M (Metaphoricity) with DashScope")

    parser.add_argument("--input", default="challenge-dataset", help="Path to the HuggingFace dataset")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model list")
    parser.add_argument("--output-dir", default="results/m")
    parser.add_argument("--k-self-consistency", type=int, default=3, help="Number of samples when logprobs are unavailable")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=700, help="Maximum token budget per call")
    parser.add_argument("--reasoning-effort", default="low", choices=REASONING_EFFORT_CHOICES, help="Hidden reasoning effort for reasoning models")
    parser.add_argument("--output-mode", default="lean", choices=["lean", "full"], help="'lean' outputs only the Score line; 'full' includes analysis")

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
