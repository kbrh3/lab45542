"""
eval/run_eval.py

Evaluation pipeline comparing baseline vs. LoRA-adapted model on
PolicyPulse legislative benchmark questions.

Metrics:
  - ROUGE-L  (recall-oriented overlap with expected output)
  - BLEU     (precision-oriented n-gram overlap)
  - Exact missing-evidence detection rate
  - Average generation latency

Usage:
    python eval/run_eval.py                  # evaluate both modes
    python eval/run_eval.py --mode baseline  # baseline only
"""

import argparse
import json
import pathlib
import time
import sys
import os

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collections import defaultdict

BENCHMARK_PATH = ROOT / "eval" / "benchmark_questions.json"
RESULTS_DIR    = ROOT / "eval" / "results"


# ── Lightweight metric helpers (no nltk dependency) ────────────────────
def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not hyp_tok or not ref_tok:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ng = _ngrams(ref_tok, n)
        hyp_ng = _ngrams(hyp_tok, n)
        if not hyp_ng:
            scores.append(0.0)
            continue
        ref_counts = defaultdict(int)
        for ng in ref_ng:
            ref_counts[ng] += 1
        matches = 0
        for ng in hyp_ng:
            if ref_counts[ng] > 0:
                matches += 1
                ref_counts[ng] -= 1
        scores.append(matches / len(hyp_ng))

    if 0.0 in scores:
        return 0.0

    import math
    brevity = min(1.0, len(hyp_tok) / len(ref_tok))
    log_avg = sum(math.log(s) for s in scores) / len(scores)
    return brevity * math.exp(log_avg)


def rouge_l(reference: str, hypothesis: str) -> float:
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not ref_tok or not hyp_tok:
        return 0.0

    m, n = len(ref_tok), len(hyp_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tok[i - 1] == hyp_tok[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n
    rec  = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ── Evaluation ─────────────────────────────────────────────────────────
def evaluate_mode(mode: str, questions: list) -> list:
    from llm.domain_model_client import set_mode, generate

    set_mode(mode)
    results = []

    for q in questions:
        t0 = time.time()
        pred = generate(q["instruction"], q.get("input", ""))
        latency = (time.time() - t0) * 1000.0

        expected = q["expected_output"]
        rl = rouge_l(expected, pred)
        bl = bleu_score(expected, pred)

        is_missing_ev = q["category"] == "missing_evidence"
        missing_detected = "not enough evidence" in pred.lower() if is_missing_ev else None

        results.append({
            "id": q["id"],
            "category": q["category"],
            "mode": mode,
            "prediction": pred,
            "expected": expected,
            "rouge_l": round(rl, 4),
            "bleu": round(bl, 4),
            "latency_ms": round(latency, 1),
            "missing_evidence_correct": missing_detected,
        })

        print(f"  [{q['id']}] ROUGE-L={rl:.3f}  BLEU={bl:.3f}  latency={latency:.0f}ms")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "adapted", "both"], default="both")
    args = parser.parse_args()

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} benchmark questions.\n")

    modes = ["baseline", "adapted"] if args.mode == "both" else [args.mode]
    all_results = []

    for mode in modes:
        print(f"{'=' * 50}")
        print(f"Evaluating: {mode}")
        print(f"{'=' * 50}")
        try:
            mode_results = evaluate_mode(mode, questions)
            all_results.extend(mode_results)
        except Exception as e:
            print(f"  [ERROR] {mode} evaluation failed: {e}")
            continue

    if not all_results:
        print("No results generated.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results → {out_path}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for mode in modes:
        subset = [r for r in all_results if r["mode"] == mode]
        if not subset:
            continue
        avg_rl = sum(r["rouge_l"] for r in subset) / len(subset)
        avg_bl = sum(r["bleu"] for r in subset) / len(subset)
        avg_lat = sum(r["latency_ms"] for r in subset) / len(subset)

        missing_q = [r for r in subset if r["missing_evidence_correct"] is not None]
        missing_acc = (
            sum(1 for r in missing_q if r["missing_evidence_correct"]) / len(missing_q)
            if missing_q else float("nan")
        )

        print(f"\n  Mode: {mode}")
        print(f"    Avg ROUGE-L       : {avg_rl:.4f}")
        print(f"    Avg BLEU          : {avg_bl:.4f}")
        print(f"    Avg Latency (ms)  : {avg_lat:.1f}")
        print(f"    Missing-Evidence  : {missing_acc:.2%}" if missing_q else "    Missing-Evidence  : N/A")

    summary = {}
    for mode in modes:
        subset = [r for r in all_results if r["mode"] == mode]
        if subset:
            summary[mode] = {
                "avg_rouge_l": round(sum(r["rouge_l"] for r in subset) / len(subset), 4),
                "avg_bleu": round(sum(r["bleu"] for r in subset) / len(subset), 4),
                "avg_latency_ms": round(sum(r["latency_ms"] for r in subset) / len(subset), 1),
                "n_questions": len(subset),
            }
    with open(RESULTS_DIR / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {RESULTS_DIR / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
