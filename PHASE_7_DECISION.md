# Phase 7: Verification Layer

## Status

**Designed and prototyped offline.** Awaits run11+ (curriculum v5) to demonstrate end-to-end value. Discriminator validated on synthetic computation-vs-guess code (14-point score gap, see scripts/verifier.py).

## Motivation

Phase 6 produced a 500M procedural model that beats GPT-2 XL (1.5B) by 12–80× on
procedural in-distribution tasks and by 198× on syntactic Python validity. On
GSM8K via PAL it reaches 3% (greedy) and 22% (oracle@N=16) — but voting can only
extract ~3–7% of the latent 22% capability. The bottleneck moved from
*generation* to *answer selection*.

The Phase 6.15 inspection of run10 candidates revealed a deeper issue: the
"correct" candidates were the model GUESSING the right answer, not computing
it. v4's gsm8k_train_python weak supervision (bind problem numbers, print
gold) taught pattern-matching, not arithmetic. Curriculum v5
(curriculum_gsm8k_v2.py) emits real arithmetic Python extracted from
`<<expr=result>>` markers in GSM8K-train answers; run11 retrains on v5.

This Phase 7 verifier sits *downstream* of run11: it ranks the N candidate
programs the model emits per problem and picks the most-likely-correct
one. The goal is to extract a substantial fraction of oracle@N — plausibly
10–15% on GSM8K with a small model that *actually computes*.

## Architecture

```
problem ──► [Procedural Core] ──► N candidate programs
                                    │
                                    ▼
                              [Verifier]
                              ├─ static analysis    (ops, chain, input use)
                              ├─ execution check    (runs without error)
                              ├─ echo filter        (penalize ans-in-problem)
                              ├─ perturbation test  (responds to inputs)
                              └─ combined score
                                    │
                                    ▼
                              top-scored answer
                                    │
                                    ▼
                         (commit / retry / defer)
```

Maps to the original B.L.A. spec as the **verification layer + commitment
object**. Output of the verifier is a `Commitment{pred, confidence,
evidence}` — what the procedural core hands off to the action router.

## Signal Sources (all offline-computable, no extra model calls)

### 1. Arithmetic op count
Count `ast.BinOp` nodes with arithmetic operators. Codes with `ops == 0`
are pure variable-binding-and-print sequences — guesses.

### 2. Final-answer derivation
Walk the AST: is `answer = ...` (or the last `print(...)` arg) a Name
that points back to a chain of assignments ending in a BinOp? If yes,
the model computed the answer; if no, it was assigned a literal.

### 3. Input-number usage
Of the numbers appearing in the problem text, how many appear *inside*
a BinOp expression in the code (not merely bound to a variable that's
never used downstream)?

### 4. Echo filter
If the predicted answer is numerically equal to a number in the
problem, downweight it. GSM8K answers are rarely problem inputs;
correct echoes are coincidence.

### 5. Perturbation responsiveness (the killer signal)
For each numeric literal in the code that ALSO appears in the problem
(an "input" literal):
  - Replace it with `value + 17` in the code
  - Re-execute
  - Check if the output changed
  
A computation-bearing program's output should be a function of its
inputs; perturbing inputs should change outputs. A guess prints the
same number regardless. We compute responsiveness = fraction of
perturbations that produced a different output.

Note: critical to perturb *only* input literals, not the answer
literal. Perturbing the answer trivially changes the output without
testing computation.

### Aggregation

```
combined_score = 2.0 * ops
               + 3.0 * (1 if is_computed else 0)
               + 1.0 * chain_depth
               + 1.0 * nums_in_op
               - 1.5 * (1 if echo else 0)
               + 4.0 * responsiveness
```

Three selection strategies, picked offline:
  - `mode`  — most common pred (baseline SC)
  - `score` — single highest-scoring candidate's pred
  - `vote`  — sum scores per pred bucket; pick best bucket

## Validation on Run10 Data (offline)

Run10 candidates.jsonl: 200 problems × 16 samples. Oracle@16 = 22%.

```
strategy   accuracy
mode       0.0%
score      0.0%   ← verifier picks but can't discriminate
vote       0.0%   ← all candidates score ≈ 0
```

**Diagnostic:** 0/3200 candidates have `ops > 0`; 0/3200 have `is_computed`;
4.2% have any perturbation responsiveness. **The data confirms run10 doesn't
compute.** No reranker can extract signal that isn't there. The fix is the
v5 curriculum (Phase 6.16); the verifier is the next stage *after* that.

## Synthetic discrimination test

```python
GOOD_CODE = "n1=16; n2=3; n3=4; n4=2; step1=n1-n2-n3; step2=step1*n4; answer=step2; print(answer)"
BAD_CODE  = "n1=16; n2=3; n3=4; answer=96; print(answer)"
```

Verifier scores: **GOOD = 14.00, BAD = 0.00.** Clean discrimination.

## Open work / next steps

1. ✅ **Run11 with curriculum v5** — done. 99% codes have arithmetic; 12.6% responsive.
2. ✅ **Run13 with curriculum v6 (chained refs) + 600K corpus** — done. 98.4% codes have arithmetic; **85.3% responsive**.
3. ✅ **Run verifier on run13 candidates** — done. Oracle@16=21.5%, Oracle@32=25%, mode-vote=4%, weighted vote=4.5%.
4. ✅ **Tune scoring weights via logistic regression** — done with negative result.
   Logistic critic on 6400 labeled candidates (LOO CV) gets **2.0%** — worse
   than mode-vote. Feature `ops` has *negative* weight: every extra
   operation is another chance to pick wrong, but simple-correct vs
   simple-wrong codes are structurally identical. Confirms offline static
   features cannot discriminate at this scale.
   (See `scripts/critic_logistic.py`.)
5. ⏳ **Real critic model**: small NN that reads (problem, code) text together
   and outputs P(correct). Requires GPU training.

## Phase 7.2: End-to-end BLA assembly (offline-done)

The verifier is one piece; the broader BLA spec was never assembled.
Phases 1-5 built each organ separately, but we'd been calling
`procedural_core.forward()` directly throughout Phase 6. Phase 7.2
wires the full architecture:

```
problem
  ↓
EntropyRouter → RETRIEVE
  ↓
TFIDFRetriever.lookup(problem, k=K)
  ↓ returns K similar (problem, python_solution) demos
  ↓
EntropyRouter → SIMULATE
  ↓
procedural_core.generate(few_shot_prompt)
  ↓ may iterate with sampling on retry
  ↓
exec_python → output
  ↓
PALCertifier → CertifierResult (passed?, confidence, details)
  ↓
CommitmentObject { claim, evidence, tests_run, uncertainty,
                  reasoning_trace, reproducibility_packet }
  ↓
return commitment
```

Three offline-built inference variants:

| Script | Components wired | Expected GSM8K acc |
|---|---|---|
| `scripts/bla_inference.py` | core + certifier + commitment | ~1.5% (matches greedy PAL) |
| `scripts/bla_inference_rag.py` | + RETRIEVE (TF-IDF over GSM8K-train) | 5-10% (literature 2-5× from RAG) |
| `scripts/bla_inference_loop.py --rag` | + adaptive retry until certified | 8-15% (RAG + verifier-driven retry) |

All three import cleanly, smoke-tested with synthetic candidates. The
TF-IDF index over 7473 GSM8K-train problems is built and cached at
`/Users/thomasdifiore/bla_artifacts/gsm8k_train_tfidf.pkl`. Retrieval
quality validated: for Janet's eggs problem (gold=18), top-3 demos
are all egg-related (Lisa's breakfast, Linda's egg basket); top
similarity 0.34 — semantic, not noise.

**The compute envelope of the full-loop variant is ~800 generations
for 200 problems (3-4 retries avg, same as SC@8), but uses the
verifier as the picker instead of voting.** This is the architecture's
adaptive-test-time-compute claim made concrete.

## Files (Phase 7.2 additions)

- `verification/pal_certifier.py` — Certifier that wraps the verifier
- `scripts/bla_inference.py` — baseline assembly
- `system2_dca/retrieval_memory.py` — TF-IDF retriever + few-shot prompt
- `scripts/bla_inference_rag.py` — RAG variant
- `scripts/bla_inference_loop.py` — adaptive retry variant
- `scripts/critic_logistic.py` — offline critic experiment (negative result)

## What's gated on the pod returning

1. Run the three BLA variants on run13 checkpoint, n=200
2. Compare accuracies against baselines (greedy 1.5%, SC@32 4.5%, oracle 25%)
3. Falsifiable threshold: full-loop variant > 8%
4. If RAG works as expected (>5%), validates "wire the brain" empirically
5. After: train a learned critic, run14 with cot_pal curriculum mixed in
4. **Add critic-model signal** — train a separate model to grade
   (problem, code, answer) triples for plausibility. Higher-cost but
   potentially much stronger than static features.
5. **Iterative refinement loop** — connect verifier to the action router's
   ASK/RETRY actions: if max score is too low, regenerate candidates with
   a different prompt or temperature.
6. **Move from offline → online** — currently the verifier is a Python
   script run after evaluation. Wire it into the inference path so
   commitments are made in real time.

## Files

- `scripts/verifier.py` — verifier prototype with all 5 signals + 3 aggregation modes
- `scripts/phase6_eval_pal_candidates.py` — eval that saves all N candidates per problem with features
- `scripts/rerank_candidates.py` — simpler vote-based reranker (superseded by verifier.py)
- `system2_dca/curriculum_gsm8k_v2.py` — fixed GSM8K-train source (precondition for verifier to have signal to operate on)

## Cost so far

Phase 6 (this session): ~$110 across 5 training runs + many evals on RunPod B200s.
Phase 7 (this design + verifier): offline, $0.
