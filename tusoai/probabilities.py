import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from tusoai.llm import run_prompt

def _renormalize(probs: Dict[str, float]) -> Dict[str, float]:
    """Clamp to a small positive floor, renormalize to sum to 1. If all invalid, use uniform."""
    cleaned = {k: float(max(0.01, v)) for k, v in probs.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        n = len(cleaned)
        return {k: 1.0 / n for k in cleaned} if n else {}
    return {k: v / total for k, v in cleaned.items()}


def _build_general_prompt(
    task_description: str,
    data_available: str,
    categories: List[str],
    initializations: List[str],
    attempt: int = 1
) -> str:
    """
    Construct a general prompt; emphasizes *initial-phase* relevance and JSON-only output.
    This prompt is strict about:
      - using only provided categories (no add/drop)
      - grounding priorities in task + data availability
      - front-loading early progress (initial usefulness)
    """
    cat_block = "\n".join(f"- {c}" for c in categories)
    init_block = "\n".join(f"- {m}" for m in initializations) if initializations else "- (none provided)"
    reminder = "" if attempt == 1 else "\nREMINDER: Output ONLY a single JSON object, no prose, no code fences."

    return f"""<role>
You are a master of machine learning and the domain relevant to this task. You are assigning initial optimization priorities for an AutoML system. You are strict about grounding: the priorities must follow what is implementable and most impactful *first* given the available data.
</role>

<task>
Assign an "initial usefulness" probability to EACH category that reflects what to prioritize first to make early progress. These probabilities are only for the initial phase and can be updated later.
</task>

<scoring_guidance>
- The probabilities across all categories MUST sum to 1.
- Use ONLY the exact category names provided as keys. Do not add, remove, or rename categories.
- Emphasize what is most useful to do FIRST (front-load impact). Later-phase items should receive less probability mass initially.
- Base priorities on the constraints implied by <data_available> and what is realistically actionable early.
- If a category is likely infeasible or low-signal given <data_available>, assign it low probability.
- If uncertain between categories, allocate using common early-phase heuristics:
  data sanity/label integrity/representation/preprocessing typically precede deep modeling and fine-grained tuning.
</scoring_guidance>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<initializations>
{init_block}
</initializations>

<categories_to_score>
{cat_block}
</categories_to_score>
</inputs>

<constraints>
- Output ONLY a single JSON object mapping category -> probability.
- Probabilities must be numeric in [0, 1] and sum to 1 (within floating point tolerance).
- No explanations, no extra fields, no comments, no code fences.
{reminder}
</constraints>

<analysis_step>
Privately:
1) Identify which categories are prerequisite/early-stage given the data (e.g., label quality, representation, preprocessing, leakage control).
2) Identify which categories are high-impact early given the candidate initializations.
3) Allocate probability mass accordingly, keeping lower mass for later-stage refinements.
</analysis_step>

<verification>
Before finalizing:
- Keys EXACTLY match the provided category names (no missing, no extras).
- All values are valid numbers.
- Values sum to 1.
- Output is valid JSON only.
</verification>
""".strip()


def get_category_probabilities(
    function_name: str,
    task_description: str,
    data_available: str,
    categories: List[str],
    initializations: List[str],
    *,
    cache_dir: str,
    max_attempts: int = 3,
    clear: bool = True,  # delete + rebuild; if False, load if available
    num_shuffles: int = 10,  # NEW: number of random orderings to average over
) -> Dict[str, float]:
    """
    Cache:
      - folder/probabilities.json (load if clear=False, rebuild if clear=True)

    Behavior:
      - Runs up to `num_shuffles` random shuffles of the category ordering.
      - For each shuffle, asks the same question and parses JSON probabilities.
      - Averages probabilities over all successful parses (then renormalizes).
      - If none succeed, returns uniform.

    Returns:
      {category: probability} with probabilities summing to 1.
    """
    if not categories:
        return {}

    probs_path = Path(cache_dir) / f"kt_data/{function_name}/probabilities.json"

    # Clear cached probabilities if requested
    if clear and probs_path.exists():
        try:
            probs_path.unlink()
            print(f"[CLEAR] Removed cache: {probs_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {probs_path}: {e}")

    # Load cache if available and not clearing
    if (not clear) and probs_path.exists():
        print(f"[CACHE] Loading probabilities from {probs_path}")
        return json.loads(probs_path.read_text(encoding="utf-8"))

    cats_original = list(categories)

    # Accumulate averages over successful shuffles
    accum = {c: 0.0 for c in cats_original}
    successful = 0

    # Pre-generate shuffled orders (unique-ish, but no need to enforce uniqueness)
    orders: List[List[str]] = []
    for _ in range(max(1, int(num_shuffles))):
        order = cats_original[:]
        random.shuffle(order)
        orders.append(order)

    def _process_one_shuffle(cats: List[str]) -> Optional[Dict[str, float]]:
        parsed: Optional[Dict[str, float]] = None

        for attempt in range(1, max_attempts + 1):
            prompt = _build_general_prompt(
                task_description=task_description,
                data_available=data_available,
                categories=cats,  # shuffled ordering
                initializations=initializations,
                attempt=attempt,
            )

            reply = run_prompt(prompt)
            if isinstance(reply, tuple):
                reply = reply[0]

            # Strip accidental code fences
            reply = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", str(reply).strip(), flags=re.MULTILINE)

            try:
                candidate = json.loads(reply)
            except Exception:
                candidate = None

            if isinstance(candidate, dict):
                # Keep only known categories; fill missing with 0; ignore extras
                scores: Dict[str, float] = {}
                for c in cats_original:
                    try:
                        scores[c] = float(candidate.get(c, 0.0))
                    except Exception:
                        scores[c] = 0.0

                # Accept only if there is some non-zero signal
                if sum(v for v in scores.values() if isinstance(v, (int, float))) > 0.0:
                    parsed = scores
                    break

        if parsed is None:
            return None

        return _renormalize(parsed)

    max_workers = max(1, len(orders))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_process_one_shuffle, orders))

    for weights in results:
        if weights is None:
            continue

        for c in cats_original:
            accum[c] += float(weights.get(c, 0.0))

        successful += 1

    if successful == 0:
        print("Warning: could not parse usable JSON from LLM for any shuffle. Using uniform probabilities.")
        weights = {c: round(1.0 / len(cats_original), 6) for c in cats_original}
        probs_path.parent.mkdir(parents=True, exist_ok=True)
        probs_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
        print(f"[SAVED] Probabilities written to {probs_path}")
        return weights

    # Average and renormalize
    avg = {c: accum[c] / successful for c in cats_original}
    avg = _renormalize(avg)
    avg = {k: round(v, 6) for k, v in avg.items()}

    probs_path.parent.mkdir(parents=True, exist_ok=True)
    probs_path.write_text(json.dumps(avg, indent=2), encoding="utf-8")
    print(f"[SAVED] Probabilities written to {probs_path} (averaged over {successful} shuffles)")

    return avg
