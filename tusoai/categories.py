import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from tusoai.llm import run_prompt

classification_categories = [
    "regularisation",
    "feature_engineering",
    "hyperparameter_tuning",
    "sampling",
    "ensemble_methods",
    "calibration",
    "feature_selection",
]

def get_task_categories(
    function_name: str,
    task_description: str,
    data_available: str,
    num_cat: int,
    *,
    cache_dir: str,
    clear: bool = True,
) -> list:
    """
    If folder/initial_categories.json exists and clear=False, loads and returns it.
    If clear=True, deletes folder/initial_categories.json (if present) and rebuilds categories.
    """
    categories_path = Path(cache_dir) / f"kt_data/{function_name}/initial_categories.json"

    # Clear cached categories if requested
    if clear and categories_path.exists():
        try:
            categories_path.unlink()
            print(f"[CLEAR] Removed cache: {categories_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {categories_path}: {e}")

    # Load cache if available and not clearing
    if (not clear) and categories_path.exists():
        print(f"[CACHE] Loading categories from {categories_path}")
        return json.loads(categories_path.read_text(encoding="utf-8"))

    prompt = f"""<role>
You are a master of machine learning and the domain relevant to this task. You specialize in designing AutoML search spaces and optimization "axes" that directly improve model performance.
</role>

<task>
Construct a list of concise, task-relevant optimization categories for an LLM-powered AutoML system.
</task>

<success_criteria>
- Categories are maximally relevant to <task_description> and <data_available>.
- Each category corresponds to a distinct axis of performance improvement (not duplicates or near-duplicates).
- Categories are actionable as an AutoML search dimension (something you could vary/ablate/optimize).
- Focus strictly on predictive performance (accuracy/quality/robustness/calibration/generalization), not runtime, scalability, logging, visualization, interpretability, or post-evaluation.
- Uses <reference_categories> only as inspiration; do not copy generic buckets unless they truly apply.
</success_criteria>

<constraints>
- Output at most <num_categories> categories.
- One category per line.
- Each line must be formatted exactly as: <c>Category Name</c>
- Category names must be short (1-6 words), specific, and domain-aware.
- Do not include explanations in the final output.
</constraints>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<reference_categories>
{classification_categories}
</reference_categories>

<num_categories>
{num_cat}
</num_categories>
</inputs>

<analysis_step>
Before producing the final list, consider (silently) what modeling interventions are most likely to move performance for this task given the available data and the LLM system. Prioritize categories that (a) leverage domain structure, (b) address likely failure modes, and (c) align with how AutoML will actually explore variants.
</analysis_step>

<verification>
Check that:
1) No category is about runtime/engineering/non-performance concerns.
2) Categories are non-overlapping and each is a true optimization axis.
3) Formatting matches the required <c>...</c> schema exactly.
</verification>

<output>
Return only the category lines.
</output>
"""

    reply = run_prompt(prompt)
    if isinstance(reply, tuple):
        reply = reply[0]

    matches = re.findall(r"<c>(.*?)</c>", reply)
    categories = [m.strip() for m in matches if m.strip()]

    if not categories:
        print("Warning: No categories extracted from LLM response.")

    # Save to folder/initial_categories.json
    categories_path.parent.mkdir(parents=True, exist_ok=True)
    categories_path.write_text(json.dumps(categories, indent=2), encoding="utf-8")
    print(f"[SAVED] Categories written to {categories_path}")

    return categories

def refine_categories_with_summaries(
    function_name: str,
    categories: List[str],
    summaries: List[Dict[str, Any]],
    task_description: str,
    data_available: str,
    *,
    cache_dir: str,
    clear: bool = True,
) -> List[str]:
    """
    Cache behavior:
      - If folder/refined_categories.json exists and clear=False, loads and returns it.
      - If clear=True, deletes folder/refined_categories.json (if present) and rebuilds.

    Update behavior:
      - ONLY appends NEW categories (does not rename/merge/remove existing categories).
      - Prints progress as:
          Added category X with paper {title} (k/N)
          Didn't add category with paper {title} (k/N)
        where k is the number of categories added so far, N is number of summaries processed.
    """
    refined_path = Path(cache_dir) / f"kt_data/{function_name}/refined_categories.json"

    if clear and refined_path.exists():
        try:
            refined_path.unlink()
            print(f"[CLEAR] Removed cache: {refined_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {refined_path}: {e}")

    if (not clear) and refined_path.exists():
        print(f"[CACHE] Loading refined categories from {refined_path}")
        return json.loads(refined_path.read_text(encoding="utf-8"))

    current = categories.copy()
    current_lower = {c.strip().lower() for c in current if c and c.strip()}

    added_count = 0
    N = len(summaries)
    if N == 0:
        print("[WARN] No summaries provided; skipping refinement.")
        refined_path.parent.mkdir(parents=True, exist_ok=True)
        refined_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"[SAVED] Refined categories written to {refined_path}")
        return current

    def _process_one(item: Dict[str, Any]) -> tuple[str, str | None]:
        title = item.get("title", "Untitled")
        bullet_points = item.get("summary", "")

        prompt = f"""<role>
You are a master of machine learning and the domain relevant to this task. You are extremely conservative about adding new optimization categories: you only add one if it is unquestionably relevant and feasible given the data.
</role>

<task>
Given a paper summary, decide whether it introduces a NEW optimization axis that is missing from the current category list AND is implementable with the available data for this AutoML system.
</task>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<current_categories>
{current}
</current_categories>

<paper>
<title>{title}</title>
<key_points>
{bullet_points}
</key_points>
</paper>
</inputs>

<decision_rule>
Return a new category ONLY IF ALL are true:
1) The paper suggests a distinct optimization axis not already covered by <current_categories>.
2) The axis is directly applicable to <task_description>.
3) The axis is implementable using ONLY <data_available> (no new labels, sensors, annotations, or external datasets unless clearly already available).
4) Adding it is very likely to increase predictive performance (generalization/robustness/calibration/accuracy) rather than just adding noise.
5) You are 100% confident the category is relevant; if there is any doubt, choose NO_CHANGE.
</decision_rule>

<success_criteria>
- If you add a category, it is:
  - A true "axis" for AutoML to vary/ablate/optimize (not a specific paper name, not a one-off trick).
  - Non-overlapping with existing categories (not a synonym or minor variant).
  - Short (1-6 words), specific, and domain-aware.
- If you do not add a category, you return NO_CHANGE.
</success_criteria>

<constraints>
- Output EXACTLY ONE line.
- The line must be formatted exactly as: <c>...</c>
- Allowed outputs are ONLY:
  - <c>NO_CHANGE</c>
  - <c>New Category Name</c>
- No additional text, punctuation, or explanation.
</constraints>

<analysis_step>
Privately apply the <decision_rule> with a very high bar. If the paper does not clearly add a new, feasible, high-signal optimization axis for this specific task/data, return <c>NO_CHANGE</c>.
</analysis_step>

<verification>
Before finalizing:
- Confirm the proposed category is not already present in <current_categories> (including near-duplicates/synonyms).
- Confirm it is feasible given <data_available>.
- Confirm the output is exactly one line in <c>...</c>.
</verification>
""".strip()

        reply = run_prompt(prompt)
        if isinstance(reply, tuple):
            reply = reply[0]

        matches = re.findall(r"<c>(.*?)</c>", reply, flags=re.S)
        if not matches:
            return (title, None)

        candidate = (matches[0] or "").strip()
        if not candidate or candidate.upper() == "NO_CHANGE":
            return (title, None)

        return (title, candidate)

    # Run all summaries in parallel with as many threads as summaries
    with ThreadPoolExecutor(max_workers=N) as ex:
        results = list(ex.map(_process_one, summaries))

    # Apply results deterministically in original order
    for title, candidate in results:
        if not candidate:
            print(f"Didn't add category with paper {title} ({added_count}/{N})")
            continue

        cand_norm = candidate.lower()
        if cand_norm in current_lower:
            print(f"Didn't add category with paper {title} ({added_count}/{N})")
            continue

        current.append(candidate)
        current_lower.add(cand_norm)
        added_count += 1
        print(f"Added category {candidate} with paper {title} ({added_count}/{N})")

    # Save
    refined_path.parent.mkdir(parents=True, exist_ok=True)
    refined_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    print(f"[SAVED] Refined categories written to {refined_path}")

    return current

def retain_top_categories(
    function_name: str,
    refined_categories: List[str],
    task_description: str,
    data_available: str,
    num_cat: int,
    *,
    cache_dir: str,
    clear: bool = True,
    max_attempts: int = 3,
) -> List[str]:
    """
    Retain a minimized set of categories from refined_categories, keeping only the most useful
    for the AutoML system given task_description + data_available. The LLM may merge/combine
    categories if feasible.

    NOTE: This version does NOT enforce returning exactly `num_cat` categories. `num_cat` is
    treated as an upper bound / soft target (or can be ignored by the model), but the function
    will accept any number of returned categories >= 1.

    Cache:
      - folder/final_categories.json (load if clear=False, rebuild if clear=True)
    """
    out_path = Path(cache_dir) / f"kt_data/{function_name}/final_categories.json"

    if clear and out_path.exists():
        try:
            out_path.unlink()
            print(f"[CLEAR] Removed cache: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {out_path}: {e}")

    if (not clear) and out_path.exists():
        print(f"[CACHE] Loading final categories from {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    if not refined_categories:
        print("[WARN] No refined categories provided; returning empty list.")
        return []

    cat_block = "\n".join(f"- {c}" for c in refined_categories)

    prompt_base = f"""<role>
You are a master of machine learning and the domain relevant to this task. You are designing the final, minimal set of optimization categories (AutoML search axes) to prioritize. You are careful, pragmatic, and avoid redundancy.
</role>

<task>
Select and possibly merge categories from the provided list to produce the final set of categories for this AutoML system, given the task and available data.
</task>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<refined_categories>
{cat_block}
</refined_categories>


</inputs>

<success_criteria>
- Each output category is a distinct optimization axis that can be explored in AutoML.
- Categories are maximally relevant to <task_description> and feasible given <data_available>.
- Redundant / overlapping categories are merged into a single clearer axis.
- Categories that appear irrelevant given <data_available> should be deprioritized or removed only when confidence is high.
- Do NOT invent entirely new axes; only select and/or merge from <refined_categories>.
- Names are concise (1-6 words), specific, and domain-aware.
- Prefer a short, high-signal list. You MAY use <soft_target_max_categories> as a loose upper bound, but correctness and relevance matter more than hitting a specific count.
</success_criteria>

<merging_rules>
- Merge only when two or more categories represent essentially the same optimization axis or one is a strict subset of another.
- When merging, choose a name that covers the union without becoming vague.
- Prefer fewer, higher-signal categories over many narrow, low-signal ones.
</merging_rules>

<constraints>
- Output ONLY category lines; no explanations.
- One category per line.
- Each line must be formatted exactly as: <c>Category Name</c>
</constraints>

<analysis_step>
Privately:
1) Remove categories that are impossible or irrelevant given <data_available>.
2) Group near-duplicates / synonyms / strict-subset categories.
3) Merge groups into clear, non-overlapping axes.
4) Rank by expected impact on predictive performance for <task_description>.
5) Output a minimal set that covers the biggest performance levers (optionally keeping the count at or below <soft_target_max_categories> if it doesn't reduce relevance).
</analysis_step>

<verification>
Before finalizing:
- Ensure every output category maps to one or more items from <refined_categories> (via selection or merge).
- Ensure no output category is a near-duplicate of another.
- Confirm feasibility with <data_available>.
- Confirm formatting: each line exactly <c>...</c> with no extra text.
</verification>
""".strip()

    final: Optional[List[str]] = None

    for attempt in range(1, max_attempts + 1):
        prompt = prompt_base
        if attempt > 1:
            prompt += "\n\n<constraints>\nREMINDER: Output only <c>...</c> lines and nothing else.\n</constraints>"

        reply = run_prompt(prompt)
        if isinstance(reply, tuple):
            reply = reply[0]

        matches = [m.strip() for m in re.findall(r"<c>(.*?)</c>", reply, flags=re.S) if m.strip()]
        # Deduplicate while preserving order
        seen = set()
        matches = [c for c in matches if not (c.lower() in seen or seen.add(c.lower()))]

        if matches:
            final = matches
            break
        else:
            final = matches  # keep last attempt for warning path

    if not final:
        print(f"[WARN] Could not extract categories from LLM after {max_attempts} attempts.")
        return refined_categories

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"[SAVED] Final categories written to {out_path}")

    return final
