import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from tusoai.llm import run_prompt

classification_initializations = [
    "logistic regression",
    "XGBoost",
    "random forest",
    "MLP classifier"
]

# ---------------- Prompt builder ------------------------------------ #
def make_initialization_prompt(task_description: str,
                               data_available: str,
                               num_init: int) -> str:
    few_shot = "\n".join(f"<m>{ex}</m>" for ex in classification_initializations)

    return f"""<role>
You are a master of machine learning and the domain relevant to this task. You design strong baseline model initializations for an AutoML system. You are pragmatic about feasibility given the available data and avoid overly specific pipelines.
</role>

<task>
Propose concise model initializations (baseline method descriptions) that are appropriate starting points for this specific task and data.
</task>

<style_reference>
Below is an example list of generic model initializations for a classification task:
{few_shot}
</style_reference>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<num_initializations>
{num_init}
</num_initializations>
</inputs>

<success_criteria>
- Output exactly <num_initializations> model initializations.
- Each initialization is directly relevant to <task_description>.
- Each initialization is feasible given <data_available> (no new labels, sensors, annotations, or external datasets unless clearly already available).
- Each initialization is a general method / model family / high-level architecture description (not a fully specified pipeline).
- The set is diverse (not minor variants of the same method).
- Focus on predictive performance; do not mention runtime, scalability, logging, visualization, evaluation setup, or tooling.
</success_criteria>

<constraints>
- Output one per line.
- Each line must be wrapped exactly like: <m>...</m>
- Keep each line short (typically 1-6 words, optionally with a short qualifier).
- Do not include explanations or extra text outside of <m>...</m>.
</constraints>

<analysis_step>
Privately:
1) Infer the likely learning setting implied by <task_description> and <data_available> (e.g., tabular/text/time-series/image; supervised/weakly supervised; class imbalance; etc.).
2) Choose strong, common baselines and a few task-appropriate modern alternatives.
3) Ensure every proposal is implementable with the provided data modality and labels.
4) Ensure diversity across model families.
</analysis_step>

<verification>
Before finalizing:
- Count: exactly <num_initializations> lines.
- Feasibility: each is implementable with <data_available>.
- Relevance: each makes sense for <task_description>.
- Format: only <m>...</m> lines, nothing else.
</verification>
""".strip()

def parse_m_tags(text: str) -> List[str]:
    """Extract <m> ... </m> contents, stripping whitespace."""
    return [m.strip() for m in re.findall(r"<m>(.*?)</m>", text, flags=re.S) if m.strip()]

def get_initializations(
    function_name: str,
    task_description: str,
    data_available: str,
    num_init: int,
    *,
    cache_dir: str,
    clear: bool = True,
) -> List[str]:
    """
    If folder/initial_solutions.json exists and clear=False, loads and returns it.
    If clear=True, deletes folder/initial_solutions.json (if present) and rebuilds.
    """
    inits_path = Path(cache_dir) / f"kt_data/{function_name}/initial_solutions.json"

    # Clear cached initializations if requested
    if clear and inits_path.exists():
        try:
            inits_path.unlink()
            print(f"[CLEAR] Removed cache: {inits_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {inits_path}: {e}")

    # Load cache if available and not clearing
    if (not clear) and inits_path.exists():
        print(f"[CACHE] Loading initial solutions from {inits_path}")
        return json.loads(inits_path.read_text(encoding="utf-8"))

    prompt = make_initialization_prompt(task_description, data_available, num_init)
    reply = run_prompt(prompt)
    if isinstance(reply, tuple):
        reply = reply[0]

    inits = parse_m_tags(reply)
    if not inits:
        print("Warning: No initializations extracted from LLM response.")

    # De-duplicate while preserving order (case-insensitive)
    seen = set()
    unique_inits = []
    for m in inits:
        k = m.strip().lower()
        if k and k not in seen:
            seen.add(k)
            unique_inits.append(m.strip())

    # Save to folder/initial_solutions.json
    inits_path.parent.mkdir(parents=True, exist_ok=True)
    inits_path.write_text(json.dumps(unique_inits, indent=2), encoding="utf-8")
    print(f"[SAVED] Initial solutions written to {inits_path}")

    return unique_inits

def refine_initializations_with_summaries(
    function_name: str,
    initializations: List[str],
    summaries: List[Dict[str, Any]],   # list of dicts with "title" + "summary"
    task_description: str,
    data_available: str,
    *,
    cache_dir: str,
    clear: bool = True,
) -> List[str]:
    """
    If folder/refined_solutions.json exists and clear=False, loads and returns it.
    If clear=True, deletes folder/refined_solutions.json (if present) and rebuilds.

    For each paper summary, ask the LLM to append ONE new initialization if it is
    a genuinely new, implementable model family/architecture for this task; otherwise
    leave the list unchanged.
    """
    refined_path = Path(cache_dir) / f"kt_data/{function_name}/refined_solutions.json"

    if clear and refined_path.exists():
        try:
            refined_path.unlink()
            print(f"[CLEAR] Removed cache: {refined_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {refined_path}: {e}")

    if (not clear) and refined_path.exists():
        print(f"[CACHE] Loading refined solutions from {refined_path}")
        return json.loads(refined_path.read_text(encoding="utf-8"))

    current = initializations.copy()
    current_lower = {x.strip().lower() for x in current if isinstance(x, str) and x.strip()}

    def _process_one(item: Dict[str, Any]) -> tuple[str, str | None, str]:
        title = item.get("title", "Untitled")
        bullet_points = item.get("summary", "")

        prompt = f"""<role>
You are a master of machine learning and the domain relevant to this task. You curate a list of baseline model initializations for an AutoML system. You are extremely conservative: you only add an initialization if it is unquestionably relevant and implementable with the available data.
</role>

<task>
Given a paper summary, decide whether it suggests ONE genuinely new model initialization (model family / architecture) that should be added to the current list for this task and data. Otherwise return NO_CHANGE.
</task>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<current_initializations>
{current}
</current_initializations>

<paper>
<title>{title}</title>
<key_points>
{bullet_points}
</key_points>
</paper>
</inputs>

<decision_rule>
Output a new initialization ONLY IF ALL are true:
1) It is directly applicable to <task_description>.
2) It is implementable using ONLY <data_available> (no new labels, sensors, annotations, or external datasets unless explicitly already available).
3) It is a genuinely new model family/architecture not already covered by <current_initializations> (including near-duplicates/synonyms).
4) It is likely useful as a baseline starting point (general method / model family / high-level architecture), not a fully specified pipeline.
5) You are 100% confident it adds signal; if there is any doubt, output NO_CHANGE.
</decision_rule>

<naming_rules>
- If the paper is a named published method, DO NOT output the method name.
  Instead, output the generic architecture/family description (e.g., "graph neural network with edge features", not "CoolGNN-XL").
- Keep it short (1-8 words; optionally with a brief qualifier).
- Avoid implementation details, training recipes, or hyperparameter values.
</naming_rules>

<constraints>
- Output EXACTLY ONE line.
- The line must be wrapped exactly as: <m>...</m>
- Allowed outputs are ONLY:
  - <m>NO_CHANGE</m>
  - <m>New Initialization</m> (replace with what the initialization actually is)
- No other text.
</constraints>

<analysis_step>
Privately:
1) Interpret the task/data modality implied by <task_description> and <data_available>.
2) Extract model-family ideas from the paper that match the modality and label setting.
3) Reject anything that needs extra data/annotations.
4) Reject anything that is already covered by the current list (including near-duplicates).
5) If one high-confidence new baseline remains, output it; otherwise output NO_CHANGE.
</analysis_step>

<verification>
Before finalizing:
- Feasibility: implementable with <data_available> only.
- Novelty: not a near-duplicate of <current_initializations>.
- Relevance: clearly suitable for <task_description>.
- Format: exactly one <m>...</m> line, no extra text.
</verification>
""".strip()

        reply = run_prompt(prompt)
        if isinstance(reply, tuple):
            reply = reply[0]

        matches = re.findall(r"<m>(.*?)</m>", reply, flags=re.S)
        if not matches:
            return (title, None, reply)

        candidate = (matches[0] or "").strip()
        if not candidate or candidate.upper() == "NO_CHANGE":
            return (title, None, reply)

        return (title, candidate, reply)

    max_workers = max(1, len(summaries))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_process_one, summaries))

    for title, candidate, reply in results:
        if not candidate:
            if reply and "<m>" not in reply:
                print("[WARN] No <m> tag found in LLM reply.")
            continue

        cand_norm = candidate.lower()
        if cand_norm in current_lower:
            continue

        current.append(candidate)
        current_lower.add(cand_norm)
        print(f"[ADD] {candidate}  (from: {title})")

    refined_path.parent.mkdir(parents=True, exist_ok=True)
    refined_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    print(f"[SAVED] Refined solutions written to {refined_path}")

    return current

def _parse_m_tags(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"<m>(.*?)</m>", text, flags=re.S) if m.strip()]

def _dedupe_keep_order_ci(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        k = x.lower()
        if not x or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def retain_top_initial_solutions(
    function_name: str,
    refined_solutions: List[str],
    task_description: str,
    data_available: str,
    num_init: int,   # kept for compatibility; treated as a soft target, not enforced
    *,
    cache_dir: str,
    clear: bool = True,
    max_attempts: int = 3,
) -> List[str]:
    """
    Filter/merge refined_solutions down to a minimal set of diverse, strong baseline initializations
    for this task and available data. The LLM may merge/rename solutions by combining from the
    provided list.

    NOTE: This version does NOT enforce returning exactly `num_init` items. `num_init` is treated
    as a soft upper bound / target.

    Cache:
      - folder/final_solutions.json (load if clear=False, rebuild if clear=True)

    Returns:
      final_solutions: List[str]
    """
    out_path = Path(cache_dir) / f"kt_data/{function_name}/final_solutions.json"

    if clear and out_path.exists():
        try:
            out_path.unlink()
            print(f"[CLEAR] Removed cache: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {out_path}: {e}")

    if (not clear) and out_path.exists():
        print(f"[CACHE] Loading final solutions from {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    if not refined_solutions:
        print("[WARN] No refined_solutions provided; returning empty list.")
        return []

    candidates = _dedupe_keep_order_ci(refined_solutions)
    cand_block = "\n".join(f"<m>{c}</m>" for c in candidates)

    base_prompt = f"""<role>
You are a master of machine learning and the domain relevant to this task. You curate a small set of strong baseline model initializations for an AutoML system. You are careful about feasibility given the available data and you reduce redundancy by merging near-duplicates when useful.
</role>

<task>
From the provided candidate initializations, select and possibly merge/rename items to produce a final, minimal set of strong, diverse baselines for this task and data.
</task>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<candidate_initializations>
{cand_block}
</candidate_initializations>
</inputs>

<decision_rule>
Only keep (or merged-keep) an initialization if ALL are true:
1) It is clearly implementable given <data_available>.
2) It is directly relevant to <task_description> (correct modality/setting).
3) It is a strong likely baseline (reasonable default to try early in AutoML).
4) It is not redundant with another kept item (different family/assumptions). If redundant, merge into one.
If uncertain about feasibility or relevance, drop it.
</decision_rule>

<merging_rules>
- You MAY merge/rename items ONLY by combining from <candidate_initializations>.
- Do NOT introduce new model families not implied by the candidates.
- If you merge, choose a clearer generic description that still matches the original candidates.
</merging_rules>

<constraints>
- Output ONLY initialization lines; no explanations.
- One per line, each wrapped exactly as: <m>Initialization</m>
- Keep each line short (1-8 words; optionally a brief qualifier).
- Do not output duplicates or near-duplicates.
</constraints>

<analysis_step>
Privately:
1) Drop infeasible items (need data not in <data_available>).
2) Group near-duplicates/synonyms and merge them.
3) Ensure diversity across remaining items (different modeling assumptions).
4) Prioritize likely strongest baselines for <task_description>.
5) Output the minimal set (optionally at or below <soft_target_max_initializations>).
</analysis_step>

<verification>
Before finalizing:
- Every output maps to one or more provided candidates (selection or merge).
- Feasibility with <data_available>.
- Non-redundancy across outputs.
- Format: only <m>...</m> lines, no extra text.
</verification>
""".strip()

    chosen: Optional[List[str]] = None

    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += "\n\n<constraints>\nREMINDER: Output only <m>...</m> lines and nothing else.\n</constraints>"

        reply = run_prompt(prompt)
        if isinstance(reply, tuple):
            reply = reply[0]

        parsed = _dedupe_keep_order_ci(_parse_m_tags(reply))
        if parsed:
            chosen = parsed
            break
        else:
            chosen = parsed  # keep last attempt for warning path

    if not chosen:
        print(f"[WARN] Could not parse final solutions after {max_attempts} attempts; returning deduped candidates.")
        chosen = candidates

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(f"[SAVED] Final solutions written to {out_path}")

    return chosen
