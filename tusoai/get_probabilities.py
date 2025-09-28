import re
import json
import math
import unicodedata
from typing import Dict, List, Optional

# Assumes `run_prompt`, `TEMP`, and `LLM_MODEL` exist in your environment.

def run_prompt(prompt, client, temperature, model):
    if model == 'gpt-5':
        temperature=1.0
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract full content from the assistant's reply
        text = resp.choices[0].message.content.strip()
        #print(resp.usage)
        return text
    except Exception as e:
        print("LLM call failed:", e)
        return ""


def _strip_code_fences(s: str) -> str:
    """Remove surrounding markdown code fences if present."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s.strip()

def _safe_json_extract(s: str):
    """Try to parse JSON directly; fallback to extracting the first {...} block."""
    s = _strip_code_fences(s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None

def _renormalize(probs: Dict[str, float]) -> Dict[str, float]:
    """Clamp negatives to 0, renormalize to sum to 1. If all zeros, use uniform."""
    cleaned = {k: float(max(0.0, v)) for k, v in probs.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        n = len(cleaned)
        return {k: 1.0 / n for k in cleaned} if n else {}
    return {k: v / total for k, v in cleaned.items()}

def _sanitize_filename(name: str) -> str:
    """Make a safe filename from an arbitrary string."""
    # Normalize unicode, remove accents
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    # Replace non-alnum with underscores, collapse repeats
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    # Trim very long names
    return name[:120] if len(name) > 120 else name

def save_probabilities(weights: Dict[str, float], task_description: str, directory: str = ".") -> str:
    """Write probabilities to {directory}/{sanitized_task_description}_probabilities.json and return the path."""
    fname = f"{task_description}_probabilities.json"
    path = f"{directory.rstrip('/')}/{fname}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, sort_keys=True)
    return path

def load_probabilities(path: str) -> Dict[str, float]:
    """Load probabilities dict from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure numeric and normalized on load
    data = {k: float(v) for k, v in data.items()}
    return _renormalize(data)


def _build_general_prompt(
    task_description: str,
    data_available: str,
    categories: List[str],
    initializations: List[str],
    attempt: int = 1
) -> str:
    """Construct a general prompt; emphasizes 'initial usefulness' and JSON-only output."""
    cat_block = "\n".join(f"- {c}" for c in categories)
    init_block = "\n".join(f"- {m}" for m in initializations) if initializations else "- (none provided)"

    reminder = "" if attempt == 1 else "\nREMINDER: Output ONLY a single JSON object, no prose, no code fences."

    prompt = f"""
You are an expert ML/DS/engineering strategist. Given the information below, assign an "initial usefulness" probability to EACH category, reflecting what to prioritize first to make early progress. These probabilities are for the initial phase and can be updated later. Higher probability ⇒ earlier/higher priority at the start. The probabilities across all categories must sum to 1. Use only the exact category names provided.

Context:
- Task: "{task_description}"
- Data available / constraints: "{data_available}"
- Candidate initializations / approaches we might try:
{init_block}

Categories to score (use these keys EXACTLY; do not add or drop any):
{cat_block}

Guidance:
- Emphasize what is most useful to do FIRST (front-load impact); later-phase items should receive less mass initially.
- Consider data readiness, preprocessing, representation, target/label availability, modeling choices, training setup, evaluation, and practical constraints implied by the data.
- If uncertain, allocate by common early-phase heuristics (data cleaning/preprocessing/representation often precede modeling/hyperparameter tuning).

Output requirements:
- Output ONLY a single JSON object mapping category → probability in [0, 1].
- The values MUST sum to 1 (within floating point tolerance).
- Do NOT include explanations, comments, extra fields, or code fences.
{reminder}
""".strip()

    return prompt

def get_category_probabilities(
    task_description: str,
    data_available: str,
    categories: List[str],
    initializations: List[str],
    client,
    prompts: Optional[Dict[str, str]] = None,  # kept for compatibility; not required
    temperature: Optional[float] = 0.5,       # defaults to TEMP if None
    model: Optional[str] = None,               # defaults to LLM_MODEL if None
    max_attempts: int = 3
) -> Dict[str, float]:
    """
    Ask the LLM to assign an 'initial usefulness' probability to each category.
    Retries up to `max_attempts` times and falls back to uniform if parsing fails.
    Returns {category: probability} with probabilities summing to 1.
    """
    if not categories:
        return {}

    # If caller supplied a prompts dict (category → prompt), we could intersect,
    # but by default we keep the full provided `categories`.
    cats = list(categories)

    parsed: Optional[Dict[str, float]] = None
    for attempt in range(1, max_attempts + 1):
        prompt = _build_general_prompt(
            task_description=task_description,
            data_available=data_available,
            categories=cats,
            initializations=initializations,
            attempt=attempt
        )
        reply = run_prompt(
            prompt,
            client=client,
            temperature=(temperature if temperature is not None else TEMP),
            model=(model if model is not None else LLM_MODEL),
        )
        candidate = _safe_json_extract(reply)

        if isinstance(candidate, dict):
            # Keep only known categories; fill missing with 0
            scores: Dict[str, float] = {}
            for c in cats:
                try:
                    scores[c] = float(candidate.get(c, 0.0))
                except Exception:
                    scores[c] = 0.0

            # If everything is zero, try again
            if sum(v for v in scores.values() if isinstance(v, (int, float))) > 0.0:
                parsed = scores
                break
        # else: try again

    if parsed is None:
        # Fallback to uniform
        print("Warning: could not parse usable JSON from LLM after retries. Using uniform probabilities.")
        return {c: round(1.0 / len(cats), 6) for c in cats}

    weights = _renormalize(parsed)
    # Round for stability/pretty-printing
    return {k: round(v, 6) for k, v in weights.items()}


