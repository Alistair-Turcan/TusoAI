import json
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tusoai.probabilities import _renormalize

_PROMPT_PACKAGE = "tusoai"
_GENERIC_PROMPTS_RESOURCE = "generic_prompts.json"
_DIAGNOSTIC_PROMPTS_RESOURCE = "diagnostic_prompts.json"
_ABLATION_PROMPTS_RESOURCE = "ablation_prompts.json"
_SIMPLIFY_PROMPTS_RESOURCE = "simplify_prompts.json"


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _read_package_text_with_fallback(resource_name: str) -> str:
    resource = resources.files(_PROMPT_PACKAGE).joinpath(resource_name)
    try:
        return resource.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return resource.read_text(encoding="latin-1")


def _load_prompt_json(resource_name: str, override_path: Optional[str] = None) -> dict:
    if override_path is not None:
        text = _read_text_with_fallback(Path(override_path))
    else:
        text = _read_package_text_with_fallback(resource_name)
    return json.loads(text)


def add_general_prompts_and_probability(
    refined_prompts: Dict[str, List[str]],
    probabilities: Dict[str, float],
    generic_prompts_path: Optional[str] = None,
    general_weight: float = 1.0,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """
    Loads the package's built-in generic_prompts.json, adds its "general" prompts into
    refined_prompts["general"], sets probabilities["general"] = general_weight, then
    renormalizes.

    Pass generic_prompts_path only when intentionally loading a custom prompt file.
    """

    generic = _load_prompt_json(_GENERIC_PROMPTS_RESOURCE, generic_prompts_path)

    general_prompts = generic.get("general", [])
    if not isinstance(general_prompts, list):
        prompt_source = generic_prompts_path or _GENERIC_PROMPTS_RESOURCE
        raise ValueError(f'"general" must be a list in {prompt_source}')

    # --- update refined_prompts (dedupe, preserve order) ---
    updated_prompts = {k: list(v) for k, v in refined_prompts.items()}
    updated_prompts.setdefault("general", [])
    seen = set(updated_prompts["general"])
    for p in general_prompts:
        p = (p or "").strip()
        if p and p not in seen:
            updated_prompts["general"].append(p)
            seen.add(p)

    # --- update probabilities ---
    updated_probs = dict(probabilities)

    # ensure all categories have a probability
    for cat in updated_prompts.keys():
        updated_probs.setdefault(cat, 0.0)

    updated_probs["general"] = float(general_weight)

    # renormalize
    updated_probs = _renormalize(updated_probs)
    updated_probs = {k: round(v, 6) for k, v in updated_probs.items()}

    return updated_prompts, updated_probs


def _base_dir_prompt_path(base_dir: Optional[str], resource_name: str) -> Optional[str]:
    if base_dir is None:
        return None
    return str(Path(base_dir) / "tusoai" / resource_name)


def load_diagnostic_prompts(base_dir: Optional[str] = None) -> List[str]:
    """Load built-in diagnostic prompts, independent of the current working directory."""
    data = _load_prompt_json(
        _DIAGNOSTIC_PROMPTS_RESOURCE,
        _base_dir_prompt_path(base_dir, _DIAGNOSTIC_PROMPTS_RESOURCE),
    )
    return list(data.get("diagnostic", []))


def load_ablation_prompts(base_dir: Optional[str] = None) -> List[str]:
    """Load built-in ablation prompts, independent of the current working directory."""
    data = _load_prompt_json(
        _ABLATION_PROMPTS_RESOURCE,
        _base_dir_prompt_path(base_dir, _ABLATION_PROMPTS_RESOURCE),
    )
    return list(data.get("ablation", []))


def load_simplify_prompts(base_dir: Optional[str] = None) -> List[str]:
    """Load built-in simplification prompts, independent of the current working directory."""
    data = _load_prompt_json(
        _SIMPLIFY_PROMPTS_RESOURCE,
        _base_dir_prompt_path(base_dir, _SIMPLIFY_PROMPTS_RESOURCE),
    )
    return list(data.get("simplify", []))


__all__ = [
    "add_general_prompts_and_probability",
    "load_ablation_prompts",
    "load_diagnostic_prompts",
    "load_simplify_prompts",
]
