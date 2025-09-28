"""
A minimal orchestrator module exposing exactly 7 functions.
No config, no main — import and call the functions in order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- Imports from your package ---
from tusoai import initialize_llm
from tusoai import extract_literature
from tusoai import summarize_literature
from tusoai import construct_categories
from tusoai import construct_prompts
from tusoai import construct_initializations
from tusoai import get_probabilities
from tusoai import optimizer

# 1) initialize

def initialize(api_key: str, openrouter: bool = True):
    return initialize_llm.init(api_key, openrouter=openrouter)


# 2) make_summaries (runs extraction + summarization)

def make_summaries(task_description: str, api_key: str, top_n: int, client: Any, model: str) -> Dict[str, str]:
    papers = extract_literature.run_extraction(task_description, api_key=api_key, top_n=top_n)
    return summarize_literature.summarise_papers(papers, client, model)


# 3) make_categories

def make_categories(
    task_description: str,
    data_available: List[str],
    num_cat: int,
    summaries: List[str],
    *,
    client: Any,
    model: str,
) -> List[str]:

    categories = construct_categories.get_task_categories(
        task_description, data_available, num_cat=num_cat, client=client, model=model
    )
    categories = construct_categories.refine_categories_with_summaries(categories,
                                                                       summaries,
                                                                       task_description,
                                                                       data_available,
                                                                       client=client,
                                                                       model=model)
    return categories


# 4) make_instructions (prompts)

def make_instructions(
    task_description: str,
    data_available: List[str],
    categories: List[str],
    summaries: Dict[str, str],
    instruction_count: int,
    *,
    client: Any,
    model: str,
) -> Dict[str, List[str]]:
    # Generate base prompts per category
    prompts = construct_prompts.generate_prompts_for_categories(
        task_description,
        data_available,
        categories,
        instruction_count,
        client,
        model=model,
    )
    # Augment from paper summaries; merge inline without helpers
    new_from_summaries = construct_prompts.generate_prompts_from_summaries(
        task_description,
        summaries,
        categories,
        data_available,
        existing_prompts=prompts,
        n_new_min=5,
        n_new_max=15,
        client=client,
        model=model,
    )
    for _paper_title, per_cat in new_from_summaries.items():
        for cat, new_list in per_cat.items():
            prompts.setdefault(cat, []).extend(new_list)
    return prompts


# 5) make_solutions (Solutions = initializations, as requested)

def make_solutions(
    task_description: str,
    data_available: List[str],
    num_init: int,
    summaries: Dict[str, str],  
    *,
    client: Any,
    model: str,
) -> List[Dict[str, Any]]:
    # Per request: Solutions == initializations (no refinement step)
    initializations = construct_initializations.get_initializations(
        task_description,
        data_available,
        num_init,
        client=client,
        model=model,
    )
    
    initializations = construct_initializations.refine_initializations_with_summaries(
        initializations,
        summaries,
        task_description=task_description,
        data_available=data_available,
        client=client,
        model=model
    )


    return initializations


# 6) make_probabilities

def make_probabilities(
    task_description: str,
    data_available: List[str],
    categories: List[str],
    solutions: List[Dict[str, Any]],
    *,
    client: Any,
    model: str,
) -> Dict[str, float]:
    return get_probabilities.get_category_probabilities(
        task_description=task_description,
        data_available=data_available,
        categories=categories,
        initializations=solutions,
        client=client,
        model=model,
    )


# 7) discover_method — runs optimizer and persists full_history after discovery

def discover_method(
    *,
    llm_model: str,
    temperature: float,
    client: Any,
    prompts: Dict[str, List[str]],
    probabilities: Dict[str, float],
    reference_filename: str,
    initialisations: List[Dict[str, Any]],
    n_generations: int,
    children_per_model: int,
    bug_retries: int,
    initial_bug_fix_attempts: int,
    timeout: int,
    n_feedback_buffer: int,
    skip_timeout: bool,
    drop_island_iter: int,
    prompt_samples: int,
    alter_info_samples: int,
    prompt_decay: float,
    hints: List[str],
    filename: str,
    use_initial: bool,
    TIME_LIMIT: int,
    task_description: str,
    val_limit: int,
    debug: bool,
    alter_info_prompts: Dict[str, Any] | None = None,
    history_path: str | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    best_model, full_history = optimizer.discover_algorithm(
        llm_model=llm_model,
        temp=temperature,
        client=client,
        prompts=prompts,
        probabilities=probabilities,
        reference_filename=reference_filename,
        initialisations=initialisations,
        n_generations=n_generations,
        children_per_model=children_per_model,
        bug_retries=bug_retries,
        initial_bug_fix_attempts=initial_bug_fix_attempts,
        timeout=timeout,
        n_feedback_buffer=n_feedback_buffer,
        skip_timeout=skip_timeout,
        alter_info_prompts=alter_info_prompts,
        drop_island_iter=drop_island_iter,
        prompt_samples=prompt_samples,
        alter_info_samples=alter_info_samples,
        prompt_decay=prompt_decay,
        hints=hints,
        filename=filename,
        use_initial=use_initial,
        TIME_LIMIT=TIME_LIMIT,
        task_description=task_description,
        val_limit=val_limit,
        debug=debug,
    )

    # Persist full_history immediately after discovery
    base = filename.rsplit(".", 1)[0] if filename else "model"
    out_json = Path(history_path) if history_path else Path(f"{base}_full_history.json")

    serializable: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for outer_k, outer_v in full_history.items():
        serializable[outer_k] = {}
        for inner_k, record_list in outer_v.items():
            serializable[outer_k][inner_k] = [
                {
                    "code": getattr(r, "code", None),
                    "file": str(getattr(r, "file", "")),
                    "accuracy": getattr(r, "accuracy", None),
                    "model_info": getattr(r, "model_info", None),
                    "lineage": getattr(r, "lineage", None),
                }
                for r in record_list
            ]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return best_model, full_history
