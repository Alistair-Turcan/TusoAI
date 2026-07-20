from __future__ import annotations

import io
import json
import os
import hashlib
from contextlib import redirect_stderr, redirect_stdout
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tusoai import categories as categories_module
from tusoai import data_preview as data_preview_module
from tusoai import initializations as initializations_module
from tusoai import instructions as instructions_module
from tusoai import probabilities as probabilities_module
from tusoai.categories import (
    get_task_categories,
    refine_categories_with_summaries,
    retain_top_categories,
)
from tusoai.llm import Tusoai
from tusoai.llm import run_prompt as base_run_prompt
from tusoai.data_preview import (
    gather_data_usage_instructions,
    infer_read_cmd_from_data_path,
    iterative_desc_from_read_command,
)
from tusoai.initializations import (
    get_initializations,
    refine_initializations_with_summaries,
    retain_top_initial_solutions,
)
from tusoai.instructions import (
    clean_refined_prompts_to_instructions,
    generate_prompts_for_categories,
    generate_prompts_from_summaries,
)
from tusoai.literature import MAX_PAPER_SEARCHES, run_download_top_pdfs
from tusoai.probabilities import get_category_probabilities
from tusoai.prompts import add_general_prompts_and_probability
from tusoai.summarize import summarize_papers
from tusoai.fs_utils import copytree_portable
from tusoai.optimization import DataTask, MethodTask


def _hashed_task_cache_dir(cache_dir: str, task_description: str, data_available: str) -> Path:
    key = f"{task_description}\n<data_available>\n{data_available}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / h


def _reuse_hashed_method_cache_if_available(scoped_cache_dir: Path, function_name: str) -> bool:
    kt_root = scoped_cache_dir / "kt_data"
    target_dir = kt_root / function_name
    if target_dir.exists():
        return True
    if not kt_root.exists():
        return False

    required = {"summaries.json", "final_instructions.json", "final_solutions.json", "probabilities.json"}
    for cand in kt_root.iterdir():
        if not cand.is_dir() or cand.name == function_name:
            continue
        present = {p.name for p in cand.iterdir() if p.is_file()}
        if required.issubset(present):
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            copytree_portable(cand, target_dir)
            return True
    return False


def _semantic_scholar_query_from_task(task_description: str) -> Tuple[str, float]:
    """Convert a verbose task description into a short Semantic Scholar query."""
    fallback = str(task_description or "").strip()
    prompt = f"""You are preparing a Semantic Scholar paper-search query.

Rewrite the task description as a short, clean research-topic query that Semantic Scholar can search well.
Use examples like: enhancer-gene linking, protein structure prediction, drug response prediction.

Rules:
- Output only the query text.
- Keep it concise: ideally 2-6 words, never more than 8 words.
- Prefer a domain task phrase over implementation details, benchmarks, runner text, file paths, metrics, or data split details.

Task description:
{fallback}
""".strip()
    try:
        reply, cost = base_run_prompt(prompt)
    except Exception:
        return fallback, 0.0

    text = str(reply or "").strip()
    text = text.replace("```", "").strip()
    text = text.splitlines()[0].strip() if text else ""
    text = text.strip(" \t\r\n'\"`.,;:")
    text = " ".join(text.split())

    # Treat non-concise or clearly malformed responses as parse failures.
    if not text or len(text) > 120 or len(text.split()) > 8 or "<" in text or "{" in text:
        return fallback, float(cost or 0.0)
    return text, float(cost or 0.0)


@contextmanager
def _track_prompt_cost(modules: List[object]):
    total_cost = {"value": 0.0}

    def _tracked_run_prompt(message: str):
        reply, cost = base_run_prompt(message)
        total_cost["value"] += float(cost or 0.0)
        return reply, cost

    originals = {}
    for mod in modules:
        originals[mod] = getattr(mod, "run_prompt", None)
        setattr(mod, "run_prompt", _tracked_run_prompt)

    try:
        yield total_cost
    finally:
        for mod, orig in originals.items():
            if orig is None:
                continue
            setattr(mod, "run_prompt", orig)


def create_data_subtask(
    function_name: str,
    task_description: str,
    file_description: str,
    data_usage: str,
    read_cmd: Optional[str] = None,
    data_path: Optional[str] = None,
    *,
    tusoai: Tusoai | None = None,
    cache_dir: str,
    data_instruction_count: int = 15,
    clear: bool = False,
    hints: Optional[List[str]] = None,
    source_path: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Tuple[DataTask, float]:
    print(f"Creating data instructions for {function_name}")
    scoped_cache_dir = _hashed_task_cache_dir(cache_dir, task_description, data_usage)

    task_path = scoped_cache_dir / f"kt_data/{function_name}/data_task.json"
    stats_path = scoped_cache_dir / f"kt_data/{function_name}/data_stats.json"

    if clear:
        if task_path.exists():
            task_path.unlink()
        if stats_path.exists():
            stats_path.unlink()

    if not clear and task_path.exists():
        cached = json.loads(task_path.read_text(encoding="utf-8"))
        return (
            DataTask(
                function_name=cached.get("function_name", function_name),
                knowledge_tree=cached.get("knowledge_tree", {"instructions": []}),
                kt_prob=cached.get("kt_prob", {"instructions": 1.0}),
                hints=list(cached.get("hints", [])),
                data_summary=cached.get("data_summary", ""),
                data_usage=cached.get("data_usage", data_usage),
                file_description=cached.get("file_description", file_description),
                read_cmd=cached.get("read_cmd", read_cmd or ""),
                source_path=cached.get("source_path", source_path),
                repo_root=cached.get("repo_root", repo_root),
            ),
            0.0,
        )

    total_cost = 0.0

    if not read_cmd:
        if not data_path:
            raise ValueError("create_data_subtask requires either read_cmd or data_path")
        if tusoai is not None:
            with tusoai.use_construction_model():
                with _track_prompt_cost([data_preview_module]) as cost_ctx:
                    read_cmd = infer_read_cmd_from_data_path(data_path, iterations=5)
                    total_cost += float(cost_ctx["value"])
        else:
            with _track_prompt_cost([data_preview_module]) as cost_ctx:
                read_cmd = infer_read_cmd_from_data_path(data_path, iterations=5)
                total_cost += float(cost_ctx["value"])
    else:
        total_cost = 0.0

    if tusoai is not None:
        with tusoai.use_construction_model():
            with _track_prompt_cost([data_preview_module]) as cost_ctx:
                if stats_path.exists():
                    cached_stats = json.loads(stats_path.read_text(encoding="utf-8"))
                    data_stats = cached_stats.get("data_stats", "")
                else:
                    data_stats = iterative_desc_from_read_command(read_cmd, iterations=5)
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    stats_path.write_text(
                        json.dumps({"data_stats": data_stats}, indent=2),
                        encoding="utf-8",
                    )

                data_instructions = gather_data_usage_instructions(
                    function_name=function_name,
                    task_description=task_description,
                    file_description=file_description,
                    data_stats=data_stats,
                    data_usage=data_usage,
                    instruction_count=data_instruction_count,
                    cache_dir=str(scoped_cache_dir),
                    clear=clear,
                )
                total_cost += float(cost_ctx["value"])
    else:
        with _track_prompt_cost([data_preview_module]) as cost_ctx:
            if stats_path.exists():
                cached_stats = json.loads(stats_path.read_text(encoding="utf-8"))
                data_stats = cached_stats.get("data_stats", "")
            else:
                data_stats = iterative_desc_from_read_command(read_cmd, iterations=5)
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                stats_path.write_text(
                    json.dumps({"data_stats": data_stats}, indent=2),
                    encoding="utf-8",
                )

            data_instructions = gather_data_usage_instructions(
                function_name=function_name,
                task_description=task_description,
                file_description=file_description,
                data_stats=data_stats,
                data_usage=data_usage,
                instruction_count=data_instruction_count,
                cache_dir=str(scoped_cache_dir),
                clear=clear,
            )
            total_cost += float(cost_ctx["value"])

    data_task = DataTask(
        function_name=function_name,
        knowledge_tree={"instructions": data_instructions},
        kt_prob={"instructions": 1.0},
        hints=list(hints or []),
        data_summary=data_stats,
        data_usage=data_usage,
        file_description=file_description,
        read_cmd=read_cmd or "",
        source_path=source_path,
        repo_root=repo_root,
    )
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(
        json.dumps(
            {
                "function_name": data_task.function_name,
                "knowledge_tree": data_task.knowledge_tree,
                "kt_prob": data_task.kt_prob,
                "hints": data_task.hints,
                "data_summary": data_task.data_summary,
                "data_usage": data_task.data_usage,
                "file_description": data_task.file_description,
                "read_cmd": data_task.read_cmd,
                "data_path": data_path,
                "source_path": data_task.source_path,
                "repo_root": data_task.repo_root,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return data_task, total_cost


def create_method_subtask(
    function_name: str,
    task_description: str,
    data_available: str,
    *,
    tusoai: Tusoai | None = None,
    cache_dir: str,
    num_cat: int = 10,
    instruction_count: int = 10,
    num_init: int = 10,
    paper_searches: int = 10,
    info_per_paper: int = 10,
    clear: bool = False,
    semantic_scholar_api_key: str | None = None,
    hints: Optional[List[str]] = None,
    use_initial: bool = True,
    source_path: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Tuple[MethodTask, float]:
    print(f"[METHOD] Building method task for {function_name}")
    if paper_searches > MAX_PAPER_SEARCHES:
        print(
            f"[METHOD] Capping paper_searches from {paper_searches} "
            f"to {MAX_PAPER_SEARCHES}"
        )
        paper_searches = MAX_PAPER_SEARCHES
    scoped_cache_dir = _hashed_task_cache_dir(cache_dir, task_description, data_available)
    if not clear:
        reused = _reuse_hashed_method_cache_if_available(scoped_cache_dir, function_name)
        if reused:
            print(f"[METHOD] Reusing hashed kt_data cache for {function_name}")

    def _quiet(label: str, fn):
        print(f"[METHOD] {label}...")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return fn()

    api_key = semantic_scholar_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    total_cost = 0.0
    if tusoai is not None:
        tusoai.activate_construction()
    try:
        # Shared paper cache across tasks/functions for reuse.
        search_dir = Path(cache_dir) / "papers"
        pdf_dir = search_dir / "pdfs"
        selected_path = search_dir / "selected.json"
        existing_pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []

        semantic_query = task_description
        semantic_query_cost = 0.0
        semantic_query_prepared = False

        def _prepare_semantic_query() -> str:
            nonlocal semantic_query, semantic_query_cost, semantic_query_prepared, total_cost
            semantic_query, semantic_query_cost = _semantic_scholar_query_from_task(task_description)
            semantic_query_prepared = True
            total_cost += float(semantic_query_cost or 0.0)
            return semantic_query

        def _write_semantic_query_metadata() -> None:
            if not semantic_query_prepared:
                return
            search_dir.mkdir(parents=True, exist_ok=True)
            (search_dir / "semantic_query.json").write_text(
                json.dumps(
                    {
                        "function_name": function_name,
                        "task_description": task_description,
                        "semantic_scholar_query": semantic_query,
                        "fallback_used": semantic_query == task_description,
                        "cost": float(semantic_query_cost or 0.0),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if clear:
            papers = run_download_top_pdfs(
                function_name=function_name,
                task=_prepare_semantic_query(),
                folder=cache_dir,
                api_key=api_key,
                top_n=paper_searches,
                clear=True,
            )
        elif len(existing_pdfs) >= paper_searches:
            if selected_path.exists():
                cached = json.loads(selected_path.read_text(encoding="utf-8"))
                papers = []
                for paper in cached[:paper_searches]:
                    paper = dict(paper)
                    local_pdf = paper.get("local_pdf") or paper.get("pdf_path")
                    if not local_pdf:
                        continue
                    paper["local_pdf"] = local_pdf
                    paper["pdf_path"] = local_pdf
                    papers.append(paper)
            else:
                papers = [
                    {"title": path.stem, "local_pdf": str(path), "pdf_path": str(path)}
                    for path in existing_pdfs[:paper_searches]
                ]
        else:
            papers = run_download_top_pdfs(
                function_name=function_name,
                task=_prepare_semantic_query(),
                folder=cache_dir,
                api_key=api_key,
                top_n=paper_searches,
                clear=False,
            )
        _write_semantic_query_metadata()
        print(f"[METHOD] {len(papers)} PDFs ready")

        with _track_prompt_cost(
            [
                categories_module,
                instructions_module,
                initializations_module,
                probabilities_module,
            ]
        ) as cost_ctx:
            summaries, summarize_cost = _quiet(
                "Summarizing papers",
                lambda: summarize_papers(
                    function_name=function_name,
                    papers=papers,
                    clear=clear,
                    tusoai=tusoai,
                    task_description=task_description,
                    data_available=data_available,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            initial_categories = _quiet(
                "Generating initial categories",
                lambda: get_task_categories(
                    function_name=function_name,
                    task_description=task_description,
                    data_available=data_available,
                    num_cat=num_cat,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            refined_categories = _quiet(
                "Refining categories with summaries",
                lambda: refine_categories_with_summaries(
                    function_name=function_name,
                    categories=initial_categories,
                    summaries=summaries,
                    task_description=task_description,
                    data_available=data_available,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            final_categories = _quiet(
                "Selecting top categories",
                lambda: retain_top_categories(
                    function_name=function_name,
                    refined_categories=refined_categories,
                    task_description=task_description,
                    data_available=data_available,
                    num_cat=num_cat,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            initial_prompts = _quiet(
                "Generating initial prompts",
                lambda: generate_prompts_for_categories(
                    function_name=function_name,
                    task_description=task_description,
                    data_available=data_available,
                    categories=final_categories,
                    to_generate=instruction_count,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            refined_prompts = _quiet(
                "Refining prompts with summaries",
                lambda: generate_prompts_from_summaries(
                    function_name=function_name,
                    task_description=task_description,
                    summaries=summaries,
                    categories=final_categories,
                    data_available=data_available,
                    existing_prompts=initial_prompts,
                    info_per_paper=info_per_paper,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            final_instructions = _quiet(
                "Finalizing instructions",
                lambda: clean_refined_prompts_to_instructions(
                    function_name=function_name,
                    refined_prompts=refined_prompts,
                    task_description=task_description,
                    data_available=data_available,
                    instruction_count=instruction_count,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            initial_solutions = _quiet(
                "Generating initial solutions",
                lambda: get_initializations(
                    function_name=function_name,
                    task_description=task_description,
                    data_available=data_available,
                    num_init=num_init,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            refined_solutions = _quiet(
                "Refining solutions with summaries",
                lambda: refine_initializations_with_summaries(
                    function_name=function_name,
                    initializations=initial_solutions,
                    summaries=summaries,
                    task_description=task_description,
                    data_available=data_available,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            final_solutions = _quiet(
                "Selecting top initial solutions",
                lambda: retain_top_initial_solutions(
                    function_name=function_name,
                    refined_solutions=refined_solutions,
                    task_description=task_description,
                    data_available=data_available,
                    num_init=num_init,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            probabilities = _quiet(
                "Scoring category probabilities",
                lambda: get_category_probabilities(
                    function_name=function_name,
                    task_description=task_description,
                    data_available=data_available,
                    categories=final_categories,
                    initializations=final_solutions,
                    max_attempts=3,
                    clear=clear,
                    cache_dir=str(scoped_cache_dir),
                ),
            )

            knowledge_tree, kt_prob = _quiet(
                "Finalizing knowledge tree",
                lambda: add_general_prompts_and_probability(
                    refined_prompts=refined_prompts,
                    probabilities=probabilities,
                ),
            )
            total_cost += float(cost_ctx["value"]) + float(summarize_cost or 0.0)
    finally:
        if tusoai is not None:
            tusoai.deactivate()

    return (
        MethodTask(
            function_name=function_name,
            knowledge_tree=knowledge_tree,
            kt_prob=kt_prob,
            hints=list(hints or []),
            initial_solutions=final_solutions,
            use_initial=use_initial,
            source_path=source_path,
            repo_root=repo_root,
        ),
        total_cost,
    )


__all__ = [
    "create_data_subtask",
    "create_method_subtask",
]
