import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tusoai.llm import (
    GENERAL_CLIENT,
    GENERAL_MAX_TOKENS,
    GENERAL_MODEL,
    GENERAL_PROVIDER,
    GENERAL_REASONING_MODE,
    GENERAL_TEMP,
    GENERAL_THINKING,
    GENERAL_THINKING_TOKENS,
    Tusoai,
    run_prompt,
    run_prompt_full,
)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful, careful assistant. Answer the user's question using the PDF content. "
    "If the PDF does not contain the answer, say so."
)

def ask_pdf(
    pdf_path: str,
    question: str,
    *,
    client: Any = None,
    provider: str = None,
    model: str = None,
    tusoai: Optional[Tusoai] = None,
    temperature: float = None,
    max_tokens: int = 10000,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    max_pages: int = 15,
) -> Tuple[str, float]:
    """
    Sends a PDF + question to the configured LLM and returns (answer_text, cost_usd).

    Requires your existing helpers:
      - _estimate_cost_usd(...)
      - _extract_claude_text(...)
      - _safe_get(...)
    """

    if tusoai is not None:
        client = tusoai.client
        provider = tusoai.provider
        model = model or tusoai.pdf_model
        if temperature is None:
            temperature = tusoai.temperature
    else:
        client = client or GENERAL_CLIENT
        provider = provider or GENERAL_PROVIDER
        model = model or GENERAL_MODEL
    temperature = GENERAL_TEMP if temperature is None else temperature

    if client is None:
        raise RuntimeError(
            "No LLM client configured. Call configure_default_client(...) "
            "or pass client/provider/model to ask_pdf()."
        )
    if not provider or not model:
        raise RuntimeError(
            "Missing provider/model configuration. Call configure_default_client(...) "
            "or pass provider/model to ask_pdf()."
        )

    import fitz

    page_limit = max(1, int(max_pages))
    text_parts: List[str] = []
    with fitz.open(pdf_path) as doc:
        pages_to_parse = min(page_limit, len(doc))
        for page_idx in range(pages_to_parse):
            page_text = doc[page_idx].get_text("text") or ""
            if page_text.strip():
                text_parts.append(page_text)
    pdf_text = "\n\n".join(text_parts).strip()
    if not pdf_text:
        raise RuntimeError(f"No extractable text found in PDF: {pdf_path}")

    # If a References section appears within the parsed pages, ignore it and everything after it.
    ref_match = re.search(r"(?im)^\s*references\s*$", pdf_text)
    if ref_match:
        pdf_text = pdf_text[: ref_match.start()].rstrip()
        if not pdf_text:
            raise RuntimeError(f"No extractable pre-references text found in PDF: {pdf_path}")

    prompt = (
        f"{system_message}\n\n"
        f"<paper_text>\n{pdf_text}\n</paper_text>\n\n"
        f"Question: {question}"
    )

    if tusoai is not None:
        thinking = tusoai.pdf_thinking
        thinking_tokens = tusoai.pdf_thinking_tokens
        reasoning_mode = tusoai.pdf_reasoning_mode
        prompt_max_tokens = max_tokens if max_tokens is not None else tusoai.max_tokens
    else:
        thinking = GENERAL_THINKING
        thinking_tokens = GENERAL_THINKING_TOKENS
        reasoning_mode = GENERAL_REASONING_MODE
        prompt_max_tokens = max_tokens if max_tokens is not None else GENERAL_MAX_TOKENS

    return run_prompt_full(
        prompt=prompt,
        client=client,
        temperature=temperature,
        model=model,
        provider=provider,
        max_tokens=prompt_max_tokens,
        thinking=thinking,
        thinking_tokens=thinking_tokens,
        reasoning_mode=reasoning_mode,
    )

# --------------------------------------------------------------------------------------
# Improved, structured prompt for extracting reusable methodological info from papers
# --------------------------------------------------------------------------------------

def build_summary_question(task_description: str) -> str:
    return (
        f"<role>\n"
        f"You are an expert scientific method developer and senior ML engineer.\n"
        f"You extract actionable methodological details from papers to help build methods under an external evaluation harness.\n"
        f"</role>\n\n"
        f"<task>\n"
        f"Extract technical and methodological information from this paper that is useful for:\n"
        f"{task_description}\n"
        f"</task>\n\n"
        f"<instructions>\n"
        f"Read the entire paper carefully.\n"
        f"As you progress, keep ADDING new bullets and REVISING existing bullets to maintain one consolidated, non-redundant list.\n"
        f"Do NOT guess. If a detail is not explicitly stated, omit it.\n"
        f"</instructions>\n\n"
        f"<what_to_extract>\n"
        f"Capture implementation-relevant details, especially anything you could reuse directly in a method:\n"
        f"- precise algorithm/procedure steps (pipelines, decision rules, thresholds, heuristics)\n"
        f"- model/architecture details (components, dimensions, objectives, losses, regularization)\n"
        f"- optimization/training recipe (hyperparameters, schedules, batch sizes, stopping, compute)\n"
        f"- data details (sources, sizes, preprocessing, labels, splits, augmentation)\n"
        f"- evaluation setup (metrics, baselines, ablations, error analysis)\n"
        f"- exact experimental settings and numeric values (when present)\n"
        f"- code-like information (pseudocode-level steps, equations described in words)\n"
        f"</what_to_extract>\n\n"
        f"<avoid>\n"
        f"- high-level background/motivation\n"
        f"- results-only summaries without method detail\n"
        f"- generic claims without specifics\n"
        f"- prose paragraphs\n"
        f"</avoid>\n\n"
        f"<output_format>\n"
        f"Output EXACTLY two sections:\n\n"
        f"<plan>\n"
        f"- Briefly state (3-7 lines) what types of details you will extract for this task.\n"
        f"</plan>\n\n"
        f"BEGIN_BULLETS\n"
        f"- [bullet 1]\n"
        f"- [bullet 2]\n"
        f"...\n"
        f"END_BULLETS\n"
        f"</output_format>\n\n"
        f"<verification>\n"
        f"Before finalizing:\n"
        f"- Are all bullets concrete and implementation-relevant?\n"
        f"- Did you avoid guessing?\n"
        f"- Did you keep only method/data/eval details relevant to the task?\n"
        f"</verification>\n"
    )


_BULLET_BLOCK_RE = re.compile(r"BEGIN_BULLETS\s*(.*?)\s*END_BULLETS", flags=re.S)
_USEFUL_TAG_RE = re.compile(r"<a>\s*(Yes|No)\s*</a>", flags=re.I)


def _extract_bullets_from_reply(reply: str) -> str:
    """
    Extract bullet list between BEGIN_BULLETS ... END_BULLETS if present, else return raw reply.
    Keeps backwards compatibility with any existing parsers downstream.
    """
    if not reply:
        return ""
    m = _BULLET_BLOCK_RE.search(reply)
    if m:
        return (m.group(1) or "").strip()
    return reply.strip()


def _is_useful_summary(summary: str, task_description: str, data_available: str) -> Tuple[bool, float]:
    """
    Ask the LLM if the extracted summary contains useful info for the task, given available data.
    Returns (is_useful, cost_usd).
    Expects LLM output: <a>Yes</a> or <a>No</a>
    """
    check_prompt = f"""
<role>
You are an expert scientific method developer and senior ML engineer.
You are filtering paper-extracted technical notes for an AutoML/agentic optimization pipeline.
</role>

<task>
Decide whether the extracted bullets contain useful, actionable methodological information for the task,
given the available data. The goal is to keep only summaries that will meaningfully inform method design.
</task>

<context>
Task:
{task_description}

Available data:
{data_available}
</context>

<extracted_bullets>
{summary}
</extracted_bullets>

<decision_criteria>
Answer <a>Yes</a> ONLY if the bullets include at least ONE of the following:
- a concrete algorithm/procedure we could implement or adapt
- specific modeling/training/evaluation settings that materially guide implementation
- data processing/labeling/splitting details that directly affect how we should build the method
- ablation/diagnostic insights that clarify which components matter and how to optimize next

Answer <a>No</a> if the bullets are mostly generic, high-level, results-only, or not actionable for this task/data.
</decision_criteria>

<reasoning_process>
Think step by step:
1) Identify the most actionable bullets (if any)
2) Check compatibility with the available data setting
3) Decide whether this will reduce search/iteration time in the next optimization generation
</reasoning_process>

<output_format>
Output EXACTLY one tag and nothing else:
<a>Yes</a>
or
<a>No</a>
</output_format>
""".strip()

    reply = run_prompt(check_prompt)
    if isinstance(reply, tuple):
        reply_text, check_cost = reply
    else:
        reply_text, check_cost = reply, 0.0

    match = _USEFUL_TAG_RE.findall(reply_text or "")
    if not match:
        # If the model fails formatting, be conservative and keep it
        return True, float(check_cost or 0.0)

    ans = match[0].strip().lower()
    return (ans == "yes"), float(check_cost or 0.0)


def summarize_papers_single(
    function_name: str,
    papers: List[Dict[str, Any]],
    *,
    tusoai: Optional[Tusoai] = None,
    task_description: str,
    data_available: str,
    cache_dir: str,
    clear: bool = True,
    question: Optional[str] = None,
    max_tokens: int = 10000,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    stop_on_error: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    If folder/summaries.json exists and clear=False, loads and returns it.
    If clear=True, deletes folder/summaries.json (if present) and rebuilds summaries.

    Changes requested:
    - Only one friendly line per PDF with (X/N) where X is retained summaries count.
    - Run a secondary usefulness check; discard summaries judged <a>No</a>.
    - Include the cost of the usefulness check in total_cost.
    - Print total_cost at the end.

    Returns: (summaries, total_cost)
    """
    summaries_path = Path(cache_dir) / f"kt_data/{function_name}/summaries.json"

    if clear and summaries_path.exists():
        try:
            summaries_path.unlink()
            print(f"[CLEAR] Removed cache: {summaries_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {summaries_path}: {e}")

    if (not clear) and summaries_path.exists():
        print(f"[CACHE] Loading summaries from {summaries_path}")
        summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
        return summaries, 0.0

    summaries: List[Dict[str, Any]] = []
    total_cost = 0.0
    target_n = len(papers)  # for (X/N) display, N = number of PDFs we attempt

    for i, p in enumerate(papers, start=1):
        title = p.get("title", "Untitled")
        pdf_path = p.get("pdf_path") or p.get("local_pdf")

        if not pdf_path or not Path(pdf_path).exists():
            print(f"Skipped {title}.pdf ({len(summaries)}/{target_n})")
            continue

        try:
            question_text = question or build_summary_question(task_description)
            raw_reply, cost = ask_pdf(
                pdf_path=pdf_path,
                question=question_text,
                tusoai=tusoai,
                max_tokens=max_tokens,
                system_message=system_message,
            )
        except Exception as e:
            print(
                f"Failed to summarize {title}.pdf ({len(summaries)}/{target_n}) "
                f"- {type(e).__name__}: {e}"
            )
            if stop_on_error:
                raise
            continue

        total_cost += float(cost or 0.0)

        # Extract just the bullets block for storage/usefulness check
        bullets = _extract_bullets_from_reply(raw_reply)

        # Secondary usefulness check (adds cost)
        useful, check_cost = _is_useful_summary(bullets, task_description, data_available)
        total_cost += float(check_cost or 0.0)

        if not useful:
            print(f"Discarded {title}.pdf ({len(summaries)}/{target_n})")
            continue

        summaries.append(
            {
                "paperId": p.get("paperId"),
                "title": title,
                "year": p.get("year"),
                "venue": p.get("venue"),
                "semanticScholarUrl": p.get("semanticScholarUrl"),
                "openAccessPdfUrl": p.get("openAccessPdfUrl"),
                "pdf_path": pdf_path,
                "summary": bullets,
                "cost_usd": float(cost or 0.0),
                "provider": GENERAL_PROVIDER,
                "model": GENERAL_MODEL,
            }
        )

        print(f"Kept {title}.pdf ({len(summaries)}/{target_n})")

    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    summaries_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[SAVED] Summaries written to {summaries_path}")

    print(f"Total cost: ${total_cost:.6f}")
    return summaries, total_cost

def summarize_papers(
    function_name: str,
    papers: List[Dict[str, Any]],
    *,
    tusoai: Optional[Tusoai] = None,
    task_description: str,
    data_available: str,
    cache_dir: str,
    clear: bool = True,
    question: Optional[str] = None,
    max_tokens: int = 10000,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    stop_on_error: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Threaded version of summarize_papers (one thread per paper).

    Preserved behaviors:
      - If cache exists and clear=False, loads and returns it (cost=0.0).
      - If clear=True, deletes cache (if present) and rebuilds.
      - Secondary usefulness check; discard if not useful (still counts cost).
      - Prints per-paper status lines with (X/N) where X is retained/kept count.
      - Prints total cost at the end.
    """
    summaries_path = Path(cache_dir) / f"kt_data/{function_name}/summaries.json"

    if clear and summaries_path.exists():
        try:
            summaries_path.unlink()
            print(f"[CLEAR] Removed cache: {summaries_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {summaries_path}: {e}")

    if (not clear) and summaries_path.exists():
        print(f"[CACHE] Loading summaries from {summaries_path}")
        summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
        return summaries, 0.0

    target_n = len(papers)  # N in (X/N)
    summaries: List[Dict[str, Any]] = []
    total_cost = 0.0

    def worker(p: Dict[str, Any]) -> Dict[str, Any]:
        title = p.get("title", "Untitled")
        pdf_path = p.get("pdf_path") or p.get("local_pdf")
        if not pdf_path or not Path(pdf_path).exists():
            return {"status": "skipped", "title": title, "cost": 0.0}

        try:
            question_text = question or build_summary_question(task_description)
            raw_reply, cost = ask_pdf(
                pdf_path=pdf_path,
                question=question_text,
                tusoai=tusoai,
                max_tokens=max_tokens,
                system_message=system_message,
            )
        except Exception as e:
            return {
                "status": "failed",
                "title": title,
                "cost": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }

        total_cost_local = float(cost or 0.0)

        # IMPORTANT: Extract bullet block (BEGIN_BULLETS ... END_BULLETS) for usefulness + storage
        bullets = _extract_bullets_from_reply(raw_reply)

        useful, check_cost = _is_useful_summary(bullets, task_description, data_available)
        total_cost_local += float(check_cost or 0.0)

        if not useful:
            return {"status": "discarded", "title": title, "cost": total_cost_local}

        return {
            "status": "kept",
            "title": title,
            "cost": total_cost_local,
            "result": {
                "paperId": p.get("paperId"),
                "title": title,
                "year": p.get("year"),
                "venue": p.get("venue"),
                "semanticScholarUrl": p.get("semanticScholarUrl"),
                "openAccessPdfUrl": p.get("openAccessPdfUrl"),
                "pdf_path": pdf_path,
                "summary": bullets,              # store bullets only (matches single-threaded)
                "cost_usd": float(cost or 0.0),  # PDF summarization cost only
                "provider": GENERAL_PROVIDER,
                "model": GENERAL_MODEL,
            },
        }

    with ThreadPoolExecutor(max_workers=max(1, len(papers))) as ex:
        future_to_paper = {ex.submit(worker, p): p for p in papers}

        for fut in as_completed(future_to_paper):
            out = fut.result()
            status = out.get("status")
            title = out.get("title", "Untitled")

            if status == "skipped":
                print(f"Skipped {title}.pdf ({len(summaries)}/{target_n})")
                continue

            if status == "failed":
                print(f"Failed to summarize {title}.pdf ({len(summaries)}/{target_n})")
                if stop_on_error:
                    raise RuntimeError(out.get("error") or "Worker failed")
                continue

            # Discarded / kept both incur cost
            total_cost += float(out.get("cost") or 0.0)

            if status == "discarded":
                print(f"Discarded {title}.pdf ({len(summaries)}/{target_n})")
                continue

            if status == "kept":
                summaries.append(out["result"])
                print(f"Kept {title}.pdf ({len(summaries)}/{target_n})")
                continue

            # Unexpected status: don't crash unless requested
            print(f"[WARN] Unknown status for {title}: {status}")
            if stop_on_error:
                raise RuntimeError(f"Unknown worker status: {status}")

    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    summaries_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[SAVED] Summaries written to {summaries_path}")

    print(f"Total cost: ${total_cost:.6f}")
    return summaries, total_cost
